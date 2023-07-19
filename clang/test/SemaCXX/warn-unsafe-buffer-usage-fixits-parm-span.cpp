// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage -fdiagnostics-parseable-fixits -fsafe-buffer-usage-suggestions -include %s %s 2>&1 | FileCheck %s

// TODO test if there's not a single character in the file after a decl or def

#ifndef INCLUDE_ME
#define INCLUDE_ME

void simple(int *p);
// CHECK-DAG: fix-it:{{.*}}:{[[@LINE-1]]:1-[[@LINE-1]]:1}:"{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} "
// CHECK-DAG: fix-it:{{.*}}:{[[@LINE-2]]:20-[[@LINE-2]]:20}:";\nvoid simple(std::span<int> p)"

#else

void simple(int *);
// CHECK-DAG: fix-it:{{.*}}:{[[@LINE-1]]:1-[[@LINE-1]]:1}:"{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} "
// CHECK-DAG: fix-it:{{.*}}:{[[@LINE-2]]:19-[[@LINE-2]]:19}:";\nvoid simple(std::span<int>)"

void simple(int *p) {
  // CHECK-DAG: fix-it:{{.*}}:{[[@LINE-1]]:13-[[@LINE-1]]:19}:"std::span<int> p"
  int tmp;
  tmp = p[5];
}
// CHECK-DAG: fix-it:{{.*}}:{[[@LINE-1]]:2-[[@LINE-1]]:2}:"\n{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} void simple(int *p) {return simple(std::span<int>(p, <# size #>));}\n"


void twoParms(int *p, int * q) {
  // CHECK-DAG: fix-it:{{.*}}:{[[@LINE-1]]:15-[[@LINE-1]]:21}:"std::span<int> p"
  // CHECK-DAG: fix-it:{{.*}}:{[[@LINE-2]]:23-[[@LINE-2]]:30}:"std::span<int> q"
  int tmp;
  tmp = p[5] + q[5];
}
// CHECK-DAG: fix-it:{{.*}}:{[[@LINE-1]]:2-[[@LINE-1]]:2}:"\n{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} void twoParms(int *p, int * q) {return twoParms(std::span<int>(p, <# size #>), q);}\n"
// CHECK-DAG: fix-it:{{.*}}:{[[@LINE-2]]:2-[[@LINE-2]]:2}:"\n{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} void twoParms(int *p, int * q) {return twoParms(p, std::span<int>(q, <# size #>));}\n"

void ptrToConst(const int * x) {
  // CHECK-DAG: fix-it:{{.*}}:{[[@LINE-1]]:17-[[@LINE-1]]:30}:"std::span<int const> x"
  int tmp = x[5];
}
// CHECK-DAG: fix-it:{{.*}}:{[[@LINE-1]]:2-[[@LINE-1]]:2}:"\n{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} void ptrToConst(const int * x) {return ptrToConst(std::span<int const>(x, <# size #>));}\n"

// The followings test cases where multiple FileIDs maybe involved
// when the analyzer loads characters from source files.

#define FUN_NAME(x) _##x##_

// The analyzer reads `void FUNNAME(macro_defined_name)(` from the
// source file.  The MACRO and this source file have different
// FileIDs.
void FUN_NAME(macro_defined_name)(int * x) {
  // CHECK-DAG: fix-it:{{.*}}:{[[@LINE-1]]:35-[[@LINE-1]]:42}:"std::span<int> x"
  int tmp = x[5];
}
// CHECK-DAG: fix-it:{{.*}}:{[[@LINE-1]]:2-[[@LINE-1]]:2}:"\n{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} void FUN_NAME(macro_defined_name)(int * x) {return FUN_NAME(macro_defined_name)(std::span<int>(x, <# size #>));}\n"


// The followings test various type specifiers
namespace {
  void simpleSpecifier(unsigned long long int *p) {
    // CHECK-DAG: fix-it:{{.*}}:{[[@LINE-1]]:24-[[@LINE-1]]:49}:"std::span<unsigned long long int> p"
    auto tmp = p[5];
  }
  // CHECK-DAG: fix-it:{{.*}}:{[[@LINE-1]]:4-[[@LINE-1]]:4}:"\n{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} void simpleSpecifier(unsigned long long int *p) {return simpleSpecifier(std::span<unsigned long long int>(p, <# size #>));}\n"

  void attrParm([[maybe_unused]] int * p) {
    // CHECK-DAG: fix-it:{{.*}}:{[[@LINE-1]]:34-[[@LINE-1]]:41}:"std::span<int> p"
    int tmp = p[5];
  }
  // CHECK-DAG: fix-it:{{.*}}:{[[@LINE-1]]:4-[[@LINE-1]]:4}:"\n{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} void attrParm({{\[}}{{\[}}maybe_unused{{\]}}{{\]}} int * p) {return attrParm(std::span<int>(p, <# size #>));}\n"

  using T = unsigned long long int;

  void usingTypenameSpecifier(T * p) {
    // CHECK-DAG: fix-it:{{.*}}:{[[@LINE-1]]:31-[[@LINE-1]]:36}:"std::span<T> p"
    int tmp = p[5];
  }
  // CHECK-DAG: fix-it:{{.*}}:{[[@LINE-1]]:4-[[@LINE-1]]:4}:"\n{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} void usingTypenameSpecifier(T * p) {return usingTypenameSpecifier(std::span<T>(p, <# size #>));}\n"

  typedef unsigned long long int T2;

  void typedefSpecifier(T2 * p) {
    // CHECK-DAG: fix-it:{{.*}}:{[[@LINE-1]]:25-[[@LINE-1]]:31}:"std::span<T2> p"
    int tmp = p[5];
  }
  // CHECK-DAG: fix-it:{{.*}}:{[[@LINE-1]]:4-[[@LINE-1]]:4}:"\n{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} void typedefSpecifier(T2 * p) {return typedefSpecifier(std::span<T2>(p, <# size #>));}\n"

  class SomeClass {
  } C;

  void classTypeSpecifier(const class SomeClass * p) {
    // CHECK-DAG: fix-it:{{.*}}:{[[@LINE-1]]:27-[[@LINE-1]]:52}:"std::span<class SomeClass const> p"
    if (++p) {}
  }
  // CHECK-DAG: fix-it:{{.*}}:{[[@LINE-1]]:4-[[@LINE-1]]:4}:"\n{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} void classTypeSpecifier(const class SomeClass * p) {return classTypeSpecifier(std::span<class SomeClass const>(p, <# size #>));}\n"

  struct {
    // anon
  } ANON_S;

  struct MyStruct {
    // namned
  } NAMED_S;

  // FIXME: `decltype(ANON_S)` represents an unnamed type but it can
  // be referred as "`decltype(ANON_S)`", so the analysis should
  // fix-it.
  void decltypeSpecifier(decltype(C) * p, decltype(ANON_S) * q, decltype(NAMED_S) * r,
                         decltype(NAMED_S) ** rr) {
    // CHECK-DAG: fix-it:{{.*}}:{[[@LINE-2]]:26-[[@LINE-2]]:41}:"std::span<decltype(C)> p"
    // CHECK-DAG: fix-it:{{.*}}:{[[@LINE-3]]:65-[[@LINE-3]]:86}:"std::span<decltype(NAMED_S)> r"
    // CHECK-DAG: fix-it:{{.*}}:{[[@LINE-3]]:26-[[@LINE-3]]:49}:"std::span<decltype(NAMED_S) *> rr"
    if (++p) {}
    if (++q) {}
    if (++r) {}
    if (++rr) {}
  }
  // CHECK-DAG: fix-it:{{.*}}:{[[@LINE-1]]:4-[[@LINE-1]]:4}:"\n{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} void decltypeSpecifier(decltype(C) * p, decltype(ANON_S) * q, decltype(NAMED_S) * r,\n{{.*}}decltype(NAMED_S) ** rr) {return decltypeSpecifier(std::span<decltype(C)>(p, <# size #>), q, r, rr);}\n
  // CHECK-DAG: fix-it:{{.*}}:{[[@LINE-2]]:4-[[@LINE-2]]:4}:"\n{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} void decltypeSpecifier(decltype(C) * p, decltype(ANON_S) * q, decltype(NAMED_S) * r,\n{{.*}}decltype(NAMED_S) ** rr) {return decltypeSpecifier(p, q, std::span<decltype(NAMED_S)>(r, <# size #>), rr);}\n"
  // CHECK-DAG: fix-it:{{.*}}:{[[@LINE-3]]:4-[[@LINE-3]]:4}:"\n{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} void decltypeSpecifier(decltype(C) * p, decltype(ANON_S) * q, decltype(NAMED_S) * r,\n{{.*}}decltype(NAMED_S) ** rr) {return decltypeSpecifier(p, q, r, std::span<decltype(NAMED_S) *>(rr, <# size #>));}\n"

#define MACRO_TYPE(T) long T

  void macroType(unsigned MACRO_TYPE(int) * p, unsigned MACRO_TYPE(long) * q) {
    // CHECK-DAG: fix-it:{{.*}}:{[[@LINE-1]]:18-[[@LINE-1]]:46}:"std::span<unsigned MACRO_TYPE(int)> p"
    // CHECK-DAG: fix-it:{{.*}}:{[[@LINE-2]]:48-[[@LINE-2]]:77}:"std::span<unsigned MACRO_TYPE(long)> q"
    int tmp = p[5];
    tmp = q[5];
  }
  // CHECK-DAG: fix-it:{{.*}}:{[[@LINE-1]]:4-[[@LINE-1]]:4}:"\n{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} void macroType(unsigned MACRO_TYPE(int) * p, unsigned MACRO_TYPE(long) * q) {return macroType(std::span<unsigned MACRO_TYPE(int)>(p, <# size #>), q);}\n"
  // CHECK-DAG: fix-it:{{.*}}:{[[@LINE-2]]:4-[[@LINE-2]]:4}:"\n{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} void macroType(unsigned MACRO_TYPE(int) * p, unsigned MACRO_TYPE(long) * q) {return macroType(p, std::span<unsigned MACRO_TYPE(long)>(q, <# size #>));}\n"
}

// The followings test various declarators:
void decayedArray(int a[]) {
  // CHECK-DAG: fix-it:{{.*}}:{[[@LINE-1]]:19-[[@LINE-1]]:26}:"std::span<int> a"
  int tmp;
  tmp = a[5];
}
// CHECK-DAG: fix-it:{{.*}}:{[[@LINE-1]]:2-[[@LINE-1]]:2}:"\n{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} void decayedArray(int a[]) {return decayedArray(std::span<int>(a, <# size #>));}\n"

void decayedArrayOfArray(int a[10][10]) {
  // CHECK-NOT: fix-it:{{.*}}:{[[@LINE-1]]
  if (++a){}
}

void complexDeclarator(int * (*a[10])[10]) {
  // CHECK-NOT: fix-it:{{.*}}:{[[@LINE-1]]
  if (++a){}
}

// Make sure we do not generate fixes for the following cases:

#define MACRO_NAME MyName

void macroIdentifier(int *MACRO_NAME) { // The fix-it ends with a macro. It will be discarded due to overlap with macros.
  // CHECK-NOT: fix-it:{{.*}}:{[[@LINE-1]]
  if (++MyName){}
}

// CHECK-NOT: fix-it:{{.*}}:
void parmHasNoName(int *p, int *) { // cannot fix the function because there is one parameter has no name.
  p[5] = 5;
}

#endif
