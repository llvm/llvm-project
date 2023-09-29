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
// CHECK-DAG: fix-it:{{.*}}:{[[@LINE-1]]:2-[[@LINE-1]]:2}:"\n{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} void twoParms(int *p, int * q) {return twoParms(std::span<int>(p, <# size #>), std::span<int>(q, <# size #>));}\n"

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
  // As parameter `q` cannot be fixed, fixes to parameters are all being given up.
  void decltypeSpecifierAnon(decltype(C) * p, decltype(ANON_S) * q, decltype(NAMED_S) * r,
                             decltype(NAMED_S) ** rr) {
    // CHECK: fix-it:{{.*}}:{[[@LINE-2]]:30-[[@LINE-2]]:45}:"std::span<decltype(C)> p"
    // CHECK: fix-it:{{.*}}:{[[@LINE-3]]:47-[[@LINE-3]]:67}:"std::span<decltype(ANON_S)> q"
    // CHECK: fix-it:{{.*}}:{[[@LINE-4]]:69-[[@LINE-4]]:90}:"std::span<decltype(NAMED_S)> r"
    // CHECK: fix-it:{{.*}}:{[[@LINE-4]]:30-[[@LINE-4]]:53}:"std::span<decltype(NAMED_S) *> rr"
    if (++p) {}
    if (++q) {}
    if (++r) {}
    if (++rr) {}
  }
  // CHECK-DAG: fix-it:{{.*}}:{[[@LINE-1]]:4-[[@LINE-1]]:4}:"\n{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} void decltypeSpecifierAnon(decltype(C) * p, decltype(ANON_S) * q, decltype(NAMED_S) * r,\n decltype(NAMED_S) ** rr) {return decltypeSpecifierAnon(std::span<decltype(C)>(p, <# size #>), std::span<decltype(ANON_S)>(q, <# size #>), std::span<decltype(NAMED_S)>(r, <# size #>), std::span<decltype(NAMED_S) *>(rr, <# size #>));}\n

  void decltypeSpecifier(decltype(C) * p, decltype(NAMED_S) * r, decltype(NAMED_S) ** rr) {
    // CHECK-DAG: fix-it:{{.*}}:{[[@LINE-1]]:26-[[@LINE-1]]:41}:"std::span<decltype(C)> p"
    // CHECK-DAG: fix-it:{{.*}}:{[[@LINE-2]]:43-[[@LINE-2]]:64}:"std::span<decltype(NAMED_S)> r"
    // CHECK-DAG: fix-it:{{.*}}:{[[@LINE-3]]:66-[[@LINE-3]]:89}:"std::span<decltype(NAMED_S) *> rr"
    if (++p) {}
    if (++r) {}
    if (++rr) {}
  }
  // CHECK-DAG: fix-it:{{.*}}:{[[@LINE-1]]:4-[[@LINE-1]]:4}:"\n{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} void decltypeSpecifier(decltype(C) * p, decltype(NAMED_S) * r, decltype(NAMED_S) ** rr) {return decltypeSpecifier(std::span<decltype(C)>(p, <# size #>), std::span<decltype(NAMED_S)>(r, <# size #>), std::span<decltype(NAMED_S) *>(rr, <# size #>));}\n

#define MACRO_TYPE(T) long T

  void macroType(unsigned MACRO_TYPE(int) * p, unsigned MACRO_TYPE(long) * q) {
    // CHECK-DAG: fix-it:{{.*}}:{[[@LINE-1]]:18-[[@LINE-1]]:46}:"std::span<unsigned MACRO_TYPE(int)> p"
    // CHECK-DAG: fix-it:{{.*}}:{[[@LINE-2]]:48-[[@LINE-2]]:77}:"std::span<unsigned MACRO_TYPE(long)> q"
    int tmp = p[5];
    tmp = q[5];
  }
  // CHECK-DAG: fix-it:{{.*}}:{[[@LINE-1]]:4-[[@LINE-1]]:4}:"\n{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} void macroType(unsigned MACRO_TYPE(int) * p, unsigned MACRO_TYPE(long) * q) {return macroType(std::span<unsigned MACRO_TYPE(int)>(p, <# size #>), std::span<unsigned MACRO_TYPE(long)>(q, <# size #>));}\n"
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

// Tests parameters with cv-qualifiers

void const_ptr(int * const x) { // expected-warning{{'x' is an unsafe pointer used for buffer access}} expected-note{{change type of 'x' to 'std::span' to preserve bounds information}}
  // CHECK: fix-it:{{.*}}:{[[@LINE-1]]:16-[[@LINE-1]]:29}:"std::span<int> const x"
  int tmp = x[5]; // expected-note{{used in buffer access here}}
}
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:2-[[@LINE-1]]:2}:"\n{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} void const_ptr(int * const x) {return const_ptr(std::span<int>(x, <# size #>));}\n"

void const_ptr_to_const(const int * const x) {// expected-warning{{'x' is an unsafe pointer used for buffer access}} expected-note{{change type of 'x' to 'std::span' to preserve bounds information}}
  // CHECK: fix-it:{{.*}}:{[[@LINE-1]]:25-[[@LINE-1]]:44}:"std::span<int const> const x"
  int tmp = x[5]; // expected-note{{used in buffer access here}}
}
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:2-[[@LINE-1]]:2}:"\n{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} void const_ptr_to_const(const int * const x) {return const_ptr_to_const(std::span<int const>(x, <# size #>));}\n"

void const_volatile_ptr(int * const volatile x) { // expected-warning{{'x' is an unsafe pointer used for buffer access}} expected-note{{change type of 'x' to 'std::span' to preserve bounds information}}
  // CHECK: fix-it:{{.*}}:{[[@LINE-1]]:25-[[@LINE-1]]:47}:"std::span<int> const volatile x"
  int tmp = x[5]; // expected-note{{used in buffer access here}}
}
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:2-[[@LINE-1]]:2}:"\n{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} void const_volatile_ptr(int * const volatile x) {return const_volatile_ptr(std::span<int>(x, <# size #>));}\n"

void volatile_const_ptr(int * volatile const x) { // expected-warning{{'x' is an unsafe pointer used for buffer access}} expected-note{{change type of 'x' to 'std::span' to preserve bounds information}}
  // CHECK: fix-it:{{.*}}:{[[@LINE-1]]:25-[[@LINE-1]]:47}:"std::span<int> const volatile x"
  int tmp = x[5]; // expected-note{{used in buffer access here}}
}
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:2-[[@LINE-1]]:2}:"\n{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} void volatile_const_ptr(int * volatile const x) {return volatile_const_ptr(std::span<int>(x, <# size #>));}\n"

void const_volatile_ptr_to_const_volatile(const volatile int * const volatile x) {// expected-warning{{'x' is an unsafe pointer used for buffer access}} expected-note{{change type of 'x' to 'std::span' to preserve bounds information}}
  // CHECK: fix-it:{{.*}}:{[[@LINE-1]]:43-[[@LINE-1]]:80}:"std::span<int const volatile> const volatile x"
  int tmp = x[5]; // expected-note{{used in buffer access here}}
}
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:2-[[@LINE-1]]:2}:"\n{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} void const_volatile_ptr_to_const_volatile(const volatile int * const volatile x) {return const_volatile_ptr_to_const_volatile(std::span<int const volatile>(x, <# size #>));}\n"


// Test if function declaration specifiers are handled correctly:

static void static_f(int *p) {
  p[5] = 5;
}
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:2-[[@LINE-1]]:2}:"\n{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} static void static_f(int *p) {return static_f(std::span<int>(p, <# size #>));}\n"

static inline void static_inline_f(int *p) {
  p[5] = 5;
}
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:2-[[@LINE-1]]:2}:"\n{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} static inline void static_inline_f(int *p) {return static_inline_f(std::span<int>(p, <# size #>));}\n"

inline void static static_inline_f2(int *p) {
  p[5] = 5;
}
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:2-[[@LINE-1]]:2}:"\n{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} inline void static static_inline_f2(int *p) {return static_inline_f2(std::span<int>(p, <# size #>));}\n"


// Test when unnamed types are involved:

typedef struct {int x;} UNNAMED_STRUCT;
struct {int x;} VarOfUnnamedType;

void useUnnamedType(UNNAMED_STRUCT * p) {  // expected-warning{{'p' is an unsafe pointer used for buffer access}} \
                                              expected-note{{change type of 'p' to 'std::span' to preserve bounds information}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:21-[[@LINE-2]]:39}:"std::span<UNNAMED_STRUCT> p"
  if (++p) {  // expected-note{{used in pointer arithmetic here}}
    // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:7-[[@LINE-1]]:10}:"(p = p.subspan(1)).data()"
  }
}
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:2-[[@LINE-1]]:2}:"\n{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} void useUnnamedType(UNNAMED_STRUCT * p) {return useUnnamedType(std::span<UNNAMED_STRUCT>(p, <# size #>));}\n"

void useUnnamedType2(decltype(VarOfUnnamedType) * p) { // expected-warning{{'p' is an unsafe pointer used for buffer access}} \
						          expected-note{{change type of 'p' to 'std::span' to preserve bounds information}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:22-[[@LINE-2]]:52}:"std::span<decltype(VarOfUnnamedType)> p"
  if (++p) {  // expected-note{{used in pointer arithmetic here}}
    // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:7-[[@LINE-1]]:10}:"(p = p.subspan(1)).data()"
  }
}
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:2-[[@LINE-1]]:2}:"\n{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} void useUnnamedType2(decltype(VarOfUnnamedType) * p) {return useUnnamedType2(std::span<decltype(VarOfUnnamedType)>(p, <# size #>));}\n"

#endif
