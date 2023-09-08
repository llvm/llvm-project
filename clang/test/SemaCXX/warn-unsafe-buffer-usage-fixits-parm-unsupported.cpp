// RUN: %clang_cc1 -std=c++20 -Wno-all -Wunsafe-buffer-usage -fcxx-exceptions -fsafe-buffer-usage-suggestions -verify %s
// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage -fcxx-exceptions -fdiagnostics-parseable-fixits -fsafe-buffer-usage-suggestions %s 2>&1 | FileCheck %s

typedef int * TYPEDEF_PTR;
#define MACRO_PTR int*

// We CANNOT fix a pointer whose type is defined in a typedef or a
// macro. Because if the typedef is changed after the fix, the fix
// becomes incorrect and may not be noticed.

// CHECK-NOT: fix-it:"{{.*}}":{[[@LINE+1]]
void typedefPointer(TYPEDEF_PTR p) {  // expected-warning{{'p' is an unsafe pointer used for buffer access}}
  if (++p) {  // expected-note{{used in pointer arithmetic here}}
  }
}

// CHECK-NOT: fix-it:"{{.*}}":{[[@LINE+1]]
void macroPointer(MACRO_PTR p) {  // expected-warning{{'p' is an unsafe pointer used for buffer access}}
  if (++p) {  // expected-note{{used in pointer arithmetic here}}
  }
}

// The analysis requires accurate source location informations from
// `TypeLoc`s of types of variable (parameter) declarations in order
// to generate fix-its for them. But those information is not always
// available (probably due to some bugs in clang but it is irrelevant
// to the safe-buffer project).  The following is an example.  When
// `_Atomic` is used, we cannot get valid source locations of the
// pointee type of `unsigned *`.  The analysis gives up in such a
// case.
// CHECK-NOT: fix-it:
void typeLocSourceLocationInvalid(_Atomic unsigned *map) { // expected-warning{{'map' is an unsafe pointer used for buffer access}}
  map[5] = 5; // expected-note{{used in buffer access here}}
}

// CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:33-[[@LINE+1]]:46}:"std::span<unsigned> map"
void typeLocSourceLocationValid(unsigned *map) { // expected-warning{{'map' is an unsafe pointer used for buffer access}} \
						    expected-note{{change type of 'map' to 'std::span' to preserve bounds information}}
  map[5] = 5; // expected-note{{used in buffer access here}}
}
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:2-[[@LINE-1]]:2}:"\n{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} void typeLocSourceLocationValid(unsigned *map) {return typeLocSourceLocationValid(std::span<unsigned>(map, <# size #>));}\n"

// We do not fix parameters participating unsafe operations for the
// following functions/methods or function-like expressions:

// CHECK-NOT: fix-it:
class A {
  // constructor & descructor
  A(int * p) {  // expected-warning{{'p' is an unsafe pointer used for buffer access}}
    int tmp;
    tmp = p[5]; // expected-note{{used in buffer access here}}
  }

  // class member methods
  void foo(int *p) { // expected-warning{{'p' is an unsafe pointer used for buffer access}}
    int tmp;
    tmp = p[5];      // expected-note{{used in buffer access here}}
  }

  // overload operator
  int operator+(int * p) { // expected-warning{{'p' is an unsafe pointer used for buffer access}}
    int tmp;
    tmp = p[5];            // expected-note{{used in buffer access here}}
    return tmp;
  }
};

// lambdas
void foo() {
  auto Lamb = [&](int *p) // expected-warning{{'p' is an unsafe pointer used for buffer access}}
    -> int {
    int tmp;
    tmp = p[5];           // expected-note{{used in buffer access here}}
    return tmp;
  };
}

// template
template<typename T>
void template_foo(T * p) { // expected-warning{{'p' is an unsafe pointer used for buffer access}}
  T tmp;
  tmp = p[5];              // expected-note{{used in buffer access here}}
}

void instantiate_template_foo() {
  int * p;
  template_foo(p);        // FIXME expected note {{in instantiation of function template specialization 'template_foo<int>' requested here}}
}

// variadic function
void vararg_foo(int * p...) { // expected-warning{{'p' is an unsafe pointer used for buffer access}}
  int tmp;
  tmp = p[5];                 // expected-note{{used in buffer access here}}
}

// constexpr functions
constexpr int constexpr_foo(int * p) { // expected-warning{{'p' is an unsafe pointer used for buffer access}}
  return p[5];                         // expected-note{{used in buffer access here}}
}

// function body is a try-block
void fn_with_try_block(int* p)    // expected-warning{{'p' is an unsafe pointer used for buffer access}}
  try {
    int tmp;

    if (p == nullptr)
      throw 42;
    tmp = p[5];                   // expected-note{{used in buffer access here}}
  }
  catch (int) {
    *p = 0;
  }

// The following two unsupported cases are not specific to
// parm-fixits. Adding them here in case they get forgotten.
void isArrayDecayToPointerUPC(int a[][10], int (*b)[10]) {
// expected-warning@-1{{'a' is an unsafe pointer used for buffer access}}
// expected-warning@-2{{'b' is an unsafe pointer used for buffer access}}
  int tmp;

  tmp = a[5][5] + b[5][5];  // expected-warning2{{unsafe buffer access}}  expected-note2{{used in buffer access here}}
}

// parameter having default values:
void parmWithDefaultValue(int * x = 0) {
  // expected-warning@-1{{'x' is an unsafe pointer used for buffer access}}
  int tmp;
  tmp = x[5]; // expected-note{{used in buffer access here}}
}

void parmWithDefaultValueDecl(int * x = 0);

void parmWithDefaultValueDecl(int * x) {
  // expected-warning@-1{{'x' is an unsafe pointer used for buffer access}}
  int tmp;
  tmp = x[5]; // expected-note{{used in buffer access here}}
}

#define MACRO_NAME MyName

// The fix-it ends with a macro. It will be discarded due to overlap with macros.
// CHECK-NOT: fix-it:{{.*}}:{[[@LINE+1]]
void macroIdentifier(int * MACRO_NAME) { // expected-warning{{'MyName' is an unsafe pointer used for buffer access}}
  if (++MyName){} // expected-note{{used in pointer arithmetic here}}
}

// CHECK-NOT: fix-it:{{.*}}:{[[@LINE+1]]
void parmHasNoName(int *p, int *) { // cannot fix the function because there is one parameter has no name. \
				       expected-warning{{'p' is an unsafe pointer used for buffer access}}
  p[5] = 5; // expected-note{{used in buffer access here}}
}
