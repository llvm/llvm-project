// RUN: %clang_cc1 -fcxx-exceptions -fsyntax-only -Wunused -Wunused-template -Wunused-exception-parameter -Wunused-member-function -fdiagnostics-print-source-range-info %s 2>&1 | FileCheck %s --strict-whitespace
#include "Inputs/diag-unused-source-ranges.h"

#define CAT(a, b) a ## b

// CHECK:      :{55:15-55:20}: warning: unused exception parameter 'param'
// CHECK-NEXT:   catch (int &param) {}
// CHECK-NEXT:               ^~~~~{{$}}

// CHECK:      :{53:7-53:12}: warning: unused variable 'local'
// CHECK-NEXT:   int local = 0;
// CHECK-NEXT:       ^~~~~{{$}}

// CHECK:      In file included from
// CHECK-NEXT: :{1:13-1:18}: warning: 'static' function 'thing' declared in header file should be declared 'static inline'
// CHECK-NEXT:   static void thing(void) {}
// CHECK-NEXT:               ^~~~~{{$}}

namespace {
class A {
  // CHECK:      :{[[@LINE+3]]:10-[[@LINE+3]]:14}: warning: member function 'func' is not needed
  // CHECK-NEXT:   void func() {}
  // CHECK-NEXT:        ^~~~{{$}}
    void func() {}
  // CHECK:      :{[[@LINE+3]]:32-[[@LINE+3]]:37}: warning: unused function template
  // CHECK-NEXT:   void templ(T) {}
  // CHECK-NEXT:        ^~~~~{{$}}
    template <typename T> void templ(T) {}
  // CHECK:      :{[[@LINE+3]]:22-[[@LINE+3]]:32}: warning: member function 'templ<int>' is not needed
  // CHECK-NEXT:   void templ<int>(int) {}
  // CHECK-NEXT:        ^~~~~~~~~~{{$}}
    template <> void templ<int>(int) {}
  // CHECK:      :{[[@LINE+3]]:22-[[@LINE+3]]:27}: warning: member function 'templ<float>' is not needed
  // CHECK-NEXT:   void templ(float) {}
  // CHECK-NEXT:        ^~~~~{{$}}
    template <> void templ(float) {}

  // CHECK:      :{[[@LINE+4]]:10-[[@LINE+4]]:13}: warning: unused function template
  // CHECK-NEXT:   void foo() {
  // CHECK-NEXT:        ^~~{{$}}
    template <typename T>
    void foo() {
      func();
      templ(0);
      templ(0.0f);
      templ(0.0);
    }
};
// CHECK:      :{[[@LINE+3]]:12-[[@LINE+3]]:23}: warning: unused function 'unused_func'
// CHECK-NEXT:   static int unused_func(int aaa, char bbb) {
// CHECK-NEXT:              ^~~~~~~~~~~{{$}}
static int unused_func(int aaa, char bbb) {
  int local = 0;
  try{}
  catch (int &param) {}
  return 0;
}

// CHECK:      :{[[@LINE+4]]:6-[[@LINE+4]]:16}: warning: unused function template
// CHECK-NEXT:   auto arrow_decl(T a, T b) ->
// CHECK-NEXT:        ^~~~~~~~~~{{$}}
template <typename T>
auto arrow_decl(T a, T b) -> decltype(a + b) { thing(); return a + b; }

// CHECK:      :{[[@LINE+4]]:6-[[@LINE+4]]:21}: warning: unused function 'arrow_decl<int>'
// CHECK-NEXT:   auto arrow_decl<int>(int a, int b) ->
// CHECK-NEXT:        ^~~~~~~~~~~~~~~{{$}}
template <>
auto arrow_decl<int>(int a, int b) -> int { return a + b; }


// CHECK:      :{[[@LINE+4]]:10-[[@LINE+4]]:20}: warning: unused function template
// CHECK-NEXT:   static T func_templ(int bbb, T ccc) {
// CHECK-NEXT:            ^~~~~~~~~~{{$}}
template <typename T>
static T func_templ(int bbb, T ccc) {
  return ccc;
}

// CHECK:      :{[[@LINE+3]]:17-[[@LINE+3]]:32}: warning: function 'func_templ<int>' is not needed
// CHECK-NEXT:   int func_templ<int>(int bbb, int ccc) {
// CHECK-NEXT:       ^~~~~~~~~~~~~~~{{$}}
template <> int func_templ<int>(int bbb, int ccc) {
  return bbb;
}

// CHECK:      :{[[@LINE+3]]:35-[[@LINE+3]]:47}: warning: unused function template
// CHECK-NEXT:   static void never_called() {
// CHECK-NEXT:               ^~~~~~~~~~~~{{$}}
template <typename T> static void never_called() {
  func_templ<int>(0, 0);
}

// CHECK:      :{[[@LINE+3]]:22-[[@LINE+3]]:31}: warning: unused variable template
// CHECK-NEXT:   int var_templ =
// CHECK-NEXT:       ^~~~~~~~~{{$}}
template <int n> int var_templ = n * var_templ<n-1>;
// CHECK:      :{[[@LINE+3]]:17-[[@LINE+3]]:29}: warning: variable 'var_templ<0>' is not needed
// CHECK-NEXT:   int var_templ<0> =
// CHECK-NEXT:       ^~~~~~~~~~~~{{$}}
template <> int var_templ<0> = 1;
struct {
// CHECK:      :{[[@LINE+3]]:8-[[@LINE+3]]:11}: warning: unused member function 'fun'
// CHECK-NEXT:   void fun() {}
// CHECK-NEXT:        ^~~{{$}}
  void fun() {}
// CHECK:      :{[[@LINE+3]]:3-[[@LINE+3]]:8}: warning: unused variable 'var_x'
// CHECK-NEXT:   } var_x;
// CHECK-NEXT:     ^~~~~{{$}}
} var_x;

// CHECK:      :{[[@LINE+5]]:12-[[@LINE+6]]:12}: warning: unused variable 'new_line'
// CHECK-NEXT:   static int CAT(new_,
// CHECK-NEXT:              ^~~~~~~~~{{$}}
// CHECK-NEXT:         line) =
// CHECK-NEXT:         ~~~~~{{$}}
static int CAT(new_,
      line) = sizeof(var_templ<0>);
}

// CHECK:      :{[[@LINE+3]]:15-[[@LINE+3]]:27}: warning: unused variable 'const_unused'
// CHECK-NEXT:   constexpr int const_unused = 1
// CHECK-NEXT:                 ^~~~~~~~~~~~{{$}}
constexpr int const_unused = 1;
