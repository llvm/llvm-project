// RUN: %clang_cc1 -std=c++1z %s -emit-llvm -fblocks -triple x86_64-apple-darwin10 -o - | FileCheck %s --implicit-check-not=should_not_be_used
// RUN: %clang_cc1 -std=c++1z %s -emit-llvm -fblocks -triple x86_64-apple-darwin10 -o - -fexperimental-new-constant-interpreter | FileCheck %s --implicit-check-not=should_not_be_used

void should_be_used_1();
void should_be_used_2();
void should_be_used_3();
void should_not_be_used();

struct A {
  constexpr explicit operator bool() const {
    return true;
  }
};

void f() {
  if constexpr (false)
    should_not_be_used();
  else
    should_be_used_1();

  if constexpr (true || ({ label: false; }))
    should_be_used_2();
  else {
    goto foo;
foo: should_not_be_used();
  }
  if constexpr (A())
    should_be_used_3();
  else
    should_not_be_used();
}

// CHECK: should_be_used_1
// CHECK: should_be_used_2
// CHECK: should_be_used_3

namespace GH223003 {
  static int internal_var = 42;
  template <bool x>
  int read() {
    if constexpr (x) {
      extern int internal_var;
      return internal_var;
    } else {
      extern int internal_var;
      return internal_var;
    }
  }
  int use() { return read<false>(); }
}

// CHECK-LABEL: define {{.*}} @_ZN8GH2230034readILb0EEEiv(
// CHECK: load i32, ptr @_ZN8GH223003L12internal_varE

namespace BlockThisCapture {
  void foo();
  struct S {
    template <bool b>
    void m() {
      ^{ if constexpr(b) (void)this; else foo();  }();
    }
  };

  void test() {
    S().m<false>();
  }
}

// CHECK-LABEL: define internal void @___ZN16BlockThisCapture1S1mILb0EEEvv_block_invoke(
// CHECK: call void @_ZN16BlockThisCapture3fooEv(
