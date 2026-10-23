// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

struct Noisy {
  Noisy();
  ~Noisy();
};

struct Function {
  template <typename F> Function(F) {}
};

struct Options {
  Function function{[noisy = Noisy{}] {}};
};

Options kOptions{};

int side();

struct ReturnsCapture {
  int x;
  int value = [value = x] { return value; }();
};

ReturnsCapture kReturnsCapture{side()};

// CHECK-LABEL: define internal void @__cxx_global_var_init
// CHECK: call void @_ZN5NoisyC1Ev
// CHECK: call void @_ZN8FunctionC1IN7Options8functionMUlvE_EEET_
// CHECK: call void @_ZN7Options8functionMUlvE_D1Ev
// CHECK: call {{.*}} @_ZNK14ReturnsCapture5valueMUlvE_clEv

// CHECK-LABEL: define linkonce_odr {{.*}} @_ZNK14ReturnsCapture5valueMUlvE_clEv
// CHECK: ret i32

// CHECK-LABEL: define {{.*}} @_ZN7Options8functionMUlvE_D2Ev
// CHECK: call void @_ZN5NoisyD1Ev
