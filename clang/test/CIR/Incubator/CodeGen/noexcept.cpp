// RUN: %clang_cc1 -x c++ -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -std=c++11 %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

void may_throw();
void no_throw() noexcept;

bool test_noexcept_func_false() {
  return noexcept(may_throw());
}
// CHECK-LABEL: cir.func{{.*}} @_Z24test_noexcept_func_falsev
// CHECK:         %[[CONST:.*]] = cir.const #false
// CHECK:         cir.return

bool test_noexcept_func_true() {
  return noexcept(no_throw());
}
// CHECK-LABEL: cir.func{{.*}} @_Z23test_noexcept_func_truev
// CHECK:         %[[CONST:.*]] = cir.const #true
// CHECK:         cir.return

auto lambda_may_throw = []() {};
auto lambda_no_throw = []() noexcept {};

bool test_noexcept_lambda_false() {
  return noexcept(lambda_may_throw());
}
// CHECK-LABEL: cir.func{{.*}} @_Z26test_noexcept_lambda_falsev
// CHECK:         %[[CONST:.*]] = cir.const #false
// CHECK:         cir.return

bool test_noexcept_lambda_true() {
  return noexcept(lambda_no_throw());
}
// CHECK-LABEL: cir.func{{.*}} @_Z25test_noexcept_lambda_truev
// CHECK:         %[[CONST:.*]] = cir.const #true
// CHECK:         cir.return
