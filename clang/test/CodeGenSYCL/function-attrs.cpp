// RUN: %clang_cc1 -fsycl-is-device -disable-llvm-passes \
// RUN:  -triple spir64 -fexceptions -emit-llvm -fno-ident %s -o - | FileCheck %s
// RUN: %if cir-enabled %{ %clang_cc1 -fclangir -fsycl-is-device -disable-llvm-passes -triple spir64 -fexceptions -emit-llvm -fno-ident %s -o - | FileCheck %s %}

int foo();

// CHECK-LABEL: define dso_local spir_func void @_Z3barv(
// CHECK-SAME: ) #[[ATTR0:[0-9]+]] {
// CHECK: call spir_func noundef i32 @_Z3foov() #[[ATTR1:[0-9]+]]
void bar() {
  int a = foo();
}

// CHECK-LABEL: define dso_local spir_func noundef i32 @_Z3foov(
// CHECK-SAME: ) #[[ATTR0]] {
int foo() {
  return 1;
}

template <typename Name, typename... Ts>
void sycl_kernel_launch(Ts...) {}

template <typename Name, typename Func>
[[clang::sycl_kernel_entry_point(Name)]] void kernel_single_task(const Func &kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class fake_kernel>([] { bar(); });
  return 0;
}
//.
// CHECK: attributes #[[ATTR0]] = { convergent mustprogress noinline norecurse nounwind optnone "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
// CHECK: attributes #[[ATTR1]] = { convergent nounwind }
//.
