// RUN: %clang_cc1 -triple spir64-unknown-linux-sycldevice -fsycl-is-device -disable-llvm-passes -x c++ -emit-llvm %s -o - | FileCheck %s
// CHECK: @[[INT:[^\w]+]] = private unnamed_addr constant [[INT_SIZE:\[[0-9]+ x i8\]]] c"_ZTSi\00"
// CHECK: @[[LAMBDA_X:[^\w]+]] = private unnamed_addr constant [[LAMBDA_X_SIZE:\[[0-9]+ x i8\]]] c"_ZTSZZ4mainENKUlvE37->5clEvEUlvE41->16\00"
// CHECK: @[[MACRO_X:[^\w]+]] = private unnamed_addr constant [[MACRO_SIZE:\[[0-9]+ x i8\]]] c"_ZTSZZ4mainENKUlvE37->5clEvEUlvE47->7~26->18\00"
// CHECK: @[[MACRO_Y:[^\w]+]] =  private unnamed_addr constant [[MACRO_SIZE]] c"_ZTSZZ4mainENKUlvE37->5clEvEUlvE47->7~26->41\00"
// CHECK: @[[LAMBDA_IN_DEP_INT:[^\w]+]] = private unnamed_addr constant [[DEP_INT_SIZE:\[[0-9]+ x i8\]]] c"_ZTSZ28lambda_in_dependent_functionIiEvvEUlvE21->12\00",
// CHECK: @[[LAMBDA_IN_DEP_X:[^\w]+]] = private unnamed_addr constant [[DEP_LAMBDA_SIZE:\[[0-9]+ x i8\]]] c"_ZTSZ28lambda_in_dependent_functionIZZ4mainENKUlvE37->5clEvEUlvE41->16EvvEUlvE21->12\00",

extern "C" void printf(const char*);

template <typename T>
void template_param() {
  printf(__unique_stable_name(T));
}

template <typename T>
T getT() { return T{}; }

template <typename T>
void lambda_in_dependent_function() {
  auto y = [] {};
  printf(__unique_stable_name(y));
}

#define DEF_IN_MACRO()                                  \
  auto MACRO_X = []() {};auto MACRO_Y = []() {};        \
  printf(__unique_stable_name(MACRO_X));                \
  printf(__unique_stable_name(MACRO_Y));

template <typename KernelName, typename KernelType>
__attribute__((sycl_kernel)) void kernel_single_task(KernelType kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class kernel>(
    []() {
      printf(__unique_stable_name(int));
      // CHECK: call spir_func void @printf(i8* getelementptr inbounds ([[INT_SIZE]], [[INT_SIZE]]* @[[INT]]

      auto x = [](){};
      printf(__unique_stable_name(x));
      printf(__unique_stable_name(decltype(x)));
      // CHECK: call spir_func void @printf(i8* getelementptr inbounds ([[LAMBDA_X_SIZE]], [[LAMBDA_X_SIZE]]* @[[LAMBDA_X]]
      // CHECK: call spir_func void @printf(i8* getelementptr inbounds ([[LAMBDA_X_SIZE]], [[LAMBDA_X_SIZE]]* @[[LAMBDA_X]]

      DEF_IN_MACRO();
      // CHECK: call spir_func void @printf(i8* getelementptr inbounds ([[MACRO_SIZE]], [[MACRO_SIZE]]* @[[MACRO_X]]
      // CHECK: call spir_func void @printf(i8* getelementptr inbounds ([[MACRO_SIZE]], [[MACRO_SIZE]]* @[[MACRO_Y]]

      template_param<int>();
      // CHECK: define linkonce_odr spir_func void @_Z14template_paramIiEvv
      // CHECK: call spir_func void @printf(i8* getelementptr inbounds ([[INT_SIZE]], [[INT_SIZE]]* @[[INT]]

      template_param<decltype(x)>();
      // CHECK: define internal spir_func void @"_Z14template_paramIZZ4mainENK3
      // CHECK: call spir_func void @printf(i8* getelementptr inbounds ([[LAMBDA_X_SIZE]], [[LAMBDA_X_SIZE]]* @[[LAMBDA_X]]

      lambda_in_dependent_function<int>();
      // CHECK: define linkonce_odr spir_func void @_Z28lambda_in_dependent_functionIiEvv
      // CHECK: call spir_func void @printf(i8* getelementptr inbounds ([[DEP_INT_SIZE]], [[DEP_INT_SIZE]]* @[[LAMBDA_IN_DEP_INT]]

      lambda_in_dependent_function<decltype(x)>();
      // CHECK: define internal spir_func void @"_Z28lambda_in_dependent_functionIZZ4mainENK3$_0clEvEUlvE_Evv
      // CHECK: call spir_func void @printf(i8* getelementptr inbounds ([[DEP_LAMBDA_SIZE]], [[DEP_LAMBDA_SIZE]]* @[[LAMBDA_IN_DEP_X]]

    });
}

