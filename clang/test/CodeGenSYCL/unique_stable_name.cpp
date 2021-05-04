// RUN: %clang_cc1 -triple spir64-unknown-unknown-sycldevice -fsycl-is-device -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s
// CHECK: @[[INT:[^\w]+]] = private unnamed_addr constant [[INT_SIZE:\[[0-9]+ x i8\]]] c"_ZTSi\00"
// CHECK: @[[LAMBDA_X:[^\w]+]] = private unnamed_addr constant [[LAMBDA_X_SIZE:\[[0-9]+ x i8\]]] c"_ZTSZZ4mainENKUlvE10000_clEvEUlvE_\00"
// CHECK: @[[LAMBDA_Y:[^\w]+]] = private unnamed_addr constant [[LAMBDA_Y_SIZE:\[[0-9]+ x i8\]]] c"_ZTSZZ4mainENKUlvE10000_clEvEUlvE_\00"
// CHECK: @[[MACRO_X:[^\w]+]] = private unnamed_addr constant [[MACRO_SIZE:\[[0-9]+ x i8\]]] c"_ZTSZZ4mainENKUlvE10000_clEvEUlvE0_\00"
// CHECK: @[[MACRO_Y:[^\w]+]] =  private unnamed_addr constant [[MACRO_SIZE]] c"_ZTSZZ4mainENKUlvE10000_clEvEUlvE1_\00"
// CHECK: @[[MACRO_MACRO_X:[^\w]+]] = private unnamed_addr constant [[MACRO_MACRO_SIZE:\[[0-9]+ x i8\]]] c"_ZTSZZ4mainENKUlvE10000_clEvEUlvE4_\00"
// CHECK: @[[MACRO_MACRO_Y:[^\w]+]] = private unnamed_addr constant [[MACRO_MACRO_SIZE]] c"_ZTSZZ4mainENKUlvE10000_clEvEUlvE5_\00"
// CHECK: @[[LAMBDA_IN_DEP_INT:[^\w]+]] = private unnamed_addr constant [[DEP_INT_SIZE:\[[0-9]+ x i8\]]] c"_ZTSZ28lambda_in_dependent_functionIiEvvEUlvE_\00",
// CHECK: @[[LAMBDA_IN_DEP_X:[^\w]+]] = private unnamed_addr constant [[DEP_LAMBDA_SIZE:\[[0-9]+ x i8\]]] c"_ZTSZ28lambda_in_dependent_functionIZZ4mainENKUlvE10000_clEvEUlvE_EvvEUlvE_\00",
// CHECK: @[[LAMBDA_TWO_DEP:[^\w]+]] = private unnamed_addr constant [[DEP_LAMBDA1_SIZE:\[[0-9]+ x i8\]]] c"_ZTSZ30lambda_two_dependent_functionsIidEvT_T0_EUlidE_\00"
extern "C" void printf(const char *) {}

template <typename T>
void template_param() {
  printf(__builtin_unique_stable_name(T));
}

template <typename T>
T getT() { return T{}; }

template <typename T>
void lambda_in_dependent_function() {
  auto y = [] {};
  printf(__builtin_unique_stable_name(y));
}

template <typename Tw, typename Tz>
void lambda_two_dependent_functions(Tw a, Tz b) {
  //auto w = [](Tw a) {return a};
  //auto z = [](Tz b) {return b};
  //auto y = [] {};
  auto p = [](Tw a, Tz b) { return ((Tz)a+b); };
  printf(__builtin_unique_stable_name(p));
}

#define DEF_IN_MACRO()                                  \
  auto MACRO_X = []() {};auto MACRO_Y = []() {};        \
  printf(__builtin_unique_stable_name(MACRO_X));        \
  printf(__builtin_unique_stable_name(MACRO_Y));

#define MACRO_CALLS_MACRO()                             \
  {DEF_IN_MACRO();}{DEF_IN_MACRO();}

template <typename KernelName, typename KernelType>
[[clang::sycl_kernel]] void kernel_single_task(KernelType kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class kernel>(
    []() {
      printf(__builtin_unique_stable_name(int));
      // CHECK: call spir_func void @printf(i8* getelementptr inbounds ([[INT_SIZE]], [[INT_SIZE]]* @[[INT]]

      auto x = [](){};
      printf(__builtin_unique_stable_name(x));
      printf(__builtin_unique_stable_name(decltype(x)));
      // CHECK: call spir_func void @printf(i8* getelementptr inbounds ([[LAMBDA_X_SIZE]], [[LAMBDA_X_SIZE]]* @[[LAMBDA_X]]
      // CHECK: call spir_func void @printf(i8* getelementptr inbounds ([[LAMBDA_Y_SIZE]], [[LAMBDA_Y_SIZE]]* @[[LAMBDA_Y]] 

      DEF_IN_MACRO();
      // CHECK: call spir_func void @printf(i8* getelementptr inbounds ([[MACRO_SIZE]], [[MACRO_SIZE]]* @[[MACRO_X]]
      // CHECK: call spir_func void @printf(i8* getelementptr inbounds ([[MACRO_SIZE]], [[MACRO_SIZE]]* @[[MACRO_Y]]

      MACRO_CALLS_MACRO();
      // CHECK: call spir_func void @printf(i8* getelementptr inbounds ([[MACRO_MACRO_SIZE]], [[MACRO_MACRO_SIZE]]* @[[MACRO_MACRO_X]]
      // CHECK: call spir_func void @printf(i8* getelementptr inbounds ([[MACRO_MACRO_SIZE]], [[MACRO_MACRO_SIZE]]* @[[MACRO_MACRO_Y]]

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
      
      lambda_two_dependent_functions<int,double>(3, 5.5);
      // CHECK: define linkonce_odr spir_func void @_Z30lambda_two_dependent_functionsIidEvT_T0_(i32 %a, double %b)
      // CHECK: call spir_func void @printf(i8* getelementptr inbounds ([55 x i8], [55 x i8]* @usn_str.13, i32 0, i32 0))

      int a = 5;
      double b = 10.7; 
      auto y = [](int a) {return a;};
      auto z = [](double b) {return b;};
      //lambda_two_dependent_functions<decltype(y), decltype(z)>(); 
    });
}
