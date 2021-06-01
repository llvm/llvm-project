// RUN: %clang_cc1 -triple spir64-unknown-unknown-sycldevice -fsycl-is-device -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s
// CHECK: @[[LAMBDA_KERNEL3:[^\w]+]] = private unnamed_addr constant [[LAMBDA_K3_SIZE:\[[0-9]+ x i8\]]] c"_ZTSZ4mainEUlPZ4mainEUlvE10000_E10000_\00"
// CHECK: @[[INT1:[^\w]+]] = private unnamed_addr constant [[INT_SIZE:\[[0-9]+ x i8\]]] c"_ZTSi\00"
// CHECK: @[[STRING:[^\w]+]] = private unnamed_addr constant [[STRING_SIZE:\[[0-9]+ x i8\]]] c"_ZTSAppL_ZZ4mainE1jE_i\00",
// CHECK: @[[INT2:[^\w]+]] = private unnamed_addr constant [[INT_SIZE]] c"_ZTSi\00"
// CHECK: @[[LAMBDA_X:[^\w]+]] = private unnamed_addr constant [[LAMBDA_X_SIZE:\[[0-9]+ x i8\]]] c"_ZTSZZ4mainENKUlvE10001_clEvEUlvE_\00"
// CHECK: @[[MACRO_X:[^\w]+]] = private unnamed_addr constant [[MACRO_SIZE:\[[0-9]+ x i8\]]] c"_ZTSZZ4mainENKUlvE10001_clEvEUlvE0_\00"
// CHECK: @[[MACRO_Y:[^\w]+]] =  private unnamed_addr constant [[MACRO_SIZE]] c"_ZTSZZ4mainENKUlvE10001_clEvEUlvE1_\00"
// CHECK: @{{.*}} = private unnamed_addr constant [36 x i8] c"_ZTSZZ4mainENKUlvE10001_clEvEUlvE2_\00", align 1
// CHECK: @{{.*}} = private unnamed_addr constant [36 x i8] c"_ZTSZZ4mainENKUlvE10001_clEvEUlvE3_\00", align 1
// CHECK: @[[MACRO_MACRO_X:[^\w]+]] = private unnamed_addr constant [[MACRO_MACRO_SIZE:\[[0-9]+ x i8\]]] c"_ZTSZZ4mainENKUlvE10001_clEvEUlvE4_\00"
// CHECK: @[[MACRO_MACRO_Y:[^\w]+]] = private unnamed_addr constant [[MACRO_MACRO_SIZE]] c"_ZTSZZ4mainENKUlvE10001_clEvEUlvE5_\00"
// CHECK: @[[INT3:[^\w]+]] = private unnamed_addr constant [[INT_SIZE]] c"_ZTSi\00"
// CHECK: @[[LAMBDA:[^\w]+]] = private unnamed_addr constant [[LAMBDA_SIZE:\[[0-9]+ x i8\]]] c"_ZTSZZ4mainENKUlvE10001_clEvEUlvE_\00"
// CHECK: @[[LAMBDA_IN_DEP_INT:[^\w]+]] = private unnamed_addr constant [[DEP_INT_SIZE:\[[0-9]+ x i8\]]] c"_ZTSZ28lambda_in_dependent_functionIiEvvEUlvE_\00",
// CHECK: @[[LAMBDA_IN_DEP_X:[^\w]+]] = private unnamed_addr constant [[DEP_LAMBDA_SIZE:\[[0-9]+ x i8\]]] c"_ZTSZ28lambda_in_dependent_functionIZZ4mainENKUlvE10001_clEvEUlvE_EvvEUlvE_\00",
// CHECK: @[[LAMBDA_NO_DEP:[^\w]+]] = private unnamed_addr constant [[NO_DEP_LAMBDA_SIZE:\[[0-9]+ x i8\]]] c"_ZTSZ13lambda_no_depIidEvT_T0_EUlidE_\00",
// CHECK: @[[LAMBDA_TWO_DEP:[^\w]+]] = private unnamed_addr constant [[DEP_LAMBDA1_SIZE:\[[0-9]+ x i8\]]] c"_ZTSZ14lambda_two_depIZZ4mainENKUlvE10001_clEvEUliE_ZZ4mainENKS0_clEvEUldE_EvvEUlvE_\00",
// CHECK: @[[LAMBDA_TWO_DEP2:[^\w]+]] = private unnamed_addr constant [[DEP_LAMBDA2_SIZE:\[[0-9]+ x i8\]]] c"_ZTSZ14lambda_two_depIZZ4mainENKUlvE10001_clEvEUldE_ZZ4mainENKS0_clEvEUliE_EvvEUlvE_\00",

extern "C" void puts(const char *) {}

template <typename T>
void template_param() {
  puts(__builtin_sycl_unique_stable_name(T));
}

template <typename T>
void lambda_in_dependent_function() {
  auto y = [] {};
  puts(__builtin_sycl_unique_stable_name(decltype(y)));
}

template <typename Tw, typename Tz>
void lambda_two_dep() {
  auto z = [] {};
  puts(__builtin_sycl_unique_stable_name(decltype(z)));
}

template <typename Tw, typename Tz>
void lambda_no_dep(Tw a, Tz b) {
  auto p = [](Tw a, Tz b) { return ((Tz)a + b); };
  puts(__builtin_sycl_unique_stable_name(decltype(p)));
}

#define DEF_IN_MACRO()                                        \
  auto MACRO_X = []() {};                                     \
  auto MACRO_Y = []() {};                                     \
  puts(__builtin_sycl_unique_stable_name(decltype(MACRO_X))); \
  puts(__builtin_sycl_unique_stable_name(decltype(MACRO_Y)));

#define MACRO_CALLS_MACRO() \
  { DEF_IN_MACRO(); }       \
  { DEF_IN_MACRO(); }

template <typename Ty>
auto func() -> decltype(__builtin_sycl_unique_stable_name(decltype(Ty::str)));

struct Derp {
  static constexpr const char str[] = "derp derp derp";
};

template <typename KernelName, typename KernelType>
[[clang::sycl_kernel]] void kernel_single_task(KernelType kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class kernel2>(func<Derp>);
  // CHECK: call spir_func void @_Z18kernel_single_taskIZ4mainE7kernel2PFPKcvEEvT0_(i8 addrspace(4)* ()* @_Z4funcI4DerpEDTu33__builtin_sycl_unique_stable_nameDtsrT_3strEEEv)

  auto l1 = []() { return 1; };
  auto l2 = [](decltype(l1) *l = nullptr) { return 2; };
  kernel_single_task<class kernel3>(l2);
  puts(__builtin_sycl_unique_stable_name(decltype(l2)));
  // CHECK: call spir_func void @_Z18kernel_single_taskIZ4mainE7kernel3Z4mainEUlPZ4mainEUlvE_E_EvT0_
  // CHECK: call spir_func void @puts(i8 addrspace(4)* addrspacecast (i8* getelementptr inbounds ([[LAMBDA_K3_SIZE]], [[LAMBDA_K3_SIZE]]* @[[LAMBDA_KERNEL3]], i32 0, i32 0) to i8 addrspace(4)*))

  constexpr const char str[] = "lalala";
  static_assert(__builtin_strcmp(__builtin_sycl_unique_stable_name(decltype(str)), "_ZTSA7_Kc\0") == 0, "unexpected mangling");

  int i = 0;
  puts(__builtin_sycl_unique_stable_name(decltype(i++)));
  // CHECK: call spir_func void @puts(i8 addrspace(4)* addrspacecast (i8* getelementptr inbounds ([[INT_SIZE]], [[INT_SIZE]]* @[[INT1]], i32 0, i32 0) to i8 addrspace(4)*))

  // FIXME: Ensure that j is incremented because VLAs are terrible.
  int j = 55;
  puts(__builtin_sycl_unique_stable_name(int[++j]));
  // CHECK: call spir_func void @puts(i8 addrspace(4)* addrspacecast (i8* getelementptr inbounds ([[STRING_SIZE]], [[STRING_SIZE]]* @[[STRING]], i32 0, i32 0) to i8 addrspace(4)*))

  // CHECK: define internal spir_func void @_Z18kernel_single_taskIZ4mainE7kernel2PFPKcvEEvT0_
  // CHECK: declare spir_func i8 addrspace(4)* @_Z4funcI4DerpEDTu33__builtin_sycl_unique_stable_nameDtsrT_3strEEEv
  // CHECK: define internal spir_func void @_Z18kernel_single_taskIZ4mainE7kernel3Z4mainEUlPZ4mainEUlvE_E_EvT0_
  // CHECK: define internal spir_func void @_Z18kernel_single_taskIZ4mainE6kernelZ4mainEUlvE0_EvT0_

  kernel_single_task<class kernel>(
      []() {
        puts(__builtin_sycl_unique_stable_name(int));
        // CHECK: call spir_func void @puts(i8 addrspace(4)* addrspacecast (i8* getelementptr inbounds ([[INT_SIZE]], [[INT_SIZE]]* @[[INT2]], i32 0, i32 0) to i8 addrspace(4)*))

        auto x = []() {};
        puts(__builtin_sycl_unique_stable_name(decltype(x)));
        // CHECK: call spir_func void @puts(i8 addrspace(4)* addrspacecast (i8* getelementptr inbounds ([[LAMBDA_X_SIZE]], [[LAMBDA_X_SIZE]]* @[[LAMBDA_X]], i32 0, i32 0) to i8 addrspace(4)*))

        DEF_IN_MACRO();
        // CHECK: call spir_func void @puts(i8 addrspace(4)* addrspacecast (i8* getelementptr inbounds ([[MACRO_SIZE]], [[MACRO_SIZE]]* @[[MACRO_X]], i32 0, i32 0) to i8 addrspace(4)*))
        // CHECK: call spir_func void @puts(i8 addrspace(4)* addrspacecast (i8* getelementptr inbounds ([[MACRO_SIZE]], [[MACRO_SIZE]]* @[[MACRO_Y]], i32 0, i32 0) to i8 addrspace(4)*))

        MACRO_CALLS_MACRO();
        // CHECK: call spir_func void @puts(i8 addrspace(4)* addrspacecast (i8* getelementptr inbounds ([[MACRO_MACRO_SIZE]], [[MACRO_MACRO_SIZE]]* @[[MACRO_MACRO_X]], i32 0, i32 0) to i8 addrspace(4)*))
        // CHECK: call spir_func void @puts(i8 addrspace(4)* addrspacecast (i8* getelementptr inbounds ([[MACRO_MACRO_SIZE]], [[MACRO_MACRO_SIZE]]* @[[MACRO_MACRO_Y]], i32 0, i32 0) to i8 addrspace(4)*))

        template_param<int>();
        // CHECK: call spir_func void @_Z14template_paramIiEvv

        template_param<decltype(x)>();
        // CHECK: call spir_func void @_Z14template_paramIZZ4mainENKUlvE0_clEvEUlvE_Evv

        lambda_in_dependent_function<int>();
        // CHECK: call spir_func void @_Z28lambda_in_dependent_functionIiEvv

        lambda_in_dependent_function<decltype(x)>();
        // CHECK: call spir_func void @_Z28lambda_in_dependent_functionIZZ4mainENKUlvE0_clEvEUlvE_Evv

        lambda_no_dep<int, double>(3, 5.5);
        // CHECK: call spir_func void @_Z13lambda_no_depIidEvT_T0_(i32 3, double 5.500000e+00)

        int a = 5;
        double b = 10.7;
        auto y = [](int a) { return a; };
        auto z = [](double b) { return b; };
        lambda_two_dep<decltype(y), decltype(z)>();
        // CHECK: call spir_func void @_Z14lambda_two_depIZZ4mainENKUlvE0_clEvEUliE_ZZ4mainENKS0_clEvEUldE_Evv

        lambda_two_dep<decltype(z), decltype(y)>();
        // CHECK: call spir_func void @_Z14lambda_two_depIZZ4mainENKUlvE0_clEvEUldE_ZZ4mainENKS0_clEvEUliE_Evv
      });
}

// CHECK: define linkonce_odr spir_func void @_Z14template_paramIiEvv
// CHECK: call spir_func void @puts(i8 addrspace(4)* addrspacecast (i8* getelementptr inbounds ([[INT_SIZE]], [[INT_SIZE]]* @[[INT3]], i32 0, i32 0) to i8 addrspace(4)*))

// CHECK: define internal spir_func void @_Z14template_paramIZZ4mainENKUlvE0_clEvEUlvE_Evv
// CHECK: call spir_func void @puts(i8 addrspace(4)* addrspacecast (i8* getelementptr inbounds ([[LAMBDA_SIZE]], [[LAMBDA_SIZE]]* @[[LAMBDA]], i32 0, i32 0) to i8 addrspace(4)*))

// CHECK: define linkonce_odr spir_func void @_Z28lambda_in_dependent_functionIiEvv
// CHECK: call spir_func void @puts(i8 addrspace(4)* addrspacecast (i8* getelementptr inbounds ([[DEP_INT_SIZE]], [[DEP_INT_SIZE]]* @[[LAMBDA_IN_DEP_INT]], i32 0, i32 0) to i8 addrspace(4)*))

// CHECK: define internal spir_func void @_Z28lambda_in_dependent_functionIZZ4mainENKUlvE0_clEvEUlvE_Evv
// CHECK: call spir_func void @puts(i8 addrspace(4)* addrspacecast (i8* getelementptr inbounds ([[DEP_LAMBDA_SIZE]], [[DEP_LAMBDA_SIZE]]* @[[LAMBDA_IN_DEP_X]], i32 0, i32 0) to i8 addrspace(4)*))

// CHECK: define linkonce_odr spir_func void @_Z13lambda_no_depIidEvT_T0_(i32 %a, double %b)
// CHECK: call spir_func void @puts(i8 addrspace(4)* addrspacecast (i8* getelementptr inbounds ([[NO_DEP_LAMBDA_SIZE]], [[NO_DEP_LAMBDA_SIZE]]* @[[LAMBDA_NO_DEP]], i32 0, i32 0) to i8 addrspace(4)*))

// CHECK: define internal spir_func void @_Z14lambda_two_depIZZ4mainENKUlvE0_clEvEUliE_ZZ4mainENKS0_clEvEUldE_Evv
// CHECK: call spir_func void @puts(i8 addrspace(4)* addrspacecast (i8* getelementptr inbounds ([[DEP_LAMBDA1_SIZE]], [[DEP_LAMBDA1_SIZE]]* @[[LAMBDA_TWO_DEP]], i32 0, i32 0) to i8 addrspace(4)*))

// CHECK: define internal spir_func void @_Z14lambda_two_depIZZ4mainENKUlvE0_clEvEUldE_ZZ4mainENKS0_clEvEUliE_Evv
// CHECK: call spir_func void @puts(i8 addrspace(4)* addrspacecast (i8* getelementptr inbounds ([[DEP_LAMBDA2_SIZE]], [[DEP_LAMBDA2_SIZE]]* @[[LAMBDA_TWO_DEP2]], i32 0, i32 0) to i8 addrspace(4)*))
