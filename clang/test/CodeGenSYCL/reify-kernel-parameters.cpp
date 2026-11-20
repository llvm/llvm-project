// RUN: %clang_cc1 -fsycl-is-host -emit-llvm -triple x86_64-unknown-linux-gnu -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-HOST %s

// A unique kernel name type is required for each declared kernel entry point.
template<int> struct KN;

struct [[clang::sycl_reified_kernel_parameter]] EmptySpecial {
  int data;
};

template<typename T>
struct Wrapper {
 T data;
 int *data1;
};

template <typename KernelInfo, typename... Ts>
auto sycl_kernel_launch(Ts&&...) {
    return [](auto&&... special_subobjects) { };
}

template <typename KernelInfo, typename... Ts>
auto sycl_reify_kernel_parameters(Ts&&...) {
    return [](auto&&... special_subobjects) { };
}


template <typename KN, typename KT>
[[clang::sycl_kernel_entry_point(KN)]]
void kernel_entry_point(KT Kernel) {
  Kernel();
}

void case1() {
    Wrapper<EmptySpecial> KernelArg;
    kernel_entry_point<KN<0>>([KernelArg](){});
}

struct [[clang::sycl_reified_kernel_parameter]] SpecialBase {
  int data;
};

struct DerivedFromSpecial : SpecialBase {
  int extra;
};

void case2() {
    DerivedFromSpecial DFS;
    kernel_entry_point<KN<1>>([DFS](){});
}

// CHECK-HOST-LABEL:   define internal void @_Z18kernel_entry_pointI2KNILi0EEZ5case1vEUlvE_EvT0_(
// CHECK-HOST-SAME:      i32 [[KERNEL_COERCE0:%.*]], ptr [[KERNEL_COERCE1:%.*]])
// CHECK-HOST:         entry:
// CHECK-HOST:          call void @_Z18sycl_kernel_launchI18__sycl_kernel_infoI2KNILi0EEEJRZ5case1vEUlvE_EEDaDpOT0_(ptr noundef nonnull align 8 dereferenceable(16) %Kernel)
// CHECK-HOST-NEXT:     %2 = getelementptr inbounds nuw %class.anon, ptr %Kernel, i32 0, i32 0
// CHECK-HOST-NEXT:     %data = getelementptr inbounds nuw %struct.Wrapper, ptr %2, i32 0, i32 0
// CHECK-HOST-NEXT:     call void @_ZZ18sycl_kernel_launchI18__sycl_kernel_infoI2KNILi0EEEJRZ5case1vEUlvE_EEDaDpOT0_ENKUlDpOT_E_clIJR12EmptySpecialEEEDaSB_(
// CHECK-HOST-SAME:       ptr noundef nonnull align 1 dereferenceable(1) %ref.tmp, ptr noundef nonnull align 4 dereferenceable(4) %data)
// CHECK-HOST-NEXT:     ret void

// Test that a class inheriting from a sycl_reified_kernel_parameter type
// accesses the SpecialBase via DerivedToBase cast.

// CHECK-HOST-LABEL:   define internal void @_Z18kernel_entry_pointI2KNILi1EEZ5case2vEUlvE_EvT0_(
// CHECK-HOST-SAME:      i64 %Kernel.coerce)
// CHECK-HOST:         entry:
// CHECK-HOST:          call void @_Z18sycl_kernel_launchI18__sycl_kernel_infoI2KNILi1EEEJRZ5case2vEUlvE_EEDaDpOT0_(ptr noundef nonnull align 4 dereferenceable(8) %Kernel)
// CHECK-HOST:          %0 = getelementptr inbounds nuw %class.anon.0, ptr %Kernel, i32 0, i32 0
// CHECK-HOST-NEXT:     call void @_ZZ18sycl_kernel_launchI18__sycl_kernel_infoI2KNILi1EEEJRZ5case2vEUlvE_EEDaDpOT0_ENKUlDpOT_E_clIJR11SpecialBaseEEEDaSB_(
// CHECK-HOST-SAME:       ptr noundef nonnull align 1 dereferenceable(1) %ref.tmp, ptr noundef nonnull align 4 dereferenceable(4) %0)
// CHECK-HOST-NEXT:     ret void
