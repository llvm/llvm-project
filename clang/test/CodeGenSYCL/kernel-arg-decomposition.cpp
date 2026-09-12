// RUN: %clang_cc1 -fsycl-is-host -emit-llvm -triple x86_64-unknown-linux-gnu -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-HOST %s

// A unique kernel name type is required for each declared kernel entry point.
template<int> struct KN;

struct [[clang::sycl_special_kernel_parameter]] EmptySpecial {
  int data;
};

template<typename T>
struct Wrapper {
 T data;
 int *data1;
};

template <typename KernelName, typename... Ts>
auto sycl_kernel_launch(const char *, Ts...) {
    return [](auto&&... special_subobjects) { };
}


template <typename KN, typename KT>
[[clang::sycl_kernel_entry_point(KN)]] void kernel_entry_point(KT Kernel) {
  Kernel();
}

void case1() {
    Wrapper<EmptySpecial> KernelArg;
    kernel_entry_point<KN<0>>([KernelArg](){});
}

struct [[clang::sycl_special_kernel_parameter]] SpecialBase {
  int data;
};

struct DerivedFromSpecial : SpecialBase {
  int extra;
};

void case2() {
    DerivedFromSpecial DFS;
    kernel_entry_point<KN<1>>([DFS](){});
}

// CHECK-HOST-LABEL: define internal void @_Z18kernel_entry_pointI2KNILi0EEZ5case1vEUlvE_EvT0_(
// CHECK-HOST-SAME: i32 [[KERNEL_COERCE0:%.*]], ptr [[KERNEL_COERCE1:%.*]])
// CHECK-HOST:  [[ENTRY:.*:]]
// CHECK-HOST:    call void @_Z18sycl_kernel_launchI2KNILi0EEJZ5case1vEUlvE_EEDaPKcDpT0_(ptr noundef @.str, i32 [[TMP3:%.*]], ptr [[TMP5:%.*]])
// CHECK-HOST-NEXT:    [[TMP6:%.*]] = getelementptr inbounds nuw [[CLASS_ANON:%.*]], ptr [[KERNEL:%.*]], i32 0, i32 0
// CHECK-HOST-NEXT:    [[DATA:%.*]] = getelementptr inbounds nuw [[STRUCT_WRAPPER:%.*]], ptr [[TMP6]], i32 0, i32 0
// CHECK-HOST-NEXT:    call void @_ZZ18sycl_kernel_launchI2KNILi0EEJZ5case1vEUlvE_EEDaPKcDpT0_ENKUlDpOT_E_clIJR12EmptySpecialEEEDaS9_(ptr noundef nonnull align 1 dereferenceable(1) [[REF_TMP:%.*]], ptr noundef nonnull align 4 dereferenceable(4) [[DATA]])
// CHECK-HOST-NEXT:    ret void

// Test that a class inheriting from a sycl_special_kernel_parameter type
// accesses the SpecialBase via DerivedToBase cast.

// CHECK-HOST-LABEL: define internal void @_Z18kernel_entry_pointI2KNILi1EEZ5case2vEUlvE_EvT0_(
// CHECK-HOST-SAME: i64 [[KERNEL_COERCE:%.*]])
// CHECK-HOST:  [[ENTRY:.*:]]
// CHECK-HOST:    call void @_Z18sycl_kernel_launchI2KNILi1EEJZ5case2vEUlvE_EEDaPKcDpT0_(ptr noundef @.str.1, i64 %{{.*}})
// CHECK-HOST:    [[TMP1:%.*]] = getelementptr inbounds nuw %class.anon.0, ptr %Kernel, i32 0, i32 0
// CHECK-HOST-NEXT:    call void @_ZZ18sycl_kernel_launchI2KNILi1EEJZ5case2vEUlvE_EEDaPKcDpT0_ENKUlDpOT_E_clIJR11SpecialBaseEEEDaS9_(ptr noundef nonnull align 1 dereferenceable(1) %ref.tmp, ptr noundef nonnull align 4 dereferenceable(4) [[TMP1]])
// CHECK-HOST-NEXT:    ret void
