// RUN: %clang_cc1 -fsycl-is-host -emit-llvm -triple x86_64-unknown-linux-gnu -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-HOST %s
// RUN: %clang_cc1 -fsycl-is-device -emit-llvm -aux-triple x86_64-unknown-linux-gnu -triple spirv64-unknown-unknown -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-DEVICE %s

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

template<typename T>
auto set_kernel_arg(const T &t) {
  return t;
}

auto set_kernel_arg(EmptySpecial &a) {
  return a.data;
}

template<typename KernelName, typename... Ts>
auto sycl_handle_special_kernel_parameters(Ts...) {
  return [](auto ...Args){ return; };
}

template<typename... Ts>
struct type_list {};

template <typename KernelName, typename... Ts>
auto sycl_kernel_launch(const char *, Ts...) {

    return [&](auto&&... extra_host_args) {
      return type_list<decltype(set_kernel_arg(extra_host_args))...>{};
  };
}


template <typename KN, typename KT>
[[clang::sycl_kernel_entry_point(KN)]] void kernel_entry_point(KT Kernel) {
  Kernel();
}

void case1() {
    Wrapper<EmptySpecial> KernelArg;
    kernel_entry_point<KN<0>>([KernelArg](){});
}

// CHECK-HOST-LABEL: define internal void @_Z18kernel_entry_pointI2KNILi0EEZ5case1vEUlvE_EvT0_(
// CHECK-HOST-SAME: i32 [[KERNEL_COERCE0:%.*]], ptr [[KERNEL_COERCE1:%.*]])
// CHECK-HOST:  [[ENTRY:.*:]]
// CHECK-HOST-NEXT:    [[KERNEL:%.*]] = alloca [[CLASS_ANON:%.*]], align 8
// CHECK-HOST-NEXT:    [[REF_TMP:%.*]] = alloca [[CLASS_ANON_0:%.*]], align 1
// CHECK-HOST-NEXT:    [[AGG_TMP:%.*]] = alloca [[CLASS_ANON]], align 8
// CHECK-HOST-NEXT:    [[UNDEF_AGG_TMP:%.*]] = alloca [[CLASS_ANON_0]], align 1
// CHECK-HOST-NEXT:    [[UNDEF_AGG_TMP1:%.*]] = alloca [[STRUCT_TYPE_LIST:%.*]], align 1
// CHECK-HOST-NEXT:    [[TMP0:%.*]] = getelementptr inbounds nuw { i32, ptr }, ptr [[KERNEL]], i32 0, i32 0
// CHECK-HOST-NEXT:    store i32 [[KERNEL_COERCE0]], ptr [[TMP0]], align 8
// CHECK-HOST-NEXT:    [[TMP1:%.*]] = getelementptr inbounds nuw { i32, ptr }, ptr [[KERNEL]], i32 0, i32 1
// CHECK-HOST-NEXT:    store ptr [[KERNEL_COERCE1]], ptr [[TMP1]], align 8
// CHECK-HOST-NEXT:    call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[AGG_TMP]], ptr align 8 [[KERNEL]], i64 16, i1 false)
// CHECK-HOST-NEXT:    [[TMP2:%.*]] = getelementptr inbounds nuw { i32, ptr }, ptr [[AGG_TMP]], i32 0, i32 0
// CHECK-HOST-NEXT:    [[TMP3:%.*]] = load i32, ptr [[TMP2]], align 8
// CHECK-HOST-NEXT:    [[TMP4:%.*]] = getelementptr inbounds nuw { i32, ptr }, ptr [[AGG_TMP]], i32 0, i32 1
// CHECK-HOST-NEXT:    [[TMP5:%.*]] = load ptr, ptr [[TMP4]], align 8
// CHECK-HOST-NEXT:    call void @_Z18sycl_kernel_launchI2KNILi0EEJZ5case1vEUlvE_EEDaPKcDpT0_(ptr noundef @.str, i32 [[TMP3]], ptr [[TMP5]])
// CHECK-HOST-NEXT:    [[TMP6:%.*]] = getelementptr inbounds nuw [[CLASS_ANON]], ptr [[KERNEL]], i32 0, i32 0
// CHECK-HOST-NEXT:    [[DATA:%.*]] = getelementptr inbounds nuw [[STRUCT_WRAPPER:%.*]], ptr [[TMP6]], i32 0, i32 0
// CHECK-HOST-NEXT:    call void @_ZZ18sycl_kernel_launchI2KNILi0EEJZ5case1vEUlvE_EEDaPKcDpT0_ENKUlDpOT_E_clIJR12EmptySpecialEEEDaS9_(ptr noundef nonnull align 1 dereferenceable(1) [[REF_TMP]], ptr noundef nonnull align 4 dereferenceable(4) [[DATA]])
// CHECK-HOST-NEXT:    ret void

// CHECK-DEVICE: define spir_kernel void @_ZTS2KNILi0EE(ptr noundef byval(%class.anon) align 8 [[KERNEL:%.*]], i32 noundef [[IDK:%.*]])
// CHECK-DEVICE:  [[ENTRY:.*:]]
// CHECK-DEVICE:    [[IDK_ADDR:%.*]] = alloca i32, align 4
// CHECK-DEVICE:    [[REF_TMP:%.*]] = alloca [[CLASS_ANON_0:%.*]], align 1
// CHECK-DEVICE:    [[AGG_TMP:%.*]] = alloca [[STRUCT_EMPTYSPECIAL:%.*]], align 4
// CHECK-DEVICE:    [[IDK_ADDR_ASCAST:%.*]] = addrspacecast ptr [[IDK_ADDR]] to ptr addrspace(4)
// CHECK-DEVICE:    [[REF_TMP_ASCAST:%.*]] = addrspacecast ptr [[REF_TMP]] to ptr addrspace(4)
// CHECK-DEVICE:    [[AGG_TMP_ASCAST:%.*]] = addrspacecast ptr [[AGG_TMP]] to ptr addrspace(4)
// CHECK-DEVICE:    [[KERNEL_ASCAST:%.*]] = addrspacecast ptr [[KERNEL]] to ptr addrspace(4)
// CHECK-DEVICE:    store i32 [[IDK]], ptr addrspace(4) [[IDK_ADDR_ASCAST]], align 4
// CHECK-DEVICE:    [[REF_TMP_ASCAST_ASCAST:%.*]] = addrspacecast ptr addrspace(4) [[REF_TMP_ASCAST]] to ptr
// CHECK-DEVICE:    [[TMP0:%.*]] = getelementptr inbounds nuw [[CLASS_ANON:%.*]], ptr addrspace(4) [[KERNEL_ASCAST]], i32 0, i32 0
// CHECK-DEVICE:    [[DATA:%.*]] = getelementptr inbounds nuw [[STRUCT_WRAPPER:%.*]], ptr addrspace(4) [[TMP0]], i32 0, i32 0
// CHECK-DEVICE:    call void @llvm.memcpy.p4.p4.i64(ptr addrspace(4) align 4 [[AGG_TMP_ASCAST]], ptr addrspace(4) align 8 [[DATA]], i64 4, i1 false)
// CHECK-DEVICE:    [[AGG_TMP_ASCAST_ASCAST:%.*]] = addrspacecast ptr addrspace(4) [[AGG_TMP_ASCAST]] to ptr
// CHECK-DEVICE:    call spir_func void @_Z37sycl_handle_special_kernel_parametersI2KNILi0EEJ12EmptySpecialEEDaDpT0_(ptr dead_on_unwind writable sret([[CLASS_ANON_0]]) align 1 [[REF_TMP_ASCAST_ASCAST]], ptr noundef byval([[STRUCT_EMPTYSPECIAL]]) align 4 [[AGG_TMP_ASCAST_ASCAST]])
// CHECK-DEVICE:    [[TMP1:%.*]] = load i32, ptr addrspace(4) [[IDK_ADDR_ASCAST]], align 4
// CHECK-DEVICE:    call spir_func void @_ZZ37sycl_handle_special_kernel_parametersI2KNILi0EEJ12EmptySpecialEEDaDpT0_ENKUlDpT_E_clIJiEEEDaS6_(ptr addrspace(4) noundef align 1 dereferenceable_or_null(1) [[REF_TMP_ASCAST]], i32 noundef [[TMP1]])
// CHECK-DEVICE:    call spir_func void @_ZZ5case1vENKUlvE_clEv(ptr addrspace(4) noundef align 8 dereferenceable_or_null(16) [[KERNEL_ASCAST]])
// CHECK-DEVICE:    ret void

