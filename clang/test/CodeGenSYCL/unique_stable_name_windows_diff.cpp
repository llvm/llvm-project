// RUN: %clang_cc1 -triple spir64-unknown-unknown -aux-triple x86_64-pc-windows-msvc -fsycl-is-device -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s --check-prefix=DEVICE
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fsycl-is-host -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s --check-prefix=HOST


template<typename KN, typename Func>
[[clang::sycl_kernel_entry_point(KN)]] void kernel(Func F){
  F();
}

template<typename KN, typename Func>
[[clang::sycl_kernel_entry_point(KN)]] void kernel2(Func F){
  F(1);
}

template<typename KN, typename Func>
[[clang::sycl_kernel_entry_point(KN)]] void kernel3(Func F){
  F(1.1);
}

int main() {
  int i;
  double d;
  float f;
  auto lambda1 = [](){};
  auto lambda2 = [](int){};
  auto lambda3 = [](double){};

  kernel<class K1>(lambda1);
  kernel2<class K2>(lambda2);
  kernel3<class K3>(lambda3);

  // Ensure the kernels are named the same between the device and host
  // invocations.
  // Call from host.
  (void)__builtin_sycl_unique_stable_name(decltype(lambda1));
  (void)__builtin_sycl_unique_stable_name(decltype(lambda2));
  (void)__builtin_sycl_unique_stable_name(decltype(lambda3));

  // Call from device.
  auto lambda4 = [](){
    (void)__builtin_sycl_unique_stable_name(decltype(lambda1));
    (void)__builtin_sycl_unique_stable_name(decltype(lambda2));
    (void)__builtin_sycl_unique_stable_name(decltype(lambda3));
  };
  kernel<class K4>(lambda4);

  // Make sure the following 3 are the same between the host and device compile.
  // Note that these are NOT the same value as each other, they differ by the
  // signature.
  // HOST: private unnamed_addr constant [17 x i8] c"_ZTSZ4mainEUlvE_\00"
  // HOST: private unnamed_addr constant [17 x i8] c"_ZTSZ4mainEUliE_\00"
  // HOST: private unnamed_addr constant [17 x i8] c"_ZTSZ4mainEUldE_\00"

  // DEVICE: private unnamed_addr addrspace(1) constant [17 x i8] c"_ZTSZ4mainEUlvE_\00"
  // DEVICE: private unnamed_addr addrspace(1) constant [17 x i8] c"_ZTSZ4mainEUliE_\00"
  // DEVICE: private unnamed_addr addrspace(1) constant [17 x i8] c"_ZTSZ4mainEUldE_\00"

  // DEVICE: define dso_local spir_kernel void @_ZTSZ4mainE2K1
  // DEVICE: call spir_func void @_ZZ4mainENKUlvE_clEv
  // DEVICE: define internal spir_func void @_ZZ4mainENKUlvE_clEv
  // DEVICE: define dso_local spir_kernel void @_ZTSZ4mainE2K2
  // DEVICE: call spir_func void @_ZZ4mainENKUliE_clEi
  // DEVICE: define internal spir_func void @_ZZ4mainENKUliE_clEi
  // DEVICE: define dso_local spir_kernel void @_ZTSZ4mainE2K3
  // DEVICE: call spir_func void @_ZZ4mainENKUldE_clEd
  // DEVICE: define internal spir_func void @_ZZ4mainENKUldE_clEd
  // DEVICE: define dso_local spir_kernel void @_ZTSZ4mainE2K4
  // DEVICE: call spir_func void @_ZZ4mainENKUlvE0_clEv
  // DEVICE: define internal spir_func void @_ZZ4mainENKUlvE0_clEv
}
