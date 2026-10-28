// RUN: %clang_cc1 -triple spir64-unknown-unknown -aux-triple x86_64-pc-windows-msvc -fsycl-is-device -disable-llvm-passes -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple spir64-unknown-unknown -aux-triple x86_64-pc-windows-msvc -fsycl-is-device -disable-llvm-passes -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM-CIR
// RUN: %clang_cc1 -triple spir64-unknown-unknown -aux-triple x86_64-pc-windows-msvc -fsycl-is-device -disable-llvm-passes -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

// TODO(cir): Add the host RUN lines (-triple x86_64-pc-windows-msvc
// -fsycl-is-host) once ClangIR implements Microsoft ABI string-literal
// mangling.

template<typename KN, typename... Ts>
void sycl_kernel_launch(const char *, Ts...) {}

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
  // CIR: cir.global "private" constant cir_private dso_local target_address_space(1) @__usn_str = #cir.const_array<"_ZTSZ4mainEUlvE_" : !cir.array<!s8i x 16>, trailing_zeros> : !cir.array<!s8i x 17> align(1)
  // CIR: cir.global "private" constant cir_private dso_local target_address_space(1) @__usn_str.1 = #cir.const_array<"_ZTSZ4mainEUliE_" : !cir.array<!s8i x 16>, trailing_zeros> : !cir.array<!s8i x 17> align(1)
  // CIR: cir.global "private" constant cir_private dso_local target_address_space(1) @__usn_str.2 = #cir.const_array<"_ZTSZ4mainEUldE_" : !cir.array<!s8i x 16>, trailing_zeros> : !cir.array<!s8i x 17> align(1)

  // ClangIR does not yet emit unnamed_addr on string literal globals.
  // LLVM-CIR: @__usn_str = private addrspace(1) constant [17 x i8] c"_ZTSZ4mainEUlvE_\00", align 1
  // LLVM-CIR: @__usn_str.1 = private addrspace(1) constant [17 x i8] c"_ZTSZ4mainEUliE_\00", align 1
  // LLVM-CIR: @__usn_str.2 = private addrspace(1) constant [17 x i8] c"_ZTSZ4mainEUldE_\00", align 1
  // OGCG: @__usn_str = private unnamed_addr addrspace(1) constant [17 x i8] c"_ZTSZ4mainEUlvE_\00", align 1
  // OGCG: @__usn_str.1 = private unnamed_addr addrspace(1) constant [17 x i8] c"_ZTSZ4mainEUliE_\00", align 1
  // OGCG: @__usn_str.2 = private unnamed_addr addrspace(1) constant [17 x i8] c"_ZTSZ4mainEUldE_\00", align 1
}
