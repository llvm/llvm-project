// RUN: %clang_cc1 -triple spir64-unknown-unknown -aux-triple x86_64-pc-windows-msvc -fsycl-is-device -disable-llvm-passes -fsycl-is-device -emit-llvm %s -o - | FileCheck %s '-D$ADDRSPACE=addrspace(1) '
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fsycl-is-device -disable-llvm-passes -fsycl-is-device -emit-llvm %s -o - | FileCheck %s '-D$ADDRSPACE='


template<typename KN, typename Func>
[[clang::sycl_kernel_entry_point(KN)]] void kernel(Func F){
  F();
}

template<typename Func>
void kernel_wrapper(Func F) {
  kernel<Func>(F);
}

template<typename KN, typename Func>
[[clang::sycl_kernel_entry_point(KN)]] void kernel2(Func F){
  F(1);
}

template<typename Func>
void kernel2_wrapper(Func F) {
  kernel2<Func>(F);
}

template<typename KN, typename Func>
[[clang::sycl_kernel_entry_point(KN)]] void kernel3(Func F){
  F(1.1);
}

template<typename Func>
void kernel3_wrapper(Func F) {
  kernel3<Func>(F);
}

int main() {
  int i;
  double d;
  float f;
  auto lambda1 = [](){};
  auto lambda2 = [](int){};
  auto lambda3 = [](double){};

  kernel_wrapper(lambda1);
  kernel2_wrapper(lambda2);
  kernel3_wrapper(lambda3);

  // Ensure the kernels are named the same between the device and host
  // invocations.
  kernel_wrapper([](){
  (void)__builtin_sycl_unique_stable_name(decltype(lambda1));
  (void)__builtin_sycl_unique_stable_name(decltype(lambda2));
  (void)__builtin_sycl_unique_stable_name(decltype(lambda3));
  });

  // Make sure the following 3 are the same between the host and device compile.
  // Note that these are NOT the same value as each other, they differ by the
  // signature.
  // CHECK: private unnamed_addr [[$ADDRSPACE]]constant [17 x i8] c"_ZTSZ4mainEUlvE_\00"
  // CHECK: private unnamed_addr [[$ADDRSPACE]]constant [17 x i8] c"_ZTSZ4mainEUliE_\00"
  // CHECK: private unnamed_addr [[$ADDRSPACE]]constant [17 x i8] c"_ZTSZ4mainEUldE_\00"

  // On Windows, ensure that we haven't broken the 'lambda numbering' for the
  // lambda itself.
  // WIN: define internal void @"??R<lambda_1
  // WIN: define internal void @"??R<lambda_2
  // WIN: define internal void @"??R<lambda_3
  // WIN: define internal void @"??R<lambda_4
}
