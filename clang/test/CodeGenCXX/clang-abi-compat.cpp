// RUN: %clang_cc1 -std=c++17 -triple x86_64-linux-gnu -fclang-abi-compat=3.0 %s -emit-llvm -o - | FileCheck --check-prefix=PRE39 --check-prefix=PRE5 %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-linux-gnu -fclang-abi-compat=3.8 %s -emit-llvm -o - | FileCheck --check-prefix=PRE39 --check-prefix=PRE5 %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-linux-gnu -fclang-abi-compat=3.9 %s -emit-llvm -o - | FileCheck --check-prefix=V39 --check-prefix=PRE5 %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-linux-gnu -fclang-abi-compat=4.0 %s -emit-llvm -o - | FileCheck --check-prefix=V39 --check-prefix=PRE5 %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-linux-gnu -fclang-abi-compat=5 %s -emit-llvm -o - | FileCheck --check-prefix=V39 --check-prefix=V5 %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-linux-gnu -fclang-abi-compat=latest %s -emit-llvm -o - | FileCheck --check-prefix=V39 --check-prefix=V5 %s

typedef __attribute__((vector_size(8))) long long v1xi64;
void clang39(v1xi64) {}
// PRE39: @_Z7clang39Dv1_x(i64
// V39: @_Z7clang39Dv1_x(double

struct A {
  A(const A&) = default;
  A(A&&);
};
void clang5(A) {}
// PRE5: @_Z6clang51A()
// V5: @_Z6clang51A(%{{.*}}*
