// RUN: %clang_cc1 -std=c++11 -fptrauth-intrinsics -fptrauth-calls -emit-llvm -o - -triple=arm64-apple-ios %s | FileCheck %s
// RUN: %clang_cc1 -std=c++11 -fptrauth-intrinsics -fptrauth-calls -emit-llvm -o - -triple=aarch64-linux-gnu %s | FileCheck %s
// RUN: %clang_cc1 -std=c++11 -fptrauth-intrinsics -fptrauth-calls -emit-llvm -o - -triple=arm64-apple-ios -fclang-abi-compat=4 %s | FileCheck %s

// clang previously emitted an incorrect discriminator for the member function
// pointer because of a bug in the mangler.

// CHECK: @_ZN17test_substitution5funcsE = global [1 x { i64, i64 }] [{ i64, i64 } { i64 ptrtoint (ptr ptrauth (ptr @_ZN17test_substitution1S1fEPvS1_, i32 0, i64 48995) to i64), i64 0 }], align 8
namespace test_substitution {
struct S { int f(void *, void *); };

typedef int (S::*s_func)(void *, void *);

s_func funcs[] = { (s_func)(&S::f) };
}


// CHECK: define {{.*}}void @_Z3fooPU9__ptrauthILj3ELb1ELj234EEPi(
void foo(int * __ptrauth(3, 1, 234) *) {}

template <class T>
void foo(T t) {}

// CHECK: define weak_odr void @_Z3fooIPU9__ptrauthILj1ELb0ELj64EEPiEvT_(
template void foo<int * __ptrauth(1, 0, 64) *>(int * __ptrauth(1, 0, 64) *);

