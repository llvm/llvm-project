// RUN: %clang_cc1 -triple x86_64-linux-pc -DREDECL -emit-llvm %s -o - | FileCheck %s -check-prefix LINUX
// RUN: %clang_cc1 -triple spir-unknown-unknown -DREDECL -DSPIR -emit-llvm %s -o - | FileCheck %s -check-prefix SPIR
// RUN: %clang_cc1 -triple x86_64-linux-pc -emit-llvm %s -o - | FileCheck %s -check-prefix LINUX
// RUN: %clang_cc1 -triple spir-unknown-unknown -DSPIR -emit-llvm %s -o - | FileCheck %s -check-prefix SPIR

#ifdef REDECL
namespace std {
#ifdef SPIR
using size_t = unsigned int;
#else
using size_t = unsigned long;
#endif // SPIR
} // namespace std

float __builtin_atan2f(float, float);
void *operator new(std::size_t);
#endif // REDECL

void foo();

void user() {
  int i;
  ::operator new(5);
  (void)__builtin_atan2f(1.1, 2.2);
  foo();
}

// LINUX: define void @_Z4userv()
// LINUX: call i8* @_Znwm
// LINUX: call float @atan2f
// LINUX: call void @_Z3foov
// LINUX: declare noalias i8* @_Znwm(i64)
// LINUX: declare float @atan2f(float, float)
// LINUX: declare void @_Z3foov()

// SPIR: define spir_func void @_Z4userv()
// SPIR: call spir_func i8* @_Znwj
// SPIR: call spir_func float @atan2f
// SPIR: call spir_func void @_Z3foov
// SPIR: declare spir_func noalias i8* @_Znwj(i32)
// SPIR: declare spir_func float @atan2f(float, float)
// SPIR: declare spir_func void @_Z3foov()
