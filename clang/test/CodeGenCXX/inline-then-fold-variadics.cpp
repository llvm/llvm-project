// RUN: %clang_cc1 -triple i386-unknown-linux-gnu -Wno-varargs -O1 -disable-llvm-passes -emit-llvm -o - %s | opt --passes=instcombine | opt -passes="expand-variadics,default<O1>" -S | FileCheck %s --check-prefixes=CHECK,X86Linux

// RUN: %clang_cc1 -triple x86_64-linux-gnu -Wno-varargs -O1 -disable-llvm-passes -emit-llvm -o - %s | opt --passes=instcombine | opt -passes="expand-variadics,default<O1>" -S | FileCheck %s --check-prefixes=CHECK,X64SystemV

// RUN: %clang_cc1 -triple i386-apple-darwin -Wno-varargs -O1 -disable-llvm-passes -emit-llvm -o - %s | opt --passes=instcombine | opt -passes="expand-variadics,default<O1>" -S | FileCheck %s --check-prefixes=CHECK,X86Darwin

// RUN: %clang_cc1 -triple x86_64-apple-darwin -Wno-varargs -O1 -disable-llvm-passes -emit-llvm -o - %s | opt --passes=instcombine | opt -passes="expand-variadics,default<O1>" -S | FileCheck %s --check-prefixes=CHECK,X64SystemV

// RUN: %clang_cc1 -triple i686-windows-msvc -Wno-varargs -O1 -disable-llvm-passes -emit-llvm -o - %s | opt --passes=instcombine | opt -passes="expand-variadics,default<O1>" -S | FileCheck %s --check-prefixes=CHECK,X86Windows

// 64 bit windows va_arg passes most types indirectly but the call instruction uses the types by value
// ___: %clang_cc1 -triple x86_64-pc-windows-msvc -Wno-varargs -O1 -disable-llvm-passes -emit-llvm -o - %s | opt --passes=instcombine | opt -passes="expand-variadics,default<O1>" -S | FileCheck %s --check-prefixes=CHECK

// Checks for consistency between clang and expand-variadics
// 1. Use clang to lower va_arg
// 2. Use expand-variadics to lower the rest of the variadic operations
// 3. Use opt -O1 to simplify the result for simpler filecheck patterns
// The simplification will fail when the two are not consistent, modulo bugs elsewhere.

#include <stdarg.h>

// This test can be simplified when expand-variadics is extended to apply to more patterns.
// The first_valist and second_valist functions can then be inlined, either in the test or
// by enabling optimisaton passes in the clang invocation.
// The explicit instcombine pass canonicalises the variadic function IR.

// More complicated tests need instcombine of ptrmask to land first.

template <typename X, typename Y>
static X first_valist(va_list va) {
  return va_arg(va, X);
}

template <typename X, typename Y>
static X first(...) {
  va_list va;
  __builtin_va_start(va, 0);
  return first_valist<X,Y>(va);
}

template <typename X, typename Y>
static Y second_valist(va_list va) {
  va_arg(va, X);
  Y r = va_arg(va, Y);
  return r;
}


template <typename X, typename Y>
static Y second(...) {
  va_list va;
  __builtin_va_start(va, 0);
  return second_valist<X,Y>(va);
}

extern "C"
{
// CHECK-LABEL: define{{.*}} i32 @first_i32_i32(i32{{.*}} %x, i32{{.*}} %y)
// CHECK:       entry:
// CHECK:       ret i32 %x
int first_i32_i32(int x, int y)
{
  return first<int,int>(x, y);
}

// CHECK-LABEL: define{{.*}} i32 @second_i32_i32(i32{{.*}} %x, i32{{.*}} %y)
// CHECK:       entry:
// CHECK:       ret i32 %y
int second_i32_i32(int x, int y)
{
  return second<int,int>(x, y);
}
}

// Permutations of an int and a double
extern "C"
{
// CHECK-LABEL: define{{.*}} i32 @first_i32_f64(i32{{.*}} %x, double{{.*}} %y)
// CHECK:       entry:
// CHECK:       ret i32 %x
int first_i32_f64(int x, double y)
{
  return first<int,double>(x, y);
}
  
// CHECK-LABEL: define{{.*}} double @second_i32_f64(i32{{.*}} %x, double{{.*}} %y)
// CHECK:       entry:

// X86Linux:    ret double %y
// X64SystemV:  ret double %y
// X86Darwin:   ret double %y
// X86Windows:  [[TMP0:%.*]] = alloca <{ [4 x i8], double }>, align 4
// X86Windows:  [[TMP1:%.*]] = getelementptr inbounds i8, ptr [[TMP0]], i32 4
// X86Windows:  store double %y, ptr [[TMP1]], align 4
// X86Windows:  [[TMP2:%.*]] = load double, ptr [[TMP0]], align 4
// X86Windows:  ret double [[TMP2]]
double second_i32_f64(int x, double y)
{
  return second<int,double>(x, y);
}

// CHECK-LABEL: define{{.*}} double @first_f64_i32(double{{.*}} %x, i32{{.*}} %y)
// CHECK:       entry:
// CHECK:       ret double %x
double first_f64_i32(double x, int y)
{
  return first<double,int>(x, y);
}

// CHECK-LABEL: define{{.*}} i32 @second_f64_i32(double{{.*}} %x, i32{{.*}} %y)
// CHECK:       entry:
// CHECK:       ret i32 %y
int second_f64_i32(double x, int y)
{
  return second<double,int>(x, y);
}   
}
