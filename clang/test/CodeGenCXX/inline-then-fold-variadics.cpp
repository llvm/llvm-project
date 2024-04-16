// RUN: %clang_cc1 -triple i386-unknown-linux-gnu -Wno-varargs -O1 -emit-llvm -o - %s | opt --passes=expand-variadics | opt -S -O1 | FileCheck %s --check-prefixes=CHECK,X86Linux


// RUN: %clang_cc1 -triple x86_64-linux-gnu -Wno-varargs -O1 -emit-llvm -o - %s | opt --passes=expand-variadics | opt -S -O1 | FileCheck %s --check-prefixes=CHECK,X64SystemV


// RUN: %clang_cc1 -triple i386-apple-darwin -Wno-varargs -O1 -emit-llvm -o - %s | opt --passes=expand-variadics | opt -S -O1 | FileCheck %s --check-prefixes=CHECK,X86Darwin

// RUN: %clang_cc1 -triple x86_64-apple-darwin -Wno-varargs -O1 -emit-llvm -o - %s | opt --passes=expand-variadics | opt -S -O1 | FileCheck %s --check-prefixes=CHECK,X64SystemV


// The clang test suite has _lots_ of windows related triples in it
// 'x86_64-pc-windows-msvc|i686-windows-msvc|thumbv7-windows|aarch64-windows|i686-windows|x86_64-windows|x86_64-unknown-windows-msvc|i386-windows-pc|x86_64--windows-msvc|i686--windows-msvc|x86_64-unknown-windows-gnu|i686-unknown-windows-msvc|i686-unknown-windows-gnu|arm64ec-pc-windows-msvc19.20.0|i686-pc-windows-msvc19.14.0|i686-pc-windows|x86_64--windows-gnu|i686--windows-gnu|thumbv7--windows|i386-windows|x86_64-unknown-windows-pc|i686--windows|x86_64--windows|i686-w64-windows-gnu'

// Might be detecting an inconsistency - maybe different alignment
// Presently failing on an unusual calling convention

// i686 windows emits suboptimal codegen. sroa removes a field from a struct which misaligns a field which blocks store/load forwarding
// RUN: %clang_cc1 -triple i686-windows-msvc -Wno-varargs -O1 -emit-llvm -o - %s | opt --passes=expand-variadics | opt -S -O1 | FileCheck %s --check-prefixes=CHECK,X86Windows


// 64 bit windows va_arg passes most type indirectly but the call instruction uses the types by value
// ___: %clang_cc1 -triple x86_64-pc-windows-msvc -Wno-varargs -O1 -emit-llvm -o - %s | opt --passes=expand-variadics | opt -S -O1 | FileCheck %s --check-prefixes=CHECK



// amdgpu emits a sequence of addrspace casts that aren't folded yet
// todo: match it anyway
// R-N: %clang_cc1 -triple amdgcn-amd-amdhsa -Wno-varargs -O1 -emit-llvm -o - %s | opt --passes=expand-variadics | opt -S -O1 | FileCheck %s

// Requires the instcombine patch that hasn't landed yet
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -Wno-varargs -O1 -emit-llvm -o - %s | opt --passes=expand-variadics | opt -S -O1 | FileCheck %s





// Not yet implemented on arm
// Also there are various x86 variants that should be in the triple

// Checks for consistency between clang and expand-va-intrinics
// 1. Use clang to lower va_arg
// 2. Use expand-variadics to lower the rest of the variadic operations
// 3. Use opt -O1 to simplify the functions to ret %arg
// The simplification to ret %arg will fail when the two are not consistent, modulo bugs elsewhere.

#include <stdarg.h>

template <typename X, typename Y>
static X first(...) {
  va_list va;
  __builtin_va_start(va, 0);
  X r = va_arg(va, X);
  va_end(va);
  return r;
}

template <typename X, typename Y>
static Y second(...) {
  va_list va;
  __builtin_va_start(va, 0);
  va_arg(va, X);
  Y r = va_arg(va, Y);
  va_end(va);
  return r;
}

typedef float float4 __attribute__((__vector_size__(16), __aligned__(16)));
typedef float float8 __attribute__((__vector_size__(32), __aligned__(32)));
typedef float float16 __attribute__((__vector_size__(64), __aligned__(64)));
typedef float float32 __attribute__((__vector_size__(128), __aligned__(128)));


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


// Permutations of an int and a float4
extern "C"
{

// CHECK-LABEL: define{{.*}} i32 @first_i32_v4f32(i32{{.*}} %x, ptr{{.*}} %y)
// CHECK:       entry:
// CHECK:       ret i32 %x
int first_i32_v4f32(int x, float4 * y)
{
  return first<int,float4>(x, *y);
}
  
// CHECK-LABEL: define{{.*}} void @second_i32_v4f32(i32{{.*}} %x, ptr{{.*}} %y, ptr{{.*}} %r)
// CHECK:       entry:
// X86Linux:    [[TMP0:%.*]] = load <4 x float>, ptr %y, align 16
// X86Linux:    store <4 x float> [[TMP0]], ptr %r, align 16
// X64SystemV:  [[TMP0:%.*]] = load <4 x float>, ptr %y, align 16
// X64SystemV:  store <4 x float> [[TMP0]], ptr %r, align 16
// X86Darwin:   [[TMP0:%.*]] = load <2 x i64>, ptr %y, align 16
// X86Darwin:   store <2 x i64> [[TMP0]], ptr %r, align 16
// X86Windows:  [[TMP0:%.*]] = alloca <{ [12 x i8], <4 x float> }>, align 4
// X86Windows:  [[TMP1:%.*]] = load <4 x float>, ptr %y, align 16
// X86Windows:  [[TMP2:%.*]] = getelementptr inbounds i8, ptr [[TMP0]], i32 12
// X86Windows:  store <4 x float> [[TMP1]], ptr [[TMP2]], align 4
// X86Windows:  [[TMP3:%.*]] = load <4 x float>, ptr [[TMP0]], align 4
// X86Windows:  store <4 x float> [[TMP3]], ptr %r, align 16
// CHECK:       ret void
void second_i32_v4f32(int x, float4 * y, float4* r)
{
  *r = second<int,float4>(x, *y);
}

    
// CHECK-LABEL: define{{.*}} void @first_v4f32_i32(ptr{{.*}} %x, i32{{.*}} %y, ptr{{.*}} %r)
// CHECK:       entry:
// X86Linux:    [[TMP0:%.*]] = load <4 x float>, ptr %x, align 16
// X86Linux:    store <4 x float> [[TMP0]], ptr %r, align 16
// X64SystemV:  [[TMP0:%.*]] = load <4 x float>, ptr %x, align 16
// X64SystemV:  store <4 x float> [[TMP0]], ptr %r, align 16
// X86Darwin:   [[TMP0:%.*]] = load <2 x i64>, ptr %x, align 16
// X86Darwin:   store <2 x i64> [[TMP0]], ptr %r, align 16
// CHECK:       ret void
  void first_v4f32_i32(float4* x, int y, float4* r)
{
 *r =first<float4,int>(*x, y);
}

// CHECK-LABEL: define{{.*}} i32 @second_v4f32_i32(ptr{{.*}} %x, i32{{.*}} %y)
// CHECK:       entry:
// CHECK:       ret i32 %y
int second_v4f32_i32(float4* x, int y)
{
  return second<float4,int>(*x, y);
}

}

// A large struct with awkwardly aligned fields

typedef struct {
  char c;
  short s;
  int i;
  long l;
  float f;
  double d;
} libcS;

extern "C"
{

// CHECK-LABEL: define{{.*}} i32 @first_i32_libcS(i32{{.*}} %x, ptr{{.*}} %y)
// CHECK:       entry:
// CHECK:       ret i32 %x
int first_i32_libcS(int x, libcS * y)
{
  return first<int,libcS>(x, *y);
}
  
// CHECK-LABEL: define{{.*}} void @second_i32_libcS(i32{{.*}} %x, ptr{{.*}} %y, ptr{{.*}} %r)
// CHECK:       entry:
// CHECK:       ret void
void second_i32_libcS(int x, libcS * y, libcS* r)
{
  *r = second<int,libcS>(x, *y);
}

    
// CHECK-LABEL: define{{.*}} void @first_libcS_i32(ptr{{.*}} %x, i32{{.*}} %y, ptr{{.*}} %r)
// CHECK:       entry:

  void first_libcS_i32(libcS* x, int y, libcS* r)
{
 *r =first<libcS,int>(*x, y);
}

// CHECK-LABEL: define{{.*}} i32 @second_libcS_i32(ptr{{.*}} %x, i32{{.*}} %y)
// CHECK:       entry:
// CHECK:       ret i32 %y
int second_libcS_i32(libcS* x, int y)
{
  return second<libcS,int>(*x, y);
}

  
}



            
