// A basic clang -cc1 command-line, and simple environment check.

// RUN: %clang %s -### -no-canonical-prefixes -target riscv64 2>&1 | FileCheck -check-prefix=CC1 %s
// CC1: clang{{.*}} "-cc1" "-triple" "riscv64"

// RUN: %clang -target riscv64 %s -emit-llvm -S -o - | FileCheck %s

typedef __builtin_va_list va_list;
typedef __SIZE_TYPE__ size_t;
typedef __PTRDIFF_TYPE__ ptrdiff_t;
typedef __WCHAR_TYPE__ wchar_t;

// CHECK: @align_c = global i32 1
int align_c = __alignof(char);

// CHECK: @align_s = global i32 2
int align_s = __alignof(short);

// CHECK: @align_i = global i32 4
int align_i = __alignof(int);

// CHECK: @align_wc = global i32 4
int align_wc = __alignof(wchar_t);

// CHECK: @align_l = global i32 8
int align_l = __alignof(long);

// CHECK: @align_ll = global i32 8
int align_ll = __alignof(long long);

// CHECK: @align_p = global i32 8
int align_p = __alignof(void*);

// CHECK: @align_f = global i32 4
int align_f = __alignof(float);

// CHECK: @align_d = global i32 8
int align_d = __alignof(double);

// CHECK: @align_ld = global i32 16
int align_ld = __alignof(long double);

// CHECK: @align_vl = global i32 8
int align_vl = __alignof(va_list);
