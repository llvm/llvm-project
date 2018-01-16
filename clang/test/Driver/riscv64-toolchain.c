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

// Check types

// CHECK: define zeroext i8 @check_char()
char check_char() { return 0; }

// CHECK: define signext i16 @check_short()
short check_short() { return 0; }

// CHECK: define signext i32 @check_int()
int check_int() { return 0; }

// CHECK: define signext i32 @check_wchar_t()
int check_wchar_t() { return 0; }

// CHECK: define i64 @check_long()
long check_long() { return 0; }

// CHECK: define i64 @check_longlong()
long long check_longlong() { return 0; }

// CHECK: define zeroext i8 @check_uchar()
unsigned char check_uchar() { return 0; }

// CHECK: define zeroext i16 @check_ushort()
unsigned short check_ushort() { return 0; }

// CHECK: define signext i32 @check_uint()
unsigned int check_uint() { return 0; }

// CHECK: define i64 @check_ulong()
unsigned long check_ulong() { return 0; }

// CHECK: define i64 @check_ulonglong()
unsigned long long check_ulonglong() { return 0; }

// CHECK: define i64 @check_size_t()
size_t check_size_t() { return 0; }

// CHECK: define float @check_float()
float check_float() { return 0; }

// CHECK: define double @check_double()
double check_double() { return 0; }

// CHECK: define fp128 @check_longdouble()
long double check_longdouble() { return 0; }
