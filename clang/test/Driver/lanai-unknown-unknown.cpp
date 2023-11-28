// RUN: %clang --target=lanai-unknown-unknown -### %s -c 2>&1 \
// RUN:   | FileCheck %s -check-prefix=ECHO
// RUN: %clang --target=lanai-unknown-unknown %s -emit-llvm -S -o - \
// RUN:   | FileCheck %s

// ECHO: {{.*}} "-cc1" {{.*}}lanai-unknown-unknown.c

typedef __builtin_va_list va_list;
typedef __SIZE_TYPE__ size_t;
typedef __PTRDIFF_TYPE__ ptrdiff_t;

extern "C" {

// CHECK: @align_c = dso_local global i32 1
int align_c = __alignof(char);

// CHECK: @align_s = dso_local global i32 2
int align_s = __alignof(short);

// CHECK: @align_i = dso_local global i32 4
int align_i = __alignof(int);

// CHECK: @align_l = dso_local global i32 4
int align_l = __alignof(long);

// CHECK: @align_ll = dso_local global i32 8
int align_ll = __alignof(long long);

// CHECK: @align_p = dso_local global i32 4
int align_p = __alignof(void*);

// CHECK: @align_vl = dso_local global i32 4
int align_vl = __alignof(va_list);

// Check types

// CHECK: signext i8 @check_char()
char check_char() { return 0; }

// CHECK: signext i16 @check_short()
short check_short() { return 0; }

// CHECK: i32 @check_int()
int check_int() { return 0; }

// CHECK: i32 @check_long()
long check_long() { return 0; }

// CHECK: i64 @check_longlong()
long long check_longlong() { return 0; }

// CHECK: zeroext i8 @check_uchar()
unsigned char check_uchar() { return 0; }

// CHECK: zeroext i16 @check_ushort()
unsigned short check_ushort() { return 0; }

// CHECK: i32 @check_uint()
unsigned int check_uint() { return 0; }

// CHECK: i32 @check_ulong()
unsigned long check_ulong() { return 0; }

// CHECK: i64 @check_ulonglong()
unsigned long long check_ulonglong() { return 0; }

// CHECK: i32 @check_size_t()
size_t check_size_t() { return 0; }

}

template<int> void Switch();
template<> void Switch<4>();
template<> void Switch<8>();
template<> void Switch<16>();

void check_pointer_size() {
  // CHECK: SwitchILi4
  Switch<sizeof(void*)>();

  // CHECK: SwitchILi8
  Switch<sizeof(long long)>();

  // CHECK: SwitchILi4
  Switch<sizeof(va_list)>();
}
