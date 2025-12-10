// REQUIRES: arm-registered-target
// RUN: %clang_cc1 -triple aarch64-none-elf \
// RUN:   -O2 \
// RUN:   -emit-llvm -fexperimental-max-bitint-width=1024 -o - %s | FileCheck %s

extern "C" {

// CHECK: @sizeof_OverSizedBitfield ={{.*}} global i32 8
// CHECK: @alignof_OverSizedBitfield ={{.*}} global i32 8
// CHECK: @sizeof_VeryOverSizedBitfield ={{.*}} global i32 16
// CHECK: @alignof_VeryOverSizedBitfield ={{.*}} global i32 16
// CHECK: @sizeof_RidiculouslyOverSizedBitfield ={{.*}} global i32 32
// CHECK: @alignof_RidiculouslyOverSizedBitfield ={{.*}} global i32 16

//BitInt alignment
struct BITINT129 {
    char ch;
    unsigned _BitInt(129) v;
};

int test_bitint129(){
  return __builtin_offsetof(struct BITINT129, v);
}
// CHECK:  ret i32 16 

struct BITINT127 {
    char ch;
    _BitInt(127) v;
};

int test_bitint127(){
  return __builtin_offsetof(struct BITINT127, v);
}
// CHECK:  ret i32 16 

struct BITINT63 {
    char ch;
    _BitInt(63) v;
};

int test_bitint63(){
  return __builtin_offsetof(struct BITINT63, v);
}
// CHECK:  ret i32 8 

struct BITINT32 {
    char ch;
    unsigned _BitInt(32) v;
};

int test_bitint32(){
  return __builtin_offsetof(struct BITINT32, v);
}
// CHECK:  ret i32 4

struct BITINT9 {
    char ch;
    unsigned _BitInt(9) v;
};

int test_bitint9(){
  return __builtin_offsetof(struct BITINT9, v);
}
// CHECK:  ret i32 2

struct BITINT8 {
    char ch;
    unsigned _BitInt(8) v;
};

int test_bitint8(){
  return __builtin_offsetof(struct BITINT8, v);
}
// CHECK:  ret i32 1

// Over-sized bitfield, which results in a 64-bit container type, so 64-bit
// alignment.
struct OverSizedBitfield {
  int x : 64;
};

unsigned sizeof_OverSizedBitfield = sizeof(OverSizedBitfield);
unsigned alignof_OverSizedBitfield = alignof(OverSizedBitfield);

// CHECK: define{{.*}} void @g7
// CHECK: call void @f7(i32 noundef 1, i64 42)
// CHECK: declare void @f7(i32 noundef, i64)
void f7(int a, OverSizedBitfield b);
void g7() {
  OverSizedBitfield s = {42};
  f7(1, s);
}

// AAPCS64 does have a 128-bit integer fundamental data type, so this gets a
// 128-bit container with 128-bit alignment. This is just within the limit of
// what can be passed directly.
struct VeryOverSizedBitfield {
  int x : 128;
};

unsigned sizeof_VeryOverSizedBitfield = sizeof(VeryOverSizedBitfield);
unsigned alignof_VeryOverSizedBitfield = alignof(VeryOverSizedBitfield);

// CHECK: define{{.*}} void @g8
// CHECK: call void @f8(i32 noundef 1, i128 42)
// CHECK: declare void @f8(i32 noundef, i128)
void f8(int a, VeryOverSizedBitfield b);
void g8() {
  VeryOverSizedBitfield s = {42};
  f8(1, s);
}

// There are no bigger fundamental data types, so this gets a 128-bit container
// and 128 bits of padding, giving the struct a size of 32 bytes, and an
// alignment of 16 bytes. This is over the PCS size limit of 16 bytes, so it
// will be passed indirectly.
struct RidiculouslyOverSizedBitfield {
  int x : 256;
};

unsigned sizeof_RidiculouslyOverSizedBitfield = sizeof(RidiculouslyOverSizedBitfield);
unsigned alignof_RidiculouslyOverSizedBitfield = alignof(RidiculouslyOverSizedBitfield);

// CHECK: define{{.*}} void @g9
// CHECK: call void @f9(i32 noundef 1, ptr dead_on_return noundef nonnull %agg.tmp)
// CHECK: declare void @f9(i32 noundef, ptr dead_on_return noundef)
void f9(int a, RidiculouslyOverSizedBitfield b);
void g9() {
  RidiculouslyOverSizedBitfield s = {42};
  f9(1, s);
}

}

