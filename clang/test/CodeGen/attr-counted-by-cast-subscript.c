// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O0 -fsanitize=array-bounds -fsanitize-trap=array-bounds -emit-llvm -o - %s | FileCheck %s

// When a 'counted_by' pointer is cast before it's indexed, the index counts
// elements of the cast-to type, while the count counts elements of the
// field's pointee type. -fsanitize=array-bounds must compare like with like:
// the count is converted to bytes and then to elements of the accessed type.

struct ints {
  int count;
  int *buf __attribute__((counted_by(count)));
};

struct uints {
  unsigned short count;
  int *buf __attribute__((counted_by(count)));
};

// Accessing bytes of an 'int' buffer: the bound is count * 4 bytes, computed
// in a 64-bit signed type so it can't overflow.
// CHECK-LABEL: define {{.*}} @test_char(
// CHECK:       [[COUNT:%.counted_by.load]] = load i32, ptr
// CHECK:       [[WIDE:%.*]] = sext i32 [[COUNT]] to i64
// CHECK:       [[BYTES:%.*]] = mul i64 [[WIDE]], 4
// CHECK:       icmp ult i64 {{%.*}}, [[BYTES]]
// CHECK:       icmp sgt i64 [[BYTES]], 0
// CHECK:       call void @llvm.ubsantrap(
char test_char(struct ints *p, int i) {
  return ((char *)p->buf)[i];
}

// Taking an address may point one past the end, so the comparison is 'ule'.
// CHECK-LABEL: define {{.*}} @test_char_address(
// CHECK:       [[COUNT:%.counted_by.load]] = load i32, ptr
// CHECK:       [[WIDE:%.*]] = sext i32 [[COUNT]] to i64
// CHECK:       [[BYTES:%.*]] = mul i64 [[WIDE]], 4
// CHECK:       icmp ule i64 {{%.*}}, [[BYTES]]
char *test_char_address(struct ints *p, int i) {
  return &((char *)p->buf)[i];
}

// Accessing 8-byte elements of an 'int' buffer: only whole elements count, so
// the bound is count * 4 / 8, rounded down. Signed division keeps a negative
// count non-positive, so it's still rejected.
// CHECK-LABEL: define {{.*}} @test_long(
// CHECK:       [[COUNT:%.counted_by.load]] = load i32, ptr
// CHECK:       [[WIDE:%.*]] = sext i32 [[COUNT]] to i64
// CHECK:       [[BYTES:%.*]] = mul i64 [[WIDE]], 4
// CHECK:       [[ELTS:%.*]] = sdiv i64 [[BYTES]], 8
// CHECK:       icmp ult i64 {{%.*}}, [[ELTS]]
// CHECK:       icmp sgt i64 [[ELTS]], 0
long test_long(struct ints *p, int i) {
  return ((long *)p->buf)[i];
}

// An unsigned count is converted with unsigned arithmetic.
// CHECK-LABEL: define {{.*}} @test_unsigned_count(
// CHECK:       [[COUNT:%.counted_by.load]] = load i16, ptr
// CHECK:       [[WIDE:%.*]] = zext i16 [[COUNT]] to i64
// CHECK:       [[BYTES:%.*]] = mul i64 [[WIDE]], 4
// CHECK:       [[ELTS:%.*]] = udiv i64 [[BYTES]], 8
// CHECK:       icmp ult i64 {{%.*}}, [[ELTS]]
long test_unsigned_count(struct uints *p, int i) {
  return ((long *)p->buf)[i];
}

// A cast to a type of the same size leaves the bound alone.
// CHECK-LABEL: define {{.*}} @test_same_size(
// CHECK:       [[COUNT:%.counted_by.load]] = load i32, ptr
// CHECK-NOT:   mul
// CHECK:       icmp ult i32 {{%.*}}, [[COUNT]]
unsigned test_same_size(struct ints *p, int i) {
  return ((unsigned *)p->buf)[i];
}

// Without a cast, the count is used as is.
// CHECK-LABEL: define {{.*}} @test_no_cast(
// CHECK:       [[COUNT:%.counted_by.load]] = load i32, ptr
// CHECK-NOT:   mul
// CHECK:       icmp ult i32 {{%.*}}, [[COUNT]]
int test_no_cast(struct ints *p, int i) {
  return p->buf[i];
}
