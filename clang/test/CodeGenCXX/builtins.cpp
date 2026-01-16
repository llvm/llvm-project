// RUN: %clang_cc1 -triple=x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s

// Builtins inside a namespace inside an extern "C" must be considered builtins.
extern "C" {
namespace X {
double __builtin_fabs(double);
float __builtin_fabsf(float) noexcept;
} // namespace X
}

int o = X::__builtin_fabs(-2.0);
// CHECK: @o ={{.*}} global i32 2, align 4

long p = X::__builtin_fabsf(-3.0f);
// CHECK: @p ={{.*}} global i64 3, align 8

int x = __builtin_abs(-2);
// CHECK: @x ={{.*}} global i32 2, align 4

long y = __builtin_abs(-2l);
// CHECK: @y ={{.*}} global i64 2, align 8

// PR8839
extern "C" char memmove();

int main() {
  // CHECK: call {{signext i8|i8}} @memmove()
  return memmove();
}

struct S;
// CHECK: define {{.*}} @_Z9addressofbR1SS0_(
S *addressof(bool b, S &s, S &t) {
  // CHECK: %[[LVALUE:.*]] = phi
  // CHECK: ret ptr %[[LVALUE]]
  return __builtin_addressof(b ? s : t);
}

namespace std { template<typename T> T *addressof(T &); }

// CHECK: define {{.*}} @_Z13std_addressofbR1SS0_(
S *std_addressof(bool b, S &s, S &t) {
  // CHECK: %[[LVALUE:.*]] = phi
  // CHECK: ret ptr %[[LVALUE]]
  return std::addressof(b ? s : t);
}

namespace std { template<typename T> T *__addressof(T &); }

// CHECK: define {{.*}} @_Z15std___addressofbR1SS0_(
S *std___addressof(bool b, S &s, S &t) {
  // CHECK: %[[LVALUE:.*]] = phi
  // CHECK: ret ptr %[[LVALUE]]
  return std::__addressof(b ? s : t);
}

extern "C" int __builtin_abs(int); // #1
long __builtin_abs(long);          // #2
extern "C" int __builtin_abs(int); // #3

extern const char char_memchr_arg[32];
char *memchr_result = __builtin_char_memchr(char_memchr_arg, 123, 32);
// CHECK: call ptr @memchr(ptr noundef @char_memchr_arg, i32 noundef 123, i64 noundef 32)

int constexpr_overflow_result() {
  constexpr int x = 1;
  // CHECK: alloca i32
  constexpr int y = 2;
  // CHECK: alloca i32
  int z;
  // CHECK: [[Z:%.+]] = alloca i32

  __builtin_sadd_overflow(x, y, &z);
  return z;
  // CHECK: [[RET_PTR:%.+]] = extractvalue { i32, i1 } %0, 0
  // CHECK: store i32 [[RET_PTR]], ptr [[Z]]
  // CHECK: [[RET_VAL:%.+]] = load i32, ptr [[Z]]
  // CHECK: ret i32 [[RET_VAL]]
}

int structured_binding_size() {
  struct S2 {int a, b;};
  return __builtin_structured_binding_size(S2);
  // CHECK: ret i32 2
}

void test_int_reference(int& a) {
  __builtin_bswapg(a);
}
// CHECK-LABEL: @_Z18test_int_referenceRi
// CHECK: store ptr %a, ptr
// CHECK: load ptr, ptr
// CHECK: load i32, ptr
// CHECK: call i32 @llvm.bswap.i32

void test_long_reference(long& a) {
  __builtin_bswapg(a);
}
// CHECK-LABEL: @_Z19test_long_referenceRl
// CHECK: store ptr %a, ptr
// CHECK: load ptr, ptr
// CHECK: load i64, ptr
// CHECK: call i64 @llvm.bswap.i64

void test_short_reference(short& a) {
  __builtin_bswapg(a);
}
// CHECK-LABEL: @_Z20test_short_referenceRs
// CHECK: store ptr %a, ptr
// CHECK: load ptr, ptr 
// CHECK: load i16, ptr
// CHECK: call i16 @llvm.bswap.i16

void test_char_reference(char& a) {
  __builtin_bswapg(a);
}
// CHECK-LABEL: @_Z19test_char_referenceRc
// CHECK: store ptr %a, ptr
// CHECK: load ptr, ptr
// CHECK-NOT: call i8 @llvm.bswap.i8
// CHECK: ret void

void test_bitint() {
  _BitInt(8) a = 0x12;
  __builtin_bswapg(a);
  _BitInt(16) b = 0x1234;
  __builtin_bswapg(b);
  _BitInt(32) c = 0x00001234;
  __builtin_bswapg(c);
  _BitInt(64) d = 0x0000000000001234;
  __builtin_bswapg(d);
  _BitInt(128) e = ~(_BitInt(128))0;
  __builtin_bswapg(e);
}
// CHECK-LABEL: @_Z11test_bitintv
// CHECK-NOT: call i8 @llvm.bswap.i8
// CHECK: call i16 @llvm.bswap.i16
// CHECK: call i32 @llvm.bswap.i32
// CHECK: call i64 @llvm.bswap.i64
// CHECK: call i128 @llvm.bswap.i128
// CHECK: ret void
