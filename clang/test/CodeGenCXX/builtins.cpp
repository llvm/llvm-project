// RUN: %clang_cc1 -triple=x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s

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
  // CHECK: ret {{.*}}* %[[LVALUE]]
  return __builtin_addressof(b ? s : t);
}

extern "C" int __builtin_abs(int); // #1
long __builtin_abs(long);          // #2
extern "C" int __builtin_abs(int); // #3

int x = __builtin_abs(-2);
// CHECK:  store i32 2, i32* @x, align 4

long y = __builtin_abs(-2l);
// CHECK:  [[Y:%.+]] = call i64 @_Z13__builtin_absl(i64 -2)
// CHECK:  store i64 [[Y]], i64* @y, align 8

extern const char char_memchr_arg[32];
char *memchr_result = __builtin_char_memchr(char_memchr_arg, 123, 32);
// CHECK: call i8* @memchr(i8* getelementptr inbounds ([32 x i8], [32 x i8]* @char_memchr_arg, i32 0, i32 0), i32 123, i64 32)
