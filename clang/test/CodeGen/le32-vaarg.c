// RUN: %clang_cc1 -triple le32-unknown-nacl -emit-llvm -o - %s | FileCheck %s
#include <stdarg.h>

int get_int(va_list *args) {
  return va_arg(*args, int);
}
// CHECK: define i32 @get_int
// CHECK: [[RESULT:%[a-z_0-9]+]] = va_arg {{.*}}, i32{{$}}
// CHECK: store i32 [[RESULT]], i32* [[LOC:%[a-z_0-9]+]]
// CHECK: [[RESULT2:%[a-z_0-9]+]] = load i32, i32* [[LOC]]
// CHECK: ret i32 [[RESULT2]]

struct Foo {
  int x;
};

struct Foo dest;

void get_struct(va_list *args) {
  dest = va_arg(*args, struct Foo);
}
// CHECK: define void @get_struct
// CHECK: [[RESULT:%[a-z_0-9]+]] = va_arg {{.*}}, %struct.Foo{{$}}
// CHECK: store %struct.Foo [[RESULT]], %struct.Foo* [[LOC:%[a-z_0-9]+]]
// CHECK: [[LOC2:%[a-z_0-9]+]] = bitcast {{.*}} [[LOC]] to i8*
// CHECK: call void @llvm.memcpy{{.*}}@dest{{.*}}, i8* align {{[0-9]+}} [[LOC2]]

void skip_struct(va_list *args) {
  va_arg(*args, struct Foo);
}
// CHECK: define void @skip_struct
// CHECK: va_arg {{.*}}, %struct.Foo{{$}}
