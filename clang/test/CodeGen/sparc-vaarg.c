// RUN: %clang_cc1 -no-opaque-pointers -triple sparc -emit-llvm -o - %s | FileCheck %s
#include <stdarg.h>

// CHECK-LABEL: define{{.*}} i32 @get_int
// CHECK: [[RESULT:%[a-z_0-9]+]] = va_arg {{.*}}, i32{{$}}
// CHECK: store i32 [[RESULT]], i32* [[LOC:%[a-z_0-9]+]]
// CHECK: [[RESULT2:%[a-z_0-9]+]] = load i32, i32* [[LOC]]
// CHECK: ret i32 [[RESULT2]]
int get_int(va_list *args) {
  return va_arg(*args, int);
}

struct Foo {
  int x;
};

struct Foo dest;

// CHECK-LABEL: define{{.*}} void @get_struct
// CHECK: [[RESULT:%[a-z_0-9]+]] = va_arg {{.*}}, %struct.Foo*{{$}}
// CHECK: [[RESULT2:%[a-z_0-9]+]] = bitcast {{.*}} [[RESULT]] to i8*
// CHECK: call void @llvm.memcpy{{.*}}@dest{{.*}}, i8* align {{[0-9]+}} [[RESULT2]]
void get_struct(va_list *args) {
 dest = va_arg(*args, struct Foo);
}

enum E { Foo_one = 1 };

enum E enum_dest;

// CHECK-LABEL: define{{.*}} void @get_enum
// CHECK: va_arg i8** {{.*}}, i32
void get_enum(va_list *args) {
  enum_dest = va_arg(*args, enum E);
}
