// RUN: %clang_cc1 -no-opaque-pointers -ffixed-point -S -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -ffixed-point -S -emit-llvm %s -o - -fpadding-on-unsigned-fixed-point | FileCheck %s

_Bool global_b = 1.0k;  // @global_b = {{*.}}global i8 1, align 1
_Bool global_b2 = 0.0k; // @global_b2 = {{*.}}global i8 0, align 1

void func(void) {
  _Accum a = 0.5k;
  unsigned _Accum ua = 0.5uk;
  _Bool b;

  // CHECK: store i8 1, i8* %b, align 1
  // CHECK-NEXT: store i8 0, i8* %b, align 1
  // CHECK: store i8 1, i8* %b, align 1
  // CHECK-NEXT: store i8 0, i8* %b, align 1
  b = 0.5k;
  b = 0.0k;
  b = 0.5uk;
  b = 0.0uk;

  // CHECK-NEXT: store i8 1, i8* %b, align 1
  // CHECK-NEXT: store i8 0, i8* %b, align 1
  // CHECK-NEXT: store i8 1, i8* %b, align 1
  // CHECK-NEXT: store i8 0, i8* %b, align 1
  b = (_Bool)0.5r;
  b = (_Bool)0.0r;
  b = (_Bool)0.5ur;
  b = (_Bool)0.0ur;

  // CHECK-NEXT: [[ACCUM:%[0-9a-z]+]] = load i32, i32* %a, align 4
  // CHECK-NEXT: [[NOTZERO:%[0-9a-z]+]] = icmp ne i32 [[ACCUM]], 0
  // CHECK-NEXT: [[FROMBOOL:%[0-9a-z]+]] = zext i1 [[NOTZERO]] to i8
  // CHECK-NEXT: store i8 [[FROMBOOL]], i8* %b, align 1
  b = a;

  // CHECK-NEXT: [[ACCUM:%[0-9a-z]+]] = load i32, i32* %ua, align 4
  // CHECK-NEXT: [[NOTZERO:%[0-9a-z]+]] = icmp ne i32 [[ACCUM]], 0
  // CHECK-NEXT: [[FROMBOOL:%[0-9a-z]+]] = zext i1 [[NOTZERO]] to i8
  // CHECK-NEXT: store i8 [[FROMBOOL]], i8* %b, align 1
  b = ua;

  // CHECK-NEXT: [[ACCUM:%[0-9a-z]+]] = load i32, i32* %a, align 4
  // CHECK-NEXT: [[NOTZERO:%[0-9a-z]+]] = icmp ne i32 [[ACCUM]], 0
  // CHECK-NEXT: br i1 [[NOTZERO]], label %if.then, label %if.end
  if (a) {
  }

  // CHECK:      [[ACCUM:%[0-9a-z]+]] = load i32, i32* %ua, align 4
  // CHECK-NEXT: [[NOTZERO:%[0-9a-z]+]] = icmp ne i32 [[ACCUM]], 0
  // CHECK-NEXT: br i1 [[NOTZERO]], label %if.then{{[0-9]+}}, label %if.end{{[0-9]+}}
  if (ua) {
  }
}
