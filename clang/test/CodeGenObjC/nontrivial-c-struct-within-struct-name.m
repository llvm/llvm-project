// RUN: %clang_cc1 -triple arm64-apple-ios11 -fobjc-arc -emit-llvm -o - %s | FileCheck %s

@class I;

typedef struct {
  I *name;
} Foo;

typedef struct {
  Foo foo;
} Bar;

typedef struct {
  Bar bar;
} Baz;

I *getI(void);

void f(void) {
  Foo foo = {getI()};
  Bar bar = {foo};
  Baz baz = {bar};
}

// CHECK: define linkonce_odr hidden void @__destructor_8_S_S_s0(ptr noundef %[[DST:.*]])
// CHECK: %[[DST_ADDR:.*]] = alloca ptr, align 8
// CHECK: store ptr %[[DST]], ptr %[[DST_ADDR]], align 8
// CHECK: %[[V0:.*]] = load ptr, ptr %[[DST_ADDR]], align 8
// CHECK: call void @__destructor_8_S_s0(ptr %[[V0]])
// CHECK: ret void

// CHECK: define linkonce_odr hidden void @__destructor_8_S_s0(ptr noundef %[[DST:.*]])
// CHECK: %[[DST_ADDR:.*]] = alloca ptr, align 8
// CHECK: store ptr %[[DST]], ptr %[[DST_ADDR]], align 8
// CHECK: %[[V0:.*]] = load ptr, ptr %[[DST_ADDR]], align 8
// CHECK: call void @__destructor_8_s0(ptr %[[V0]])
// CHECK: ret void

// CHECK: define linkonce_odr hidden void @__destructor_8_s0(ptr noundef %dst)
// CHECK: %[[DST_ADDR:.*]] = alloca ptr, align 8
// CHECK: store ptr %[[DST]], ptr %[[DST_ADDR]], align 8
// CHECK: %[[V0:.*]] = load ptr, ptr %[[DST_ADDR]], align 8
// CHECK: call void @llvm.objc.storeStrong(ptr %[[V0]], ptr null)
// CHECK: ret void
