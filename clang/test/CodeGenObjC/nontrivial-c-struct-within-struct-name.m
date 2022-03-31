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

// CHECK: define linkonce_odr hidden void @__destructor_8_S_S_s0(i8** noundef %[[DST:.*]])
// CHECK: %[[DST_ADDR:.*]] = alloca i8**, align 8
// CHECK: store i8** %[[DST]], i8*** %[[DST_ADDR]], align 8
// CHECK: %[[V0:.*]] = load i8**, i8*** %[[DST_ADDR]], align 8
// CHECK: call void @__destructor_8_S_s0(i8** %[[V0]])
// CHECK: ret void

// CHECK: define linkonce_odr hidden void @__destructor_8_S_s0(i8** noundef %[[DST:.*]])
// CHECK: %[[DST_ADDR:.*]] = alloca i8**, align 8
// CHECK: store i8** %[[DST]], i8*** %[[DST_ADDR]], align 8
// CHECK: %[[V0:.*]] = load i8**, i8*** %[[DST_ADDR]], align 8
// CHECK: call void @__destructor_8_s0(i8** %[[V0]])
// CHECK: ret void

// CHECK: define linkonce_odr hidden void @__destructor_8_s0(i8** noundef %dst)
// CHECK: %[[DST_ADDR:.*]] = alloca i8**, align 8
// CHECK: store i8** %[[DST]], i8*** %[[DST_ADDR]], align 8
// CHECK: %[[V0:.*]] = load i8**, i8*** %[[DST_ADDR]], align 8
// CHECK: call void @llvm.objc.storeStrong(i8** %[[V0]], i8* null)
// CHECK: ret void
