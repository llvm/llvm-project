// RUN: %clang_cc1 -Werror -triple i386-linux -Wno-strict-prototypes -emit-llvm -o - %s | FileCheck %s

// Test that different order of declarations is acceptable and that
// implementing different redeclarations is acceptable.

typedef union {
  int i;
  float f;
} TU __attribute__((transparent_union));

// CHECK-LABEL: define{{.*}} void @f0(i32 %tu.coerce)
// CHECK: %tu = alloca %union.TU, align 4
// CHECK: %coerce.dive = getelementptr inbounds nuw %union.TU, ptr %tu, i32 0, i32 0
// CHECK: store i32 %tu.coerce, ptr %coerce.dive, align 4
void f0(TU tu) {}
void f0(int i);

// CHECK-LABEL: define{{.*}} void @f1(i32 noundef %tu.coerce)
// CHECK: %tu = alloca %union.TU, align 4
// CHECK: %coerce.dive = getelementptr inbounds nuw %union.TU, ptr %tu, i32 0, i32 0
// CHECK: store i32 %tu.coerce, ptr %coerce.dive, align 4
void f1(int i);
void f1(TU tu) {}

// CHECK-LABEL: define{{.*}} void @f2(i32 noundef %i)
// CHECK: %i.addr = alloca i32, align 4
// CHECK: store i32 %i, ptr %i.addr, align 4
void f2(TU tu);
void f2(int i) {}

// CHECK-LABEL: define{{.*}} void @f3(i32 noundef %i)
// CHECK: %i.addr = alloca i32, align 4
// CHECK: store i32 %i, ptr %i.addr, align 4
void f3(int i) {}
void f3(TU tu);

// Also test functions with parameters specified K&R style.
// CHECK-LABEL: define{{.*}} void @knrStyle(i32 noundef %tu.coerce)
// CHECK: %tu = alloca %union.TU, align 4
// CHECK: %coerce.dive = getelementptr inbounds nuw %union.TU, ptr %tu, i32 0, i32 0
// CHECK: store i32 %tu.coerce, ptr %coerce.dive, align 4
void knrStyle(int i);
void knrStyle(tu) TU tu; {}
