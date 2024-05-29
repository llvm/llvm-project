// RUN: %clang_cc1 -x objective-c++ -std=c++11 %s -triple x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s

// CHECK: [[NAME:@.*]] = private unnamed_addr constant [9 x i8] c"position\00"
// CHECK: [[SEL:@.*]] = internal externally_initialized global ptr [[NAME]]

@interface I {
  int position;
}
@property(nonatomic) int position;
@end

struct S {
  void *operator new(__SIZE_TYPE__, int);
};

template <typename T>
struct TS {
  void *operator new(__SIZE_TYPE__, T);
};

I *GetI();

int main() {
  @autoreleasepool {
    // CHECK: [[I:%.+]] = alloca ptr
    auto* i = GetI();
    i.position = 42;

    // This is so we can find the next line more easily.
    // CHECK: store double
    double d = 42.0;

    // CHECK: [[I1:%.+]] = load ptr, ptr [[I]]
    // CHECK-NEXT: [[SEL1:%.+]] = load ptr, ptr [[SEL]]
    // CHECK-NEXT: [[POS1:%.+]] = call {{.*}} i32 @objc_msgSend(ptr {{.*}} [[I1]], ptr {{.*}} [[SEL1]])
    // CHECK-NEXT: call {{.*}} ptr @_ZN1SnwEmi(i64 {{.*}} 1, i32 {{.*}} [[POS1]])
    new (i.position) S;

    // CHECK: [[I2:%.+]] = load ptr, ptr [[I]]
    // CHECK-NEXT: [[SEL2:%.+]] = load ptr, ptr [[SEL]]
    // CHECK-NEXT: [[POS2:%.+]] = call {{.*}} i32 @objc_msgSend(ptr {{.*}} [[I2]], ptr {{.*}} [[SEL2]])
    // CHECK-NEXT: [[DBL:%.+]] = sitofp i32 [[POS2]] to double
    // CHECK-NEXT: call {{.*}} ptr  @_ZN2TSIdEnwEmd(i64 {{.*}} 1, double {{.*}} [[DBL]])
    new (i.position) TS<double>;
  }
}
