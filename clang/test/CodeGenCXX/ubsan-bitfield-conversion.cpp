// RUN: %clang -x c++ -fsanitize=implicit-bitfield-conversion -target x86_64-linux -S -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-BITFIELD-CONVERSION
// RUN: %clang -x c++ -fsanitize=implicit-integer-conversion -target x86_64-linux -S -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK
// RUN: %clang -x c++ -fsanitize=implicit-conversion -target x86_64-linux -S -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-BITFIELD-CONVERSION

struct S {
  int a:3;
  char b:2;
};

class C : public S {
  public:
    short c:3;
};

S s;
C c;

// CHECK-LABEL: define{{.*}} void @{{.*foo1.*}}
void foo1(int x) {
  s.a = x;
  // CHECK: store i8 %{{.*}}
  // CHECK-BITFIELD-CONVERSION: [[BFRESULTSHL:%.*]] = shl i8 {{.*}}, 5
  // CHECK-BITFIELD-CONVERSION-NEXT: [[BFRESULTASHR:%.*]] = ashr i8 [[BFRESULTSHL]], 5
  // CHECK-BITFIELD-CONVERSION-NEXT: [[BFRESULTCAST:%.*]] = sext i8 [[BFRESULTASHR]] to i32
  // CHECK-BITFIELD-CONVERSION: call void @__ubsan_handle_implicit_conversion
  // CHECK-BITFIELD-CONVERSION-NEXT: br label %[[CONT:.*]], !nosanitize
  c.a = x;
  // CHECK: store i8 %{{.*}}
  // CHECK-BITFIELD-CONVERSION: [[BFRESULTSHL:%.*]] = shl i8 {{.*}}, 5
  // CHECK-BITFIELD-CONVERSION-NEXT: [[BFRESULTASHR:%.*]] = ashr i8 [[BFRESULTSHL]], 5
  // CHECK-BITFIELD-CONVERSION-NEXT: [[BFRESULTCAST:%.*]] = sext i8 [[BFRESULTASHR]] to i32
  // CHECK-BITFIELD-CONVERSION: call void @__ubsan_handle_implicit_conversion
  // CHECK-BITFIELD-CONVERSION-NEXT: br label %[[CONT:.*]], !nosanitize
  // CHECK-BITFIELD-CONVERSION: [[CONT]]:
  // CHECK-NEXT: ret void
}

// CHECK-LABEL: define{{.*}} void @{{.*foo2.*}}
void foo2(int x) {
  s.b = x;
  // CHECK: store i8 %{{.*}}
  // CHECK-BITFIELD-CONVERSION: [[BFRESULTSHL:%.*]] = shl i8 {{.*}}, 6
  // CHECK-BITFIELD-CONVERSION-NEXT: [[BFRESULTASHR:%.*]] = ashr i8 [[BFRESULTSHL]], 6
  // CHECK-BITFIELD-CONVERSION: call void @__ubsan_handle_implicit_conversion
  // CHECK-BITFIELD-CONVERSION-NEXT: br label %[[CONT:.*]], !nosanitize
  c.b = x;
  // CHECK: store i8 %{{.*}}
  // CHECK-BITFIELD-CONVERSION: [[BFRESULTSHL:%.*]] = shl i8 {{.*}}, 6
  // CHECK-BITFIELD-CONVERSION-NEXT: [[BFRESULTASHR:%.*]] = ashr i8 [[BFRESULTSHL]], 6
  // CHECK-BITFIELD-CONVERSION: call void @__ubsan_handle_implicit_conversion
  // CHECK-BITFIELD-CONVERSION-NEXT: br label %[[CONT:.*]], !nosanitize
  // CHECK-BITFIELD-CONVERSION: [[CONT]]:
  // CHECK-NEXT: ret void
}

// CHECK-LABEL: define{{.*}} void @{{.*foo3.*}}
void foo3() {
  s.a++;
  // CHECK: store i8 %{{.*}}
  // CHECK-NEXT: [[BFRESULTSHL:%.*]] = shl i8 {{.*}}, 5
  // CHECK-NEXT: [[BFRESULTASHR:%.*]] = ashr i8 [[BFRESULTSHL]], 5
  // CHECK-NEXT: [[BFRESULTCAST:%.*]] = sext i8 [[BFRESULTASHR]] to i32
  // CHECK-BITFIELD-CONVERSION: call void @__ubsan_handle_implicit_conversion
  // CHECK-BITFIELD-CONVERSION-NEXT: br label %[[CONT:.*]], !nosanitize
  c.a++;
  // CHECK: store i8 %{{.*}}
  // CHECK-NEXT: [[BFRESULTSHL:%.*]] = shl i8 {{.*}}, 5
  // CHECK-NEXT: [[BFRESULTASHR:%.*]] = ashr i8 [[BFRESULTSHL]], 5
  // CHECK-NEXT: [[BFRESULTCAST:%.*]] = sext i8 [[BFRESULTASHR]] to i32
  // CHECK-BITFIELD-CONVERSION: call void @__ubsan_handle_implicit_conversion
  // CHECK-BITFIELD-CONVERSION-NEXT: br label %[[CONT:.*]], !nosanitize
  // CHECK-BITFIELD-CONVERSION: [[CONT]]:
  // CHECK-NEXT: ret void
}

// CHECK-LABEL: define{{.*}} void @{{.*foo4.*}}
void foo4(int x) {
  s.a += x;
  // CHECK: store i8 %{{.*}}
  // CHECK-NEXT: [[BFRESULTSHL:%.*]] = shl i8 {{.*}}, 5
  // CHECK-NEXT: [[BFRESULTASHR:%.*]] = ashr i8 [[BFRESULTSHL]], 5
  // CHECK-NEXT: [[BFRESULTCAST:%.*]] = sext i8 [[BFRESULTASHR]] to i32
  // CHECK-BITFIELD-CONVERSION: call void @__ubsan_handle_implicit_conversion
  // CHECK-BITFIELD-CONVERSION-NEXT: br label %[[CONT:.*]], !nosanitize
  c.a += x;
  // CHECK: store i8 %{{.*}}
  // CHECK-NEXT: [[BFRESULTSHL:%.*]] = shl i8 {{.*}}, 5
  // CHECK-NEXT: [[BFRESULTASHR:%.*]] = ashr i8 [[BFRESULTSHL]], 5
  // CHECK-NEXT: [[BFRESULTCAST:%.*]] = sext i8 [[BFRESULTASHR]] to i32
  // CHECK-BITFIELD-CONVERSION: call void @__ubsan_handle_implicit_conversion
  // CHECK-BITFIELD-CONVERSION-NEXT: br label %[[CONT:.*]], !nosanitize
  // CHECK-BITFIELD-CONVERSION: [[CONT]]:
  // CHECK-NEXT: ret void
}