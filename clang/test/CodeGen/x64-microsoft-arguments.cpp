// RUN: %clang_cc1 -triple x86_64-windows-msvc -ffreestanding -emit-llvm -O0 \
// RUN: -x c++ -o - %s | FileCheck %s

int global_i = 0;

// Pass and return object with a reference type (pass directly, return indirectly).
// CHECK: define dso_local void @"?f1@@YA?AUS1@@XZ"(ptr dead_on_unwind noalias writable sret(%struct.S1) align 8 {{.*}})
// CHECK: call void @"?func1@@YA?AUS1@@U1@@Z"(ptr dead_on_unwind writable sret(%struct.S1) align 8 {{.*}}, i64 {{.*}})
struct S1 {
  int& r;
};

S1 func1(S1 x);
S1 f1() {
  S1 x{ global_i };
  return func1(x);
}

// Pass and return object with a reference type within an inner struct (pass directly, return indirectly).
// CHECK: define dso_local void @"?f2@@YA?AUS2@@XZ"(ptr dead_on_unwind noalias writable sret(%struct.S2) align 8 {{.*}})
// CHECK: call void @"?func2@@YA?AUS2@@U1@@Z"(ptr dead_on_unwind writable sret(%struct.S2) align 8 {{.*}}, i64 {{.*}})
struct Inner {
  int& r;
};

struct S2 {
  Inner i;
};

S2 func2(S2 x);
S2 f2() {
  S2 x{ { global_i } };
  return func2(x);
}

// Pass and return object with a reference type (pass directly, return indirectly).
// CHECK: define dso_local void @"?f3@@YA?AUS3@@XZ"(ptr dead_on_unwind noalias writable sret(%struct.S3) align 8 {{.*}})
// CHECK: call void @"?func3@@YA?AUS3@@U1@@Z"(ptr dead_on_unwind writable sret(%struct.S3) align 8 {{.*}}, i64 {{.*}})
struct S3 {
  const int& r;
};

S3 func3(S3 x);
S3 f3() {
  S3 x{ global_i };
  return func3(x);
}

// Pass and return object with a reference type within an inner struct (pass directly, return indirectly).
// CHECK: define dso_local void @"?f4@@YA?AUS4@@XZ"(ptr dead_on_unwind noalias writable sret(%struct.S4) align 8 {{.*}})
// CHECK: call void @"?func4@@YA?AUS4@@U1@@Z"(ptr dead_on_unwind writable sret(%struct.S4) align 8 {{.*}}, i64 {{.*}})
struct InnerConst {
  const int& r;
};

struct S4 {
  InnerConst i;
};

S4 func4(S4 x);
S4 f4() {
  S4 x{ { global_i } };
  return func4(x);
}
