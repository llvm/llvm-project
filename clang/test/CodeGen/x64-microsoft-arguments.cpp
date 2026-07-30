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

// Pass and return an object with an explicitly deleted copy assignment operator (pass directly, return indirectly).
// CHECK: define dso_local void @"?f5@@YA?AUS5@@XZ"(ptr dead_on_unwind noalias writable sret(%struct.S5) align 4 {{.*}})
// CHECK: call void @"?func5@@YA?AUS5@@U1@@Z"(ptr dead_on_unwind writable sret(%struct.S5) align 4 {{.*}}, i32 {{.*}})
struct S5 {
  S5& operator=(const S5&) = delete;
  int i;
};

S5 func5(S5 x);
S5 f5() {
  S5 x{ 1 };
  return func5(x);
}

// Pass and return an object with an explicitly defaulted copy assignment operator that is implicitly deleted (pass directly, return indirectly).
// CHECK: define dso_local void @"?f6@@YA?AUS6@@XZ"(ptr dead_on_unwind noalias writable sret(%struct.S6) align 8 {{.*}})
// CHECK: call void @"?func6@@YA?AUS6@@U1@@Z"(ptr dead_on_unwind writable sret(%struct.S6) align 8 {{.*}}, i64 {{.*}})
struct S6 {
  S6& operator=(const S6&) = default;
  int& i;
};

S6 func6(S6 x);
S6 f6() {
  S6 x{ global_i };
  return func6(x);
}
