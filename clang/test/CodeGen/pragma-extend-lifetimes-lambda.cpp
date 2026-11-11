// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++11 -O1 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefix=NO-EXTENSION --implicit-check-not="call void (...) @llvm.fake.use("
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++11 -O1 -disable-llvm-passes -emit-llvm -fextend-variable-liveness=this -o - %s | FileCheck %s --check-prefixes=CHECK,THIS
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++11 -O1 -disable-llvm-passes -emit-llvm -fextend-variable-liveness=all -o - %s | FileCheck %s --check-prefixes=CHECK,ALL

// Keep the lambda call operators intact without adding the automatic optnone
// attribute used at -O0.
// NO-EXTENSION: define

// Lifetime extension is initially enabled by the command-line option.
void foo() {
  auto lambda = [](int value) {};
  lambda(42);
}

// CHECK-LABEL: define internal{{.*}}foo{{.*}}clEi
// THIS-NOT: @llvm.fake.use
// ALL: call void (...) @llvm.fake.use(i32 %
// CHECK: call void (...) @llvm.fake.use(ptr %
// CHECK-NOT: @llvm.fake.use
// CHECK: ret void

#pragma clang extend_lifetimes disable
// Disable lifetime extension.
void bar() {
  auto lambda = [](int value) {};
  lambda(42);
}

// CHECK-LABEL: define internal{{.*}}bar{{.*}}clEi
// CHECK-NOT: @llvm.fake.use
// CHECK: ret void

// The disabling pragma spans multiple lambda definitions.
void bat() {
  auto lambda = [](int value) {};
  lambda(42);
}

// CHECK-LABEL: define internal{{.*}}bat{{.*}}clEi
// CHECK-NOT: @llvm.fake.use
// CHECK: ret void

#pragma clang extend_lifetimes enable
// Enable lifetime extension.
void baz() {
  auto lambda = [](int value) {};
  lambda(42);
}

// CHECK-LABEL: define internal{{.*}}baz{{.*}}clEi
// THIS-NOT: @llvm.fake.use
// ALL: call void (...) @llvm.fake.use(i32 %
// CHECK: call void (...) @llvm.fake.use(ptr %
// CHECK-NOT: @llvm.fake.use
// CHECK: ret void

// Disabling inside a lambda leaves that lambda enabled.
void goo() {
  auto lambda = [](int value) {
#pragma clang extend_lifetimes disable
  };
  lambda(42);
}

// CHECK-LABEL: define internal{{.*}}goo{{.*}}clEi
// THIS-NOT: @llvm.fake.use
// ALL: call void (...) @llvm.fake.use(i32 %
// CHECK: call void (...) @llvm.fake.use(ptr %
// CHECK-NOT: @llvm.fake.use
// CHECK: ret void

// The previous disable applies here; enabling inside affects later lambdas.
void moo() {
  auto lambda = [](int value) {
#pragma clang extend_lifetimes enable
  };
  lambda(42);
}

// CHECK-LABEL: define internal{{.*}}moo{{.*}}clEi
// CHECK-NOT: @llvm.fake.use
// CHECK: ret void

// The enable in the previous lambda applies to this lambda.
void zoo() {
  auto lambda = [](int value) {};
  lambda(42);
}

// CHECK-LABEL: define internal{{.*}}zoo{{.*}}clEi
// THIS-NOT: @llvm.fake.use
// ALL: call void (...) @llvm.fake.use(i32 %
// CHECK: call void (...) @llvm.fake.use(ptr %
// CHECK-NOT: @llvm.fake.use
// CHECK: ret void

// The nodebug attribute suppresses extension on the call operator.
void fie() {
  auto lambda = [](int value) __attribute__((nodebug)) {};
  lambda(42);
}

// CHECK-LABEL: define internal{{.*}}fie{{.*}}clEi
// CHECK-NOT: @llvm.fake.use
// CHECK: ret void