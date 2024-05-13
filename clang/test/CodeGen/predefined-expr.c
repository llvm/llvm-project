// RUN: %clang_cc1 %s -triple %itanium_abi_triple -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -fms-extensions %s -triple %itanium_abi_triple -emit-llvm -o - | FileCheck %s

// CHECK: @__func__.plainFunction = private unnamed_addr constant [14 x i8] c"plainFunction\00"
// CHECK: @__PRETTY_FUNCTION__.plainFunction = private unnamed_addr constant [25 x i8] c"void plainFunction(void)\00"
// CHECK: @__func__.externFunction = private unnamed_addr constant [15 x i8] c"externFunction\00"
// CHECK: @__PRETTY_FUNCTION__.externFunction = private unnamed_addr constant [26 x i8] c"void externFunction(void)\00"
// CHECK: @__func__.privateExternFunction = private unnamed_addr constant [22 x i8] c"privateExternFunction\00"
// CHECK: @__PRETTY_FUNCTION__.privateExternFunction = private unnamed_addr constant [33 x i8] c"void privateExternFunction(void)\00"
// CHECK: @__func__.__captured_stmt = private unnamed_addr constant [25 x i8] c"functionWithCapturedStmt\00"
// CHECK: @__PRETTY_FUNCTION__.__captured_stmt = private unnamed_addr constant [36 x i8] c"void functionWithCapturedStmt(void)\00"
// CHECK: @__func__.staticFunction = private unnamed_addr constant [15 x i8] c"staticFunction\00"
// CHECK: @__PRETTY_FUNCTION__.staticFunction = private unnamed_addr constant [26 x i8] c"void staticFunction(void)\00"

int printf(const char *, ...);

void plainFunction(void) {
  printf("__func__ %s\n", __func__);
  printf("__FUNCTION__ %s\n", __FUNCTION__);
  printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
}

extern void externFunction(void) {
  printf("__func__ %s\n", __func__);
  printf("__FUNCTION__ %s\n", __FUNCTION__);
  printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
}

__private_extern__ void privateExternFunction(void) {
  printf("__func__ %s\n", __func__);
  printf("__FUNCTION__ %s\n", __FUNCTION__);
  printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
}

void functionWithCapturedStmt(void) {
  #pragma clang __debug captured
  {
    printf("__func__ %s\n", __func__);
    printf("__FUNCTION__ %s\n", __FUNCTION__);
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
  }
}

static void staticFunction(void) {
  printf("__func__ %s\n", __func__);
  printf("__FUNCTION__ %s\n", __FUNCTION__);
  printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
}

int main(void) {
  plainFunction();
  externFunction();
  privateExternFunction();
  functionWithCapturedStmt();
  staticFunction();

  return 0;
}
