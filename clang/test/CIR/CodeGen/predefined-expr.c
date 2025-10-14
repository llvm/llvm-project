// RUN: %clang_cc1 %s -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -o %t.cir
// RUN: FileCheck %s --input-file=%t.cir --check-prefix=CIR
// RUN: %clang_cc1 %s -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -o %t-cir.ll
// RUN: FileCheck %s --input-file=%t-cir.ll --check-prefix=LLVM
// RUN: %clang_cc1 %s -triple x86_64-unknown-linux-gnu -emit-llvm -o %t.ll
// RUN: FileCheck %s --input-file=%t.ll --check-prefix=OGCG

// CIR: cir.global "private" constant cir_private dso_local @__func__.plainFunction = #cir.const_array<"plainFunction\00" : !cir.array<!s8i x 14>>
// CIR: cir.global "private" constant cir_private dso_local @__PRETTY_FUNCTION__.plainFunction = #cir.const_array<"void plainFunction(void)\00" : !cir.array<!s8i x 25>>
// CIR: cir.global "private" constant cir_private dso_local @__func__.externFunction = #cir.const_array<"externFunction\00" : !cir.array<!s8i x 15>>
// CIR: cir.global "private" constant cir_private dso_local @__PRETTY_FUNCTION__.externFunction = #cir.const_array<"void externFunction(void)\00" : !cir.array<!s8i x 26>>
// CIR: cir.global "private" constant cir_private dso_local @__func__.privateExternFunction = #cir.const_array<"privateExternFunction\00" : !cir.array<!s8i x 22>>
// CIR: cir.global "private" constant cir_private dso_local @__PRETTY_FUNCTION__.privateExternFunction = #cir.const_array<"void privateExternFunction(void)\00" : !cir.array<!s8i x 33>>
// CIR: cir.global "private" constant cir_private dso_local @__func__.staticFunction = #cir.const_array<"staticFunction\00" : !cir.array<!s8i x 15>>
// CIR: cir.global "private" constant cir_private dso_local @__PRETTY_FUNCTION__.staticFunction = #cir.const_array<"void staticFunction(void)\00" : !cir.array<!s8i x 26>>

// TODO(cir): These should be unnamed_addr
// LLVM: @__func__.plainFunction = private constant [14 x i8] c"plainFunction\00"
// LLVM: @__PRETTY_FUNCTION__.plainFunction = private constant [25 x i8] c"void plainFunction(void)\00"
// LLVM: @__func__.externFunction = private constant [15 x i8] c"externFunction\00"
// LLVM: @__PRETTY_FUNCTION__.externFunction = private constant [26 x i8] c"void externFunction(void)\00"
// LLVM: @__func__.privateExternFunction = private constant [22 x i8] c"privateExternFunction\00"
// LLVM: @__PRETTY_FUNCTION__.privateExternFunction = private constant [33 x i8] c"void privateExternFunction(void)\00"
// LLVM: @__func__.staticFunction = private constant [15 x i8] c"staticFunction\00"
// LLVM: @__PRETTY_FUNCTION__.staticFunction = private constant [26 x i8] c"void staticFunction(void)\00"

// OGCG: @__func__.plainFunction = private unnamed_addr constant [14 x i8] c"plainFunction\00"
// OGCG: @__PRETTY_FUNCTION__.plainFunction = private unnamed_addr constant [25 x i8] c"void plainFunction(void)\00"
// OGCG: @__func__.externFunction = private unnamed_addr constant [15 x i8] c"externFunction\00"
// OGCG: @__PRETTY_FUNCTION__.externFunction = private unnamed_addr constant [26 x i8] c"void externFunction(void)\00"
// OGCG: @__func__.privateExternFunction = private unnamed_addr constant [22 x i8] c"privateExternFunction\00"
// OGCG: @__PRETTY_FUNCTION__.privateExternFunction = private unnamed_addr constant [33 x i8] c"void privateExternFunction(void)\00"
// OGCG: @__func__.staticFunction = private unnamed_addr constant [15 x i8] c"staticFunction\00"
// OGCG: @__PRETTY_FUNCTION__.staticFunction = private unnamed_addr constant [26 x i8] c"void staticFunction(void)\00"

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

// TODO(cir): Add support for __captured_stmt

static void staticFunction(void) {
  printf("__func__ %s\n", __func__);
  printf("__FUNCTION__ %s\n", __FUNCTION__);
  printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
}

int main(void) {
  plainFunction();
  externFunction();
  privateExternFunction();
  staticFunction();

  return 0;
}
