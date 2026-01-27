// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefixes=LLVM,OGCG

extern "C" {
// CIR: cir.func {{.*}} @bar() -> !s32i attributes {noreturn} {
// LLVM: Function Attrs:{{.*}} noreturn
// LLVM-NEXT: define {{.*}} i32 @bar() #[[BAR_FOO_ATTR:.*]] {
__attribute((noreturn))
int bar() { }

// Note: Classic codegen puts this here, so we need this to make sure the
// FunctionAttrs from `trap` doesn't interfere with 'foo'. However, CIR->LLVM
// lowering puts the trap decl at the end, so it isn't here to worry about.
// OGCG: declare void @llvm.trap

// CIR: cir.func {{.*}} @foo() -> !s32i attributes {noreturn} {
// LLVM: Function Attrs:{{.*}} noreturn
// LLVM-NEXT: define {{.*}} i32 @foo() #[[BAR_FOO_ATTR]] {
[[noreturn]]
int foo() { }

void caller() {
  // CIR: cir.call @bar() {noreturn} : () -> !s32i
  // LLVM: call i32 @bar() #[[CALL_ATTR:.*]]
  bar();
}

void caller2() {
  // CIR: cir.call @foo() {noreturn} : () -> !s32i
  // LLVM: call i32 @foo() #[[CALL_ATTR]]
  foo();
}

// LLVM: attributes #[[BAR_FOO_ATTR]] = {{.*}}noreturn
// LLVM: attributes #[[CALL_ATTR]] = {{.*}}noreturn

}
