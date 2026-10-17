// Tests that a function definition carries a single call graph section type
// identifier: its declaration and its body are annotated separately, and the
// second annotation must replace the first one.

// RUN: %clang_cc1 -triple x86_64-unknown-linux -fexperimental-call-graph-section \
// RUN: -emit-llvm -o - %s | FileCheck %s

// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fexperimental-call-graph-section \
// RUN: -emit-llvm -o - %s | FileCheck %s

void foo(void);

// CHECK: define {{(dso_local )?}}void @bar(){{[^!]*}}!callgraph ![[TVOID:[0-9]+]] {
void bar(void) {
  foo();
}

// The declaration of foo was created and annotated for the call in bar.
// CHECK: define {{(dso_local )?}}void @foo(){{[^!]*}}!callgraph ![[TVOID]] {
void foo(void) {
}
