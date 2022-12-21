#define PATHSIZE 256

static const int len = 1234;

void foo() {
  char arr[
// RUN: %clang_cc1 -fsyntax-only -code-completion-macros -code-completion-at=%s:%(line-1):12 %s -o - | FileCheck %s
// CHECK: COMPLETION: len
// CHECK: COMPLETION: PATHSIZE
