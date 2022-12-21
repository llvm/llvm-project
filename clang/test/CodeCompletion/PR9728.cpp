namespace N {
struct SFoo;
}

struct brokenfile_t {
  brokenfile_t (N::
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-1):20 %s -o - | FileCheck %s
  // CHECK: SFoo

