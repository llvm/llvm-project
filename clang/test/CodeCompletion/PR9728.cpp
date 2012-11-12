namespace N {
struct SFoo;
}

struct brokenfile_t {
  brokenfile_t (N::
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:6:20 %s -o - | FileCheck %s
  // CHECK: SFoo

