void f() {
  auto foo = bar;
  switch(foo) {
    case x:
      break;
  }
}

// RUN: not %clang_cc1 -fsyntax-only -fno-recovery-ast -code-completion-at=%s:4:10 %s | FileCheck %s -allow-empty
// CHECK-NOT: COMPLETION: foo
