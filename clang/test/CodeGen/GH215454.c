// RUN: not %clang_cc1 -std=gnu99 -emit-llvm-only %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -std=gnu99 -emit-llvm-only -fno-recovery-ast %s

// A declaration that declares nothing at the end of a statement expression
// leaves a RecoveryExpr condition behind without an error being emitted.
// CodeGen must not try to constant-fold it.

#define c(a, b)                                                                \
  {;__typeof__(b);}

void conditions(int e) {
  // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: cannot compile this scalar expression yet
  if (({ ; int; })) {}
  // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: cannot compile this scalar expression yet
  switch (({ ; int; })) {}
  // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: cannot compile this scalar expression yet
  while (({ ; int; })) {}
  // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: cannot compile this scalar expression yet
  do {} while (({ ; int; }));
  // Reproducer from the issue.
  // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: cannot compile this scalar expression yet
  if((c(, e););
}
