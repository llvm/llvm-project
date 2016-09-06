// Ensure that declarations without definitions don't have maps emitted for them

// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only %s > %t
// FileCheck -input-file %t %s
// RUN: FileCheck -check-prefix BAR -input-file %t %s

// FOO: foo:
// FOO-NOT: foo:
inline int foo() { return 0; }
extern inline int foo();

// BAR: bar:
// BAR-NOT: bar:
int bar() { return 0; }
extern int bar();
