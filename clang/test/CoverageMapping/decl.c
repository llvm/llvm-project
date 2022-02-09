// Ensure that declarations without definitions don't have maps emitted for them

// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only %s > %t
// FileCheck -input-file %t %s
// RUN: FileCheck -check-prefix BAR -input-file %t %s

// FOO: foo:
// FOO-NOT: foo:
inline int foo(void) { return 0; }
extern inline int foo(void);

// BAR: bar:
// BAR-NOT: bar:
int bar(void) { return 0; }
extern int bar(void);
