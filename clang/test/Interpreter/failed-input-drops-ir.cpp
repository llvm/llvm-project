// REQUIRES: host-supports-jit
// RUN: cat %s | clang-repl 2>&1 | FileCheck %s
// RUN: cat %s | clang-repl 2>&1 | FileCheck %s --check-prefix=NEG

// A failed input must not leave its IR behind. A definition before the error
// must not block a later definition, and its initializers must not run.

extern "C" int printf(const char *, ...);

int f() { return 1; } int g() { return no_such_name; }
// CHECK-DAG: error: use of undeclared identifier 'no_such_name'
int f() { return 1; }
// NEG-NOT: error: definition with same mangled name
printf("f() = %d\n", f());
// CHECK-DAG: f() = 1

int v = printf("v init\n"); int w = no_such_name;
printf("stmt\n"); no_such_name;
int ok = 1;
// NEG-NOT: {{^}}v init
// NEG-NOT: {{^}}stmt
printf("ok = %d\n", ok);
// CHECK-DAG: ok = 1

// CodeGen reports this error at the end of the unit. The input fails, and the
// next input works.
extern "C" int no_target(); extern "C" int ali() __attribute__((alias("no_target")));
// CHECK-DAG: error: alias must point to a defined variable or function
printf("after = %d\n", ok + 1);
// CHECK-DAG: after = 2

%quit
