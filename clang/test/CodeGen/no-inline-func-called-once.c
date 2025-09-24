// REQUIRES: x86-registered-target
// RUN: %clang -O1 -S -emit-llvm %s -fno-inline-functions-called-once -o - | FileCheck %s --check-prefix=NOINLINE

// We verify three things:
//   1) There is a surviving call to bad_function (so it wasn’t inlined).
//   2) bad_function’s definition exists and carries an attribute group id.
//   3) That attribute group includes 'noinline'.

// The call is earlier in the IR than the callee/attributes, so use -DAG for the
// first two checks to avoid order constraints, then pin the attributes match.

// NOINLINE-DAG: call{{.*}} @bad_function{{.*}}
// NOINLINE-DAG: define internal{{.*}} @bad_function{{.*}} #[[ATTR:[0-9]+]]
// NOINLINE: attributes #[[ATTR]] = { {{.*}}noinline{{.*}} }

volatile int G;

static void bad_function(void) {
  // Volatile side effect ensures the call can’t be DCE’d.
  G++;
}

static void test(void) {
  // Exactly one TU-local caller of bad_function.
  bad_function();
}

int main(void) {
  // Make the caller reachable so it survives global DCE.
  test();
  return 0;
}
