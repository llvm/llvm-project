// RUN: %clang -c %s -o /dev/null -Xfoo < %s 2>&1 | FileCheck --check-prefix=CHECK_STANDALONE_FOO %s
// RUN: %clang -c %s -o /dev/null -Xfoo=bar 2>&1 | FileCheck --check-prefix=CHECK_JOINED_FOO %s

// This test ensures that we only warn on -X<unknown> and -X<unknown=value>
// in case it is used downstream. If we error, we can't ignore it and some
// use of these (ignored) flags are in legacy use.
// TODO: Deprecate with timebox warning so consumers can respond.

// CHECK_STANDALONE_FOO: warning: argument unused during compilation: '-Xfoo' [-Wunused-command-line-argument]
// CHECK_JOINED_FOO: warning: argument unused during compilation: '-Xfoo=bar' [-Wunused-command-line-argument]

// CHECK-NOT: clang{.*}: error: unknown argument:

void f(void) {}
