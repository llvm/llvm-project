// REQUIRES: lld
// UNSUPPORTED: ps4, ps5

// RUN: echo "-nostartfiles -nostdlib -fuse-ld=lld -gen-reproducer=error -fcrash-diagnostics-dir=%t" \
// RUN:   | sed -e 's/\\/\\\\/g' > %t.rsp

// RUN: not %clang %s @%t.rsp -fcrash-diagnostics=all 2>&1 \
// RUN:   | FileCheck %s

// Test that the reproducer can still be created even when the input source cannot be preprocessed
// again, like when reading from stdin.
// RUN: not %clang -x c - @%t.rsp -fcrash-diagnostics=all 2>&1 < %s \
// RUN:   | FileCheck %s

// check that we still get lld's output
// CHECK: error: undefined symbol: {{_?}}a

// CHECK: Preprocessed source(s) and associated run script(s) are located at:
// CHECK-NEXT: note: diagnostic msg: {{.*}}linker-crash-{{.*}}.tar
// CHECK-NEXT: note: diagnostic msg:
// CHECK: ********************

// RUN: not %clang %s @%t.rsp -fcrash-diagnostics=compiler 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NO-LINKER
// RUN: not %clang %s @%t.rsp 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NO-LINKER

// NO-LINKER-NOT: Preprocessed source(s) and associated run script(s) are located at:

extern int a;
int main() {
  return a;
}
