// REQUIRES: lld
// XFAIL: *

// RUN: not %clang %s -target x86_64-linux -c -fuse-ld=lld -gen-reproducer=error -fcrash-diagnostics-dir=%t -fcrash-diagnostics=all 2>&1 \
// RUN:   | FileCheck %s

// check that we still get lld's output
// CHECK: error: undefined symbol: a

// CHECK: Preprocessed source(s) and associated run script(s) are located at:
// CHECK-NEXT: note: diagnostic msg: {{.*}}/lld-repro-{{.*}}.c
// CHECK-NEXT: note: diagnostic msg: {{.*}}/linker-crash-{{.*}}.tar
// CHECK-NEXT: note: diagnostic msg: {{.*}}/lld-repro-{{.*}}.sh
// CHECK-NEXT: note: diagnostic msg:
// CHECK: ********************

// RUN: not %clang %s -target x86_64-linux -c -fuse-ld=lld -gen-reproducer=error -fcrash-diagnostics-dir=%t -fcrash-diagnostics=compiler 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NO-LINKER
// RUN: not %clang %s -target x86_64-linux -c -fuse-ld=lld -gen-reproducer=error -fcrash-diagnostics-dir=%t 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NO-LINKER

// NO-LINKER-NOT: Preprocessed source(s) and associated run script(s) are located at:

extern int a;
int main() {
  return a;
}
