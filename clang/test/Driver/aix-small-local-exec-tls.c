// RUN: %clang -target powerpc64-unknown-aix -S -emit-llvm %s -o - | FileCheck %s
// RUN: %clang -target powerpc-unknown-aix -S -emit-llvm %s -o - | FileCheck %s
// RUN: %clang -target powerpc64le-unknown-linux-gnu -S -emit-llvm %s -o - | FileCheck %s
// RUN: %clang -target powerpc64-unknown-linux-gnu -S -emit-llvm %s -o - | FileCheck %s

// RUN: %clang -target powerpc64-unknown-aix -maix-small-local-exec-tls -S -emit-llvm \
// RUN:    %s -o - | FileCheck %s --check-prefix=CHECK-AIX_SMALL_LOCALEXEC_TLS

// RUN: not %clang -target powerpc-unknown-aix -maix-small-local-exec-tls \
// RUN:    -fsyntax-only %s 2>&1 | FileCheck --check-prefix=CHECK-UNSUPPORTED-AIX32 %s
// RUN: not %clang -target powerpc64le-unknown-linux-gnu -maix-small-local-exec-tls \
// RUN:    -fsyntax-only %s 2>&1 | FileCheck --check-prefix=CHECK-UNSUPPORTED-LINUX %s
// RUN: not %clang -target powerpc64-unknown-linux-gnu -maix-small-local-exec-tls \
// RUN:    -fsyntax-only %s 2>&1 | FileCheck --check-prefix=CHECK-UNSUPPORTED-LINUX %s

int test(void) {
  return 0;
}

// CHECK: test() #0 {
// CHECK: attributes #0 = {
// CHECK-SAME: -aix-small-local-exec-tls

// CHECK-UNSUPPORTED-AIX32: option '-maix-small-local-exec-tls' cannot be specified on this target
// CHECK-UNSUPPORTED-LINUX: option '-maix-small-local-exec-tls' cannot be specified on this target

// CHECK-AIX_SMALL_LOCALEXEC_TLS: test() #0 {
// CHECK-AIX_SMALL_LOCALEXEC_TLS: attributes #0 = {
// CHECK-AIX_SMALL_LOCALEXEC_TLS-SAME: +aix-small-local-exec-tls

