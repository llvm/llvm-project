// Check we get a warning if -fdynamic-debugging is specified for an
// unsupported target. Dynamic debugging currently emits the inner module as
// an object regardless of output flags (e.g. -emit-llvm).

// RUN: %clang -cc1 %s -emit-llvm -debug-info-kind=limited -fdynamic-debugging -o - -triple aarch64-unknown-unknown 2>&1  | FileCheck %s --check-prefix=WITHOUT_TARGET
// WITHOUT_TARGET: warning: ignoring -fdynamic-debugging: unable to create target: 'No available targets are compatible with triple "aarch64-unknown-unknown"'
// WITHOUT_TARGET-NOT: .debug_llvm_dyndbg

// Prevent rotten green test by checking we do see .debug_llvm_dyndbg otherwise.
// RUN: %clang -cc1 %s -emit-obj -debug-info-kind=limited -fdynamic-debugging -o - -triple x86_64-unknown-unknown 2>&1 | FileCheck %s --check-prefix=WITH_TARGET
// WITH_TARGET-NOT: ignoring -fdynamic-debugging:
// WITH_TARGET: .debug_llvm_dyndbg

int g;
