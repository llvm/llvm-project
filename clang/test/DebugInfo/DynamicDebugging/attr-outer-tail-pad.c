// RUN: %clang -cc1 %s -triple %itanium_abi_triple -debug-info-kind=constructor -fdynamic-debugging -o - \
// RUN:    -emit-llvm --discard-dynamic-debugging-debug-module \
// RUN: | FileCheck %s --check-prefix=X86

/// FIXME: Add negative tests for other targets (if in doubt, don't tail-pad).

/// Pad functions to minimum of 5 bytes for insertion of 32 rel jump.
// X86: define dso_local i32 @f() #0
// X86: attributes #0 = {{{.*}}"tail-pad-to-size"="5"{{.*}}"tail-pad-value"="144"{{.*}}}
int f() { return 0; }
