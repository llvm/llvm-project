// Check that -fstack-limit-variable=<var> attaches the valued
// "stack-limit-variable" function attribute (together with the default
// "stack-limit-trap-number"="255"), that -fstack-limit-trap-number=<n>
// overrides the trap value, and that both are absent by default.

// The attribute is emitted identically on every architecture the backend
// check currently supports: Thumb1 (thumbv6m / Cortex-M0) and Thumb2
// (thumbv7m / Cortex-M3, thumbv7em / Cortex-M4 and later).
// RUN: %clang_cc1 -triple thumbv6m-none-eabi -emit-llvm -o - %s -fstack-limit-variable=__stack_boundary | FileCheck %s -check-prefix=DEFAULT-TRAP
// RUN: %clang_cc1 -triple thumbv7m-none-eabi -emit-llvm -o - %s -fstack-limit-variable=__stack_boundary | FileCheck %s -check-prefix=DEFAULT-TRAP
// RUN: %clang_cc1 -triple thumbv7em-none-eabi -emit-llvm -o - %s -fstack-limit-variable=__stack_boundary | FileCheck %s -check-prefix=DEFAULT-TRAP

// RUN: %clang_cc1 -triple thumbv7em-none-eabi -emit-llvm -o - %s -fstack-limit-variable=my_limit | FileCheck %s -check-prefix=CUSTOM-VAR
// RUN: %clang_cc1 -triple thumbv7em-none-eabi -emit-llvm -o - %s -fstack-limit-variable=__stack_boundary -fstack-limit-trap-number=42 | FileCheck %s -check-prefix=CUSTOM-TRAP
// RUN: %clang_cc1 -triple thumbv7em-none-eabi -emit-llvm -o - %s | FileCheck %s -check-prefix=NO-ATTR

// DEFAULT-TRAP: define{{.*}} void @foo() #[[ATTR:[0-9]+]]
// DEFAULT-TRAP: attributes #[[ATTR]] = {{{.*}} "stack-limit-trap-number"="255" {{.*}}"stack-limit-variable"="__stack_boundary"

// CUSTOM-VAR: define{{.*}} void @foo() #[[ATTR:[0-9]+]]
// CUSTOM-VAR: attributes #[[ATTR]] = {{{.*}} "stack-limit-trap-number"="255" {{.*}}"stack-limit-variable"="my_limit"

// CUSTOM-TRAP: define{{.*}} void @foo() #[[ATTR:[0-9]+]]
// CUSTOM-TRAP: attributes #[[ATTR]] = {{{.*}} "stack-limit-trap-number"="42" {{.*}}"stack-limit-variable"="__stack_boundary"

// NO-ATTR-NOT: "stack-limit-variable"
// NO-ATTR-NOT: "stack-limit-trap-number"

void foo(void) {}
