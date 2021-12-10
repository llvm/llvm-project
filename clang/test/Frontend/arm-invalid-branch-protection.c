// REQUIRES: arm-registered-target
// RUN: %clang -target arm-arm-none-eabi -mbranch-protection=pac-ret+b-key -c %s -o /dev/null 2>&1 | FileCheck %s
// RUN: %clang -target arm-arm-none-eabi -mbranch-protection=pac-ret+b-key+leaf -c %s -o /dev/null 2>&1 | FileCheck %s
// RUN: %clang -target arm-arm-none-eabi -mbranch-protection=bti+pac-ret+b-key -c %s -o /dev/null 2>&1 | FileCheck %s
// RUN: %clang -target arm-arm-none-eabi -mbranch-protection=bti+pac-ret+b-key+leaf -c %s -o /dev/null 2>&1 | FileCheck %s

// CHECK: warning: invalid branch protection option 'b-key' in '-mbranch-protection={{[a-z+-]*}}' [-Wbranch-protection]
