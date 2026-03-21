// Test -mlarge-eh-encoding flag forwarding to cc1 and LTO plugin-opt.

// RUN: %clang -### --target=x86_64-unknown-linux -mlarge-eh-encoding %s 2>&1 | FileCheck %s --check-prefixes=CC1,LTO-NEG
// RUN: %clang -### --target=x86_64-unknown-linux -mlarge-eh-encoding -mno-large-eh-encoding %s 2>&1 | FileCheck %s --check-prefix=CC1-NEG

// CC1:         "-mlarge-eh-encoding"
// CC1-NEG-NOT: "-mlarge-eh-encoding"

// RUN: %clang -### --target=x86_64-unknown-linux -flto -mlarge-eh-encoding %s 2>&1 | FileCheck %s --check-prefix=LTO
// RUN: %clang -### --target=x86_64-unknown-linux -flto=thin -mlarge-eh-encoding %s 2>&1 | FileCheck %s --check-prefix=LTO
// RUN: %clang -### --target=x86_64-unknown-linux -flto -mlarge-eh-encoding -mno-large-eh-encoding %s 2>&1 | FileCheck %s --check-prefix=LTO-NEG

// LTO:         "-plugin-opt=-large-eh-encoding"
// LTO-NEG-NOT: "-plugin-opt=-large-eh-encoding"
