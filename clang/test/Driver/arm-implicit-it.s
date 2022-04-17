// RUN: %clang -target armv7--none-eabi -### %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-DEFAULT

// RUN: %clang -target armv7--none-eabi -mimplicit-it=arm -### %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-ARM

// RUN: %clang -target armv7--none-eabi -mimplicit-it=thumb -### %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-THUMB

// RUN: %clang -target armv7--none-eabi -mimplicit-it=never -### %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-NEVER

// RUN: %clang -target armv7--none-eabi -mimplicit-it=always -### %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-ALWAYS

// RUN: %clang -target armv7--none-eabi -mimplicit-it=thisisnotavalidoption -### %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-INVALID

// CHECK-DEFAULT-NOT: "-arm-implicit-it
// CHECK-ARM: "-arm-implicit-it=arm"
// CHECK-THUMB: "-arm-implicit-it=thumb"
// CHECK-NEVER: "-arm-implicit-it=never"
// CHECK-ALWAYS: "-arm-implicit-it=always"
// CHECK-INVALID: error: unsupported argument 'thisisnotavalidoption' to option '-mimplicit-it='
