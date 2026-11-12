// RUN: %clang -### -S \
// RUN:  --save-temps --undefine-macro=FOO --undefine-macro BAR \
// RUN: --param=FOO --output=FOO %s 2>&1 | \
// RUN: FileCheck %s

// CHECK-LABEL: "-cc1"
// CHECK: "-E"
// CHECK: "-U" "FOO"
// CHECK: "-U" "BAR"
// CHECK: "-o" "option-aliases.i"

// CHECK-LABEL: "-cc1"
// CHECK: "-S"
// CHECK: "-o" "FOO"
