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

// RUN: %clang -### -S --language c --define-macro=A --define-macro B -mllvm=-a -mllvm -b --output BAR %s 2>&1 | FileCheck %s --check-prefix=CHECK2

// CHECK2:      "-cc1"
// CHECK2-SAME: "-D" "A" "-D" "B"
// CHECK2-SAME: "-mllvm" "-a" "-mllvm" "-b"
// CHECK2-SAME: "-o" "BAR" "-x" "c"
