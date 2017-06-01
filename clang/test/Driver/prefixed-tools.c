// RUN: %clang -### -B%S/Inputs/prefixed_tools_tree -o %t.o -no-integrated-as -fuse-ld=ld \
// RUN:        -target x86_64--linux %s 2>&1 | \
// RUN: FileCheck --check-prefix=CHECK-M64 %s

// RUN: %clang -### -B%S/Inputs/prefixed_tools_tree -o %t.o -no-integrated-as -fuse-ld=ld \
// RUN:        -m32 -target x86_64--linux %s 2>&1 | \
// RUN: FileCheck --check-prefix=CHECK-M32 %s

// CHECK-M64: "{{.*}}{{/|\\\\}}prefixed_tools_tree{{/|\\\\}}x86_64--linux-as"
// CHECK-M64: "{{.*}}{{/|\\\\}}prefixed_tools_tree{{/|\\\\}}x86_64--linux-ld"
// CHECK-M32: "{{.*}}{{/|\\\\}}prefixed_tools_tree{{/|\\\\}}x86_64--linux-as"
// CHECK-M32: "{{.*}}{{/|\\\\}}prefixed_tools_tree{{/|\\\\}}x86_64--linux-ld"
