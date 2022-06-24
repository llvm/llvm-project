// Check -B driver option.

/// Target triple prefix is not detected for -B.
// RUN: %clang %s -### -o %t.o -target i386-unknown-linux \
// RUN:     -B %S/Inputs/B_opt_tree/dir1 -fuse-ld=ld 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-B-OPT-TRIPLE %s
// CHECK-B-OPT-TRIPLE-NOT: "{{.*}}/Inputs/B_opt_tree/dir1{{/|\\\\}}i386-unknown-linux-ld"

// RUN: %clang %s -### -o %t.o -target i386-unknown-linux \
// RUN:     -B %S/Inputs/B_opt_tree/dir2 -fuse-ld=ld 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-B-OPT-DIR %s
// CHECK-B-OPT-DIR: "{{.*}}/Inputs/B_opt_tree/dir2{{/|\\\\}}ld"

// RUN: %clang %s -### -o %t.o -target i386-unknown-linux \
// RUN:     -B %S/Inputs/B_opt_tree/dir3/prefix- -fuse-ld=ld 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-B-OPT-PREFIX %s
// CHECK-B-OPT-PREFIX: "{{.*}}/Inputs/B_opt_tree/dir3{{/|\\\\}}prefix-ld"

// RUN: %clang %s -### -o %t.o -target i386-unknown-linux \
// RUN:     -B %S/Inputs/B_opt_tree/dir3/prefix- \
// RUN:     -B %S/Inputs/B_opt_tree/dir2 2>&1 -fuse-ld=ld \
// RUN:   | FileCheck --check-prefix=CHECK-B-OPT-MULT %s
// CHECK-B-OPT-MULT: "{{.*}}/Inputs/B_opt_tree/dir3{{/|\\\\}}prefix-ld"

// RUN: %clang -B %S/Inputs/does_not_exist -print-search-dirs \
// RUN:     -target aarch64-linux-gnu \
// RUN:   | FileCheck --check-prefix=CHECK-B-OPT-INVALID %s
// CHECK-B-OPT-INVALID-NOT: /..//bin
