// UNSUPPORTED: system-windows
// A basic clang -cc1 command-line.

// RUN: %clang %s -### -target avr --sysroot %S/Inputs/basic_avr_tree -resource-dir=%S/Inputs/resource_dir 2>&1 | FileCheck --check-prefix=CHECK1 %s
// CHECK1: clang{{.*}} "-cc1" "-triple" "avr"
// CHECK1-SAME: "-resource-dir" "[[RESOURCE:[^"]+]]"
// CHECK1-SAME: "-isysroot" "[[SYSROOT:[^"]+/basic_avr_tree]]"
// CHECK1-SAME: "-internal-isystem"
// CHECK1-SAME: {{^}} "[[SYSROOT]]/usr/lib/gcc/avr/5.4.0/../../../avr/include"
// CHECK1-NOT:  "-L
// CHECK1:      avr-ld"
// CHECK1-SAME: "-o" "a.out"
// CHECK1-SAME: {{^}} "--gc-sections"

// RUN: %clang %s -### -target avr --sysroot %S/Inputs/basic_avr_tree_2/opt/local -S 2>&1 | FileCheck --check-prefix=CHECK2 %s
// CHECK2: clang{{.*}} "-cc1" "-triple" "avr"
// CHECK2-SAME: "-isysroot" "[[SYSROOT:[^"]+/basic_avr_tree_2/opt/local]]"
// CHECK2-SAME: "-internal-isystem"
// CHECK2-SAME: {{^}} "[[SYSROOT]]/lib/gcc/avr/10.3.0/../../../../avr/include"

// RUN: %clang %s -### -target avr --sysroot %S/Inputs/basic_avr_tree_2 -S 2>&1 | FileCheck --check-prefix=CHECK3 %s
// CHECK3: clang{{.*}} "-cc1" "-triple" "avr"
// CHECK3-SAME: "-isysroot" "[[SYSROOT:[^"]+/basic_avr_tree_2]]"
// CHECK3-SAME: "-internal-isystem"
// CHECK3-SAME: {{^}} "[[SYSROOT]]/usr/avr/include"

// RUN: %clang %s -### -target avr 2>&1 | FileCheck -check-prefix=CC1 %s
// CC1: clang{{.*}} "-cc1" "-triple" "avr" {{.*}} "-fno-use-init-array"

// RUN: %clang %s -### -target avr --sysroot %S/Inputs/basic_avr_tree 2>&1 -nostdinc | FileCheck --check-prefix=NOSTDINC %s
// RUN: %clang %s -### -target avr --sysroot %S/Inputs/basic_avr_tree 2>&1 -nostdlibinc | FileCheck --check-prefix=NOSTDINC %s
// NOSTDINC-NOT: "-internal-isystem" {{".*avr/include"}}
