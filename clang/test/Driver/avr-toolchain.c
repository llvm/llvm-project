// UNSUPPORTED: system-windows
// A basic clang -cc1 command-line.

// RUN: %clang -### %s --target=avr --sysroot=%S/Inputs/basic_avr_tree -resource-dir=%S/Inputs/resource_dir 2>&1 | FileCheck --check-prefix=CHECK1 %s
// CHECK1: "-cc1" "-triple" "avr"
// CHECK1-SAME: "-resource-dir" "[[RESOURCE:[^"]+]]"
// CHECK1-SAME: "-isysroot" "[[SYSROOT:[^"]+/basic_avr_tree]]"
// CHECK1-SAME: "-internal-isystem"
// CHECK1-SAME: {{^}} "[[SYSROOT]]/usr/lib/gcc/avr/5.4.0/../../../avr/include"
// CHECK1-NOT:  "-L
// CHECK1:      avr-ld"
// CHECK1-SAME: "-o" "a.out"
// CHECK1-SAME: {{^}} "--gc-sections"

// RUN: %clang -### %s --target=avr --sysroot=%S/Inputs/basic_avr_tree_2/opt/local -S 2>&1 | FileCheck --check-prefix=CHECK2 %s
// CHECK2: "-cc1" "-triple" "avr"
// CHECK2-SAME: "-isysroot" "[[SYSROOT:[^"]+/basic_avr_tree_2/opt/local]]"
// CHECK2-SAME: "-internal-isystem"
// CHECK2-SAME: {{^}} "[[SYSROOT]]/lib/gcc/avr/10.3.0/../../../../avr/include"

// RUN: %clang -### %s --target=avr --sysroot=%S/Inputs/basic_avr_tree_2 -S 2>&1 | FileCheck --check-prefix=CHECK3 %s
// CHECK3: "-cc1" "-triple" "avr"
// CHECK3-SAME: "-isysroot" "[[SYSROOT:[^"]+/basic_avr_tree_2]]"
// CHECK3-SAME: "-internal-isystem"
// CHECK3-SAME: {{^}} "[[SYSROOT]]/usr/avr/include"

// RUN: %clang -### %s --target=avr 2>&1 | FileCheck -check-prefix=CC1 %s
// CC1: "-cc1" "-triple" "avr" {{.*}} "-fno-use-init-array" "-fno-use-cxa-atexit"

// RUN: %clang -### %s --target=avr -fuse-init-array -fuse-cxa-atexit 2>&1 | FileCheck -check-prefix=CHECK4 %s
// CHECK4: "-cc1" "-triple" "avr"
// CHECK4-NOT: "-fno-use-init-array"
// CHECK4-NOT: "-fno-use-cxa-atexit"

// RUN: %clang -### %s --target=avr --sysroot=%S/Inputs/basic_avr_tree 2>&1 -nostdinc | FileCheck --check-prefix=NOSTDINC %s
// RUN: %clang -### %s --target=avr --sysroot=%S/Inputs/basic_avr_tree 2>&1 -nostdlibinc | FileCheck --check-prefix=NOSTDINC %s
// NOSTDINC-NOT: "-internal-isystem" {{".*avr/include"}}

// RUN: %clang -### --target=avr --sysroot=%S/Inputs/basic_avr_tree -mmcu=atmega328 %s 2>&1 | FileCheck --check-prefix=NOWARN %s
// RUN: %clang -### --target=avr --sysroot=%S/Inputs/basic_avr_tree -mmcu=atmega328 -S %s 2>&1 | FileCheck --check-prefix=NOWARN %s
// RUN: %clang -### --target=avr --sysroot=%S/Inputs/ -mmcu=atmega328 -S %s 2>&1 | FileCheck --check-prefix=NOWARN %s
// NOWARN-NOT: warning:

// RUN: %clang -### --target=avr --sysroot=%S/Inputs/basic_avr_tree -S %s 2>&1 | FileCheck --check-prefixes=NOMCU,LINKA %s
// RUN: %clang -### --target=avr --sysroot=%S/Inputs/ -S %s 2>&1 | FileCheck --check-prefixes=NOMCU,LINKA %s
// RUN: %clang -### --target=avr --sysroot=%S/Inputs/basic_avr_tree %s 2>&1 | FileCheck --check-prefixes=NOMCU,LINKB %s
// NOMCU: warning: no target microcontroller specified on command line, cannot link standard libraries, please pass -mmcu=<mcu name>
// LINKB: warning: standard library not linked and so no interrupt vector table or compiler runtime routines will be linked
// LINKB: warning: support for passing the data section address to the linker for microcontroller '' is not implemented
// NOMCU-NOT: warning: {{.*}} avr-gcc
// NOMCU-NOT: warning: {{.*}} avr-libc
// LINKA-NOT: warning: {{.*}} interrupt vector
// LINKA-NOT: warning: {{.*}} data section address

// RUN: %clang -### --target=avr --sysroot=%S/Inputs/ -mmcu=atmega328 %s 2>&1 | FileCheck --check-prefixes=NOGCC %s
// NOGCC: warning: standard library not linked and so no interrupt vector table or compiler runtime routines will be linked
// NOGCC-NOT: warning: {{.*}} microcontroller
// NOGCC-NOT: warning: {{.*}} avr-libc
// NOGCC-NOT: warning: {{.*}} data section address

// RUN: %clang -### --target=avr --sysroot=%S/Inputs/basic_avr_tree -mmcu=atmega328 %s -fuse-ld=avrld 2>&1 | FileCheck --check-prefix=NOLD %s
// NOLD: error: invalid linker

// RUN: %clang -### --target=avr --sysroot=%S/Inputs/basic_avr_tree -mmcu=atmega328 %s -fuse-ld=%S/Inputs/basic_avr_tree/usr/bin/ld.lld 2>&1 | FileCheck --check-prefix=LLD %s
// LLD: {{".*lld"}}
// LLD-NOT: "avr-ld"
// LLD-NOT: "-mavr5"

// RUN: %clang -### --target=avr --sysroot=%S/Inputs/basic_avr_tree -mmcu=atmega328 %s -T avr.lds 2>&1 | FileCheck --check-prefix=LDS0 %s
// LDS0: "-T" "avr.lds" "-mavr5"

// RUN: %clang -### --target=avr --sysroot=%S/Inputs/basic_avr_tree -mmcu=atmega328 %s -fuse-ld=%S/Inputs/basic_avr_tree/usr/bin/ld.lld -T avr.lds 2>&1 | FileCheck --check-prefix=LDS1 %s
// LDS1: "-T" "avr.lds"
// LDS1-NOT: "-mavr5"

// RUN: %clang %s -### --target=avr -mmcu=atmega328 --sysroot=%S/Inputs/basic_avr_tree/ -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir --rtlib=libgcc 2>&1 | FileCheck --check-prefix=LIBGCC %s
// LIBGCC: "-lgcc"
// LIBGCC-NOT: libclang_rt

// RUN: %clang %s -### --target=avr -mmcu=atmega328 --sysroot=%S/Inputs/basic_avr_tree/ -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir --rtlib=compiler-rt 2>&1 | FileCheck --check-prefix=COMRT %s
// COMRT: avr/libclang_rt.builtins.a
// COMRT-NOT: "-lgcc"

// RUN: %clang %s -### --target=avr --sysroot=%S/Inputs/basic_avr_tree/ -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir --rtlib=compiler-rt 2>&1 | FileCheck --check-prefix=NOMCU %s
// RUN: %clang %s -### --target=avr --sysroot=%S/Inputs/basic_avr_tree/ -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir --rtlib=libgcc 2>&1 | FileCheck --check-prefix=NOMCU %s
// NOMCU-NOT: libclang_rt
// NOMCU-NOT: "-lgcc"
