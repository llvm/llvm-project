// A basic clang -cc1 command-line.

// RUN: %clang %s -### -no-canonical-prefixes -target avr 2>&1 | FileCheck -check-prefix=CC1 %s
// CC1: clang{{.*}} "-cc1" "-triple" "avr"

// RUN: %clang %s -### -no-canonical-prefixes -target avr --sysroot %S/Inputs/basic_avr_tree 2>&1 | FileCheck -check-prefix CC1A %s
// CC1A: clang{{.*}} "-cc1" "-triple" "avr" {{.*}} "-internal-isystem" {{".*avr/include"}}

// RUN: %clang %s -### -target avr --sysroot %S/Inputs/basic_avr_tree 2>&1 -nostdinc | FileCheck -check-prefix CC1B %s
// CC1B-NOT: "-internal-isystem" {{".*avr/include"}}

// RUN: %clang %s -### -target avr --sysroot %S/Inputs/basic_avr_tree 2>&1 -nostdlibinc | FileCheck -check-prefix CC1C %s
// CC1C-NOT: "-internal-isystem" {{".*avr/include"}}
