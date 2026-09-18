// -----------------------------------------------------------------------------
// Tests that target features are forwarded to the external assembler
// (llvm-mc) via -mattr= when -fno-integrated-as is used, so that the
// external assembler path is on par with the integrated assembler.
// -----------------------------------------------------------------------------

// Baseline: no HVX means no HVX features in -mattr=.
// RUN: %clang -### -c %s --target=hexagon-unknown-elf -fno-integrated-as \
// RUN:   -mcpu=hexagonv79 2>&1 | FileCheck -check-prefix=CHECK-NOHVX %s
// CHECK-NOHVX: llvm-mc
// CHECK-NOHVX-SAME: "-mcpu=hexagonv79"
// CHECK-NOHVX-NOT: "+hvx

// -mhvx enables HVX for the external assembler.
// RUN: %clang -### -c %s --target=hexagon-unknown-elf -fno-integrated-as \
// RUN:   -mcpu=hexagonv79 -mhvx 2>&1 | FileCheck -check-prefix=CHECK-HVX %s
// CHECK-HVX: llvm-mc
// CHECK-HVX-SAME: "-mcpu=hexagonv79"
// CHECK-HVX-SAME: "-mattr={{[^"]*}}+hvxv79

// -mhvx= selects an explicit HVX version.
// RUN: %clang -### -c %s --target=hexagon-unknown-elf -fno-integrated-as \
// RUN:   -mcpu=hexagonv79 -mhvx=v68 2>&1 | FileCheck -check-prefix=CHECK-HVXV68 %s
// CHECK-HVXV68: llvm-mc
// CHECK-HVXV68-SAME: "-mattr={{[^"]*}}+hvxv68

// -mhvx-length= is forwarded.
// RUN: %clang -### -c %s --target=hexagon-unknown-elf -fno-integrated-as \
// RUN:   -mcpu=hexagonv79 -mhvx -mhvx-length=128b 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-HVXLEN %s
// CHECK-HVXLEN: llvm-mc
// CHECK-HVXLEN-SAME: "-mattr={{[^"]*}}+hvx-length128b

// -mhvx-qfloat is forwarded.
// RUN: %clang -### -c %s --target=hexagon-unknown-elf -fno-integrated-as \
// RUN:   -mcpu=hexagonv79 -mhvx -mhvx-qfloat 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-HVXQFLOAT %s
// CHECK-HVXQFLOAT: llvm-mc
// CHECK-HVXQFLOAT-SAME: "-mattr={{[^"]*}}+hvx-qfloat

// -mhvx-ieee-fp is forwarded (previously the only HVX flag handled here).
// RUN: %clang -### -c %s --target=hexagon-unknown-elf -fno-integrated-as \
// RUN:   -mcpu=hexagonv73 -mhvx -mhvx-ieee-fp 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-HVXIEEE %s
// CHECK-HVXIEEE: llvm-mc
// CHECK-HVXIEEE-SAME: "-mattr={{[^"]*}}+hvx-ieee-fp
