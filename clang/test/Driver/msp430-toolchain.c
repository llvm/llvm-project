// A basic clang -cc1 command-line, and simple environment check.

// RUN: %clang %s -### -no-canonical-prefixes -target msp430 2>&1 \
// RUN:   | FileCheck -check-prefix=CC1 %s
// CC1: clang{{.*}} "-cc1" "-triple" "msp430"

// RUN: %clang %s -### -no-canonical-prefixes -target msp430 \
// RUN:   --gcc-toolchain=%S/Inputs/basic_msp430_tree 2>&1 \
// RUN:   | FileCheck -check-prefix=MSP430 %s

// MSP430: "{{.*}}Inputs/basic_msp430_tree/lib/gcc/msp430-elf/7.3.1/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}msp430-elf-ld"
// MSP430: "-L{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/7.3.1/430"
// MSP430: "-L{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/7.3.1/../../..{{/|\\\\}}..{{/|\\\\}}msp430-elf{{/|\\\\}}lib/430"
// MSP430: "{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/7.3.1/../../..{{/|\\\\}}..{{/|\\\\}}msp430-elf{{/|\\\\}}lib/430{{/|\\\\}}crt0.o"
// MSP430: "{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/7.3.1/430{{/|\\\\}}crtbegin.o"
// MSP430: "--start-group" "-lmul_none" "-lgcc" "-lc" "-lcrt" "-lnosys" "--end-group"
// MSP430: "{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/7.3.1/430{{/|\\\\}}crtend.o"
// MSP430: "{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/7.3.1/../../..{{/|\\\\}}..{{/|\\\\}}msp430-elf{{/|\\\\}}lib/430{{/|\\\\}}crtn.o"

// RUN: %clang %s -### -no-canonical-prefixes -target msp430 -nodefaultlibs \
// RUN:   --gcc-toolchain=%S/Inputs/basic_msp430_tree 2>&1 \
// RUN:   | FileCheck -check-prefix=MSP430-NO-DFT-LIB %s

// MSP430-NO-DFT-LIB: "{{.*}}Inputs/basic_msp430_tree/lib/gcc/msp430-elf/7.3.1/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}msp430-elf-ld"
// MSP430-NO-DFT-LIB: "-L{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/7.3.1/430"
// MSP430-NO-DFT-LIB: "-L{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/7.3.1/../../..{{/|\\\\}}..{{/|\\\\}}msp430-elf{{/|\\\\}}lib/430"
// MSP430-NO-DFT-LIB: "{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/7.3.1/../../..{{/|\\\\}}..{{/|\\\\}}msp430-elf{{/|\\\\}}lib/430{{/|\\\\}}crt0.o"
// MSP430-NO-DFT-LIB: "{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/7.3.1/430{{/|\\\\}}crtbegin.o"
// MSP430-NO-DFT-LIB: "--start-group" "-lmul_none" "-lgcc" "--end-group"
// MSP430-NO-DFT-LIB: "{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/7.3.1/430{{/|\\\\}}crtend.o"
// MSP430-NO-DFT-LIB: "{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/7.3.1/../../..{{/|\\\\}}..{{/|\\\\}}msp430-elf{{/|\\\\}}lib/430{{/|\\\\}}crtn.o"

// RUN: %clang %s -### -no-canonical-prefixes -target msp430 -nostartfiles \
// RUN:   --gcc-toolchain=%S/Inputs/basic_msp430_tree 2>&1 \
// RUN:   | FileCheck -check-prefix=MSP430-NO-START %s

// MSP430-NO-START: "{{.*}}Inputs/basic_msp430_tree/lib/gcc/msp430-elf/7.3.1/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}msp430-elf-ld"
// MSP430-NO-START: "-L{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/7.3.1/430"
// MSP430-NO-START: "-L{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/7.3.1/../../..{{/|\\\\}}..{{/|\\\\}}msp430-elf{{/|\\\\}}lib/430"
// MSP430-NO-START: "--start-group" "-lmul_none" "-lgcc" "-lc" "-lcrt" "-lnosys" "--end-group"

// RUN: %clang %s -### -no-canonical-prefixes -target msp430 -nostdlib \
// RUN:   --gcc-toolchain=%S/Inputs/basic_msp430_tree 2>&1 \
// RUN:   | FileCheck -check-prefix=MSP430-NO-STD-LIB %s

// MSP430-NO-STD-LIB: "{{.*}}Inputs/basic_msp430_tree/lib/gcc/msp430-elf/7.3.1/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}msp430-elf-ld"
// MSP430-NO-STD-LIB: "-L{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/7.3.1/430"
// MSP430-NO-STD-LIB: "-L{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/7.3.1/../../..{{/|\\\\}}..{{/|\\\\}}msp430-elf{{/|\\\\}}lib/430"
// MSP430-NO-STD-LIB: "--start-group" "-lmul_none" "-lgcc" "--end-group"

// RUN: %clang %s -### -no-canonical-prefixes -target msp430 -mmcu=msp430f147 2>&1 \
// RUN:   | FileCheck -check-prefix=MSP430-HWMult-16BIT %s
// RUN: %clang %s -### -no-canonical-prefixes -target msp430 -mmcu=msp430f147 -mhwmult=auto 2>&1 \
// RUN:   | FileCheck -check-prefix=MSP430-HWMult-16BIT %s
// RUN: %clang %s -### -no-canonical-prefixes -target msp430 -mhwmult=16bit 2>&1 \
// RUN:   | FileCheck -check-prefix=MSP430-HWMult-16BIT %s

// MSP430-HWMult-16BIT: "--start-group" "-lmul_16"

// RUN: %clang %s -### -no-canonical-prefixes -target msp430 -mmcu=msp430f4783 2>&1 \
// RUN:   | FileCheck -check-prefix=MSP430-HWMult-32BIT %s
// RUN: %clang %s -### -no-canonical-prefixes -target msp430 -mmcu=msp430f4783 -mhwmult=auto 2>&1 \
// RUN:   | FileCheck -check-prefix=MSP430-HWMult-32BIT %s
// RUN: %clang %s -### -no-canonical-prefixes -target msp430 -mhwmult=32bit 2>&1 \
// RUN:   | FileCheck -check-prefix=MSP430-HWMult-32BIT %s

// MSP430-HWMult-32BIT: "--start-group" "-lmul_32"

// RUN: %clang %s -### -no-canonical-prefixes -target msp430 -mhwmult=f5series 2>&1 \
// RUN:   | FileCheck -check-prefix=MSP430-HWMult-F5 %s
// MSP430-HWMult-F5: "--start-group" "-lmul_f5"

// RUN: %clang %s -### -no-canonical-prefixes -target msp430 -mhwmult=none 2>&1 \
// RUN:   | FileCheck -check-prefix=MSP430-HWMult-NONE %s
// RUN: %clang %s -### -no-canonical-prefixes -target msp430 -mhwmult=none -mmcu=msp430f4783 2>&1 \
// RUN:   | FileCheck -check-prefix=MSP430-HWMult-NONE %s

// MSP430-HWMult-NONE: "--start-group" "-lmul_none"
