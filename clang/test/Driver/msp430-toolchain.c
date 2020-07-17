// Splitting some tests into POS and NEG parts so the latter can validate
// output fragments as large as possible for absence of some text.

// Test for include paths and other cc1 flags

// RUN: %clang %s -### -no-canonical-prefixes -target msp430 -E \
// RUN:   --sysroot="%S/Inputs/basic_msp430_tree" 2>&1 \
// RUN:   | FileCheck -check-prefix=INCLUDE-DIRS %s
// INCLUDE-DIRS: "{{.*}}clang{{.*}}" "-cc1" "-triple" "msp430"
// INCLUDE-DIRS: "-internal-isystem" "{{.*}}/Inputs/basic_msp430_tree{{/|\\\\}}msp430-elf{{/|\\\\}}include"

// Tests for passing flags to msp430-elf-ld (not file-related)

// RUN: %clang %s -### -no-canonical-prefixes -target msp430 --sysroot="" > %t 2>&1
// RUN: FileCheck -check-prefix=DEFAULT-POS %s < %t
// RUN: FileCheck -check-prefix=DEFAULT-NEG %s < %t
// DEFAULT-POS: "{{.*}}msp430-elf-ld"
// DEFAULT-POS: "--gc-sections"
// DEFAULT-NEG-NOT: "--relax"

// RUN: %clang %s -### -no-canonical-prefixes -target msp430 --sysroot="" \
// RUN:   -r 2>&1 | FileCheck --check-prefixes=NO-GC-SECTIONS,RELOCATABLE-OBJECT %s
// RUN: %clang %s -### -no-canonical-prefixes -target msp430 --sysroot="" \
// RUN:   -g 2>&1 | FileCheck -check-prefix=NO-GC-SECTIONS %s
// NO-GC-SECTIONS: "{{.*}}msp430-elf-ld"
// NO-GC-SECTIONS-NOT: "--gc-sections"
// RELOCATABLE-OBJECT-NOT: crt0.o
// RELOCATABLE-OBJECT-NOT: crtbegin
// RELOCATABLE-OBJECT-NOT: crtend
// RELOCATABLE-OBJECT-NOT: "-l{{.*}}"

// RUN: %clang %s -### -no-canonical-prefixes -target msp430 --sysroot="" \
// RUN:   -Wl,--some-linker-arg 2>&1 | FileCheck -check-prefix=WL-ARG %s
// WL-ARG: "{{.*}}msp430-elf-ld"
// WL-ARG: "--some-linker-arg"

// Trivially mapped options: arbitrarily split into two disjoint groups
// to check both "on"/present and "off"/absent state (when appropriate).

// RUN: %clang %s -### -no-canonical-prefixes -target msp430 --sysroot="" \
// RUN:   -o /tmp/test.elf -r -t -z muldefs -mrelax > %t 2>&1
// RUN: FileCheck -check-prefix=MISC-FLAGS-1-POS %s < %t
// RUN: FileCheck -check-prefix=MISC-FLAGS-1-NEG %s < %t
// MISC-FLAGS-1-POS: "{{.*}}msp430-elf-ld"
// MISC-FLAGS-1-POS-DAG: "--relax"
// MISC-FLAGS-1-POS-DAG: "-o" "/tmp/test.elf"
// MISC-FLAGS-1-POS-DAG: "-r"
// MISC-FLAGS-1-POS-DAG: "-t"
// MISC-FLAGS-1-POS-DAG: "-z" "muldefs"
// MISC-FLAGS-1-NEG: "{{.*}}msp430-elf-ld"
// MISC-FLAGS-1-NEG-NOT: "-e{{.*}}"
// MISC-FLAGS-1-NEG-NOT: "-s"
// MISC-FLAGS-1-NEG-NOT: "-u"

// RUN: %clang %s -### -no-canonical-prefixes -target msp430 --sysroot="" \
// RUN:   -e EntryPoint -s -u __undef > %t 2>&1
// RUN: FileCheck -check-prefix=MISC-FLAGS-2-POS %s < %t
// RUN: FileCheck -check-prefix=MISC-FLAGS-2-NEG %s < %t
// MISC-FLAGS-2-POS: "{{.*}}msp430-elf-ld"
// MISC-FLAGS-2-POS: "-e" "EntryPoint" "-s" "-u" "__undef"
// MISC-FLAGS-2-NEG: "{{.*}}msp430-elf-ld"
// MISC-FLAGS-2-NEG-NOT: "-r"
// MISC-FLAGS-2-NEG-NOT: "-t"
// MISC-FLAGS-2-NEG-NOT: "-z"
// MISC-FLAGS-2-NEG-NOT: "--relax"

// Tests for -nostdlib, -nostartfiles, -nodefaultfiles and -f(no-)exceptions

// RUN: %clang %s -### -no-canonical-prefixes -target msp430 -rtlib=libgcc \
// RUN:   --sysroot="%S/Inputs/basic_msp430_tree" > %t 2>&1
// RUN: FileCheck -check-prefix=LIBS-DEFAULT-POS %s < %t
// RUN: FileCheck -check-prefix=LIBS-DEFAULT-NEG %s < %t
// RUN: %clang %s -### -no-canonical-prefixes -target msp430 -rtlib=libgcc \
// RUN:   --gcc-toolchain="%S/Inputs/basic_msp430_tree" --sysroot="" 2>&1 \
// RUN:   | FileCheck -check-prefix=LIBS-DEFAULT-GCC-TOOLCHAIN %s
// LIBS-DEFAULT-POS: "{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/8.3.1/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}msp430-elf-ld"
// LIBS-DEFAULT-POS: "{{.*}}/Inputs/basic_msp430_tree{{/|\\\\}}msp430-elf{{/|\\\\}}lib/430{{/|\\\\}}crt0.o"
// LIBS-DEFAULT-POS: "{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/8.3.1/430{{/|\\\\}}crtbegin_no_eh.o"
// LIBS-DEFAULT-POS: "-L{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/8.3.1/430"
// LIBS-DEFAULT-POS: "-L{{.*}}/Inputs/basic_msp430_tree{{/|\\\\}}msp430-elf{{/|\\\\}}lib/430"
// LIBS-DEFAULT-POS: "-lgcc" "--start-group" "-lmul_none" "-lc" "-lgcc" "-lcrt" "-lnosys" "--end-group" "-lgcc"
// LIBS-DEFAULT-POS: "{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/8.3.1/430{{/|\\\\}}crtend_no_eh.o" "-lgcc"
// LIBS-DEFAULT-GCC-TOOLCHAIN: "{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/8.3.1/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}msp430-elf-ld"
// LIBS-DEFAULT-GCC-TOOLCHAIN: "{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/8.3.1/../../..{{/|\\\\}}..{{/|\\\\}}msp430-elf{{/|\\\\}}lib/430{{/|\\\\}}crt0.o"
// LIBS-DEFAULT-GCC-TOOLCHAIN: "{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/8.3.1/430{{/|\\\\}}crtbegin_no_eh.o"
// LIBS-DEFAULT-GCC-TOOLCHAIN: "-L{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/8.3.1/430"
// LIBS-DEFAULT-GCC-TOOLCHAIN: "-L{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/8.3.1/../../..{{/|\\\\}}..{{/|\\\\}}msp430-elf{{/|\\\\}}lib/430"
// LIBS-DEFAULT-GCC-TOOLCHAIN: "-lgcc" "--start-group" "-lmul_none" "-lc" "-lgcc" "-lcrt" "-lnosys" "--end-group" "-lgcc"
// LIBS-DEFAULT-GCC-TOOLCHAIN: "{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/8.3.1/430{{/|\\\\}}crtend_no_eh.o" "-lgcc"
// LIBS-DEFAULT-NEG-NOT: crtbegin.o
// LIBS-DEFAULT-NEG-NOT: -lssp_nonshared
// LIBS-DEFAULT-NEG-NOT: -lssp
// LIBS-DEFAULT-NEG-NOT: clang_rt
// LIBS-DEFAULT-NEG-NOT: crtend.o
// LIBS-DEFAULT-NEG-NOT: /exceptions

// RUN: %clang %s -### -no-canonical-prefixes -target msp430 -rtlib=compiler-rt \
// RUN:   --sysroot="%S/Inputs/basic_msp430_tree" > %t 2>&1
// RUN: FileCheck -check-prefix=LIBS-COMPILER-RT-POS %s < %t
// RUN: FileCheck -check-prefix=LIBS-COMPILER-RT-NEG %s < %t
// LIBS-COMPILER-RT-POS: "{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/8.3.1/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}msp430-elf-ld"
// LIBS-COMPILER-RT-POS: "{{.*}}/Inputs/basic_msp430_tree{{/|\\\\}}msp430-elf{{/|\\\\}}lib/430{{/|\\\\}}crt0.o"
// LIBS-COMPILER-RT-POS: "{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/8.3.1/430{{/|\\\\}}crtbegin_no_eh.o"
// LIBS-COMPILER-RT-POS: "-L{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/8.3.1/430"
// LIBS-COMPILER-RT-POS: "-L{{.*}}/Inputs/basic_msp430_tree{{/|\\\\}}msp430-elf{{/|\\\\}}lib/430"
// LIBS-COMPILER-RT-POS: "{{[^"]*}}libclang_rt.builtins-msp430.a" "--start-group" "-lmul_none" "-lc" "{{[^"]*}}libclang_rt.builtins-msp430.a" "-lcrt" "-lnosys" "--end-group" "{{[^"]*}}libclang_rt.builtins-msp430.a"
// LIBS-COMPILER-RT-POS: "{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/8.3.1/430{{/|\\\\}}crtend_no_eh.o" "{{[^"]*}}libclang_rt.builtins-msp430.a"
// LIBS-COMPILER-RT-NEG-NOT: crtbegin.o
// LIBS-COMPILER-RT-NEG-NOT: -lssp_nonshared
// LIBS-COMPILER-RT-NEG-NOT: -lssp
// LIBS-COMPILER-RT-NEG-NOT: -lgcc
// LIBS-COMPILER-RT-NEG-NOT: crtend.o
// LIBS-COMPILER-RT-NEG-NOT: /exceptions

// RUN: %clang %s -### -no-canonical-prefixes -target msp430 -rtlib=libgcc -fexceptions \
// RUN:   --sysroot="%S/Inputs/basic_msp430_tree" > %t 2>&1
// RUN: FileCheck -check-prefix=LIBS-EXC-POS %s < %t
// RUN: FileCheck -check-prefix=LIBS-EXC-NEG %s < %t
// LIBS-EXC-POS: "{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/8.3.1/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}msp430-elf-ld"
// LIBS-EXC-POS: "{{.*}}/Inputs/basic_msp430_tree{{/|\\\\}}msp430-elf{{/|\\\\}}lib/430/exceptions{{/|\\\\}}crt0.o"
// LIBS-EXC-POS: "{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/8.3.1/430/exceptions{{/|\\\\}}crtbegin.o"
// LIBS-EXC-POS: "-L{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/8.3.1/430/exceptions"
// LIBS-EXC-POS: "-L{{.*}}/Inputs/basic_msp430_tree{{/|\\\\}}msp430-elf{{/|\\\\}}lib/430/exceptions"
// LIBS-EXC-POS: "-lgcc" "--start-group" "-lmul_none" "-lc" "-lgcc" "-lcrt" "-lnosys" "--end-group"
// LIBS-EXC-POS: "{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/8.3.1/430/exceptions{{/|\\\\}}crtend.o" "-lgcc"
// LIBS-EXC-NEG-NOT: "{{.*}}/430"
// LIBS-EXC-NEG-NOT: "{{.*}}430/crt{{.*}}"

// RUN: %clang %s -### -no-canonical-prefixes -target msp430 -rtlib=libgcc \
// RUN:   -fstack-protector  --sysroot="%S/Inputs/basic_msp430_tree" 2>&1 \
// RUN:   | FileCheck -check-prefix=LIBS-SSP %s
// LIBS-SSP: "{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/8.3.1/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}msp430-elf-ld"
// LIBS-SSP: "{{.*}}/Inputs/basic_msp430_tree{{/|\\\\}}msp430-elf{{/|\\\\}}lib/430{{/|\\\\}}crt0.o"
// LIBS-SSP: "{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/8.3.1/430{{/|\\\\}}crtbegin_no_eh.o"
// LIBS-SSP: "-L{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/8.3.1/430"
// LIBS-SSP: "-L{{.*}}/Inputs/basic_msp430_tree{{/|\\\\}}msp430-elf{{/|\\\\}}lib/430"
// LIBS-SSP: "-lssp_nonshared" "-lssp"
// LIBS-SSP: "-lgcc" "--start-group" "-lmul_none" "-lc" "-lgcc" "-lcrt" "-lnosys" "--end-group"
// LIBS-SSP: "{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/8.3.1/430{{/|\\\\}}crtend_no_eh.o" "-lgcc"

// RUN: %clang %s -### -no-canonical-prefixes -target msp430 -rtlib=libgcc -nodefaultlibs \
// RUN:   --sysroot="%S/Inputs/basic_msp430_tree" > %t 2>&1
// RUN: FileCheck -check-prefix=LIBS-NO-DFT-POS %s < %t
// RUN: FileCheck -check-prefix=LIBS-NO-DFT-NEG %s < %t
// LIBS-NO-DFT-POS: "{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/8.3.1/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}msp430-elf-ld"
// LIBS-NO-DFT-POS: "{{.*}}/Inputs/basic_msp430_tree{{/|\\\\}}msp430-elf{{/|\\\\}}lib/430{{/|\\\\}}crt0.o"
// LIBS-NO-DFT-POS: "{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/8.3.1/430{{/|\\\\}}crtbegin_no_eh.o"
// LIBS-NO-DFT-POS: "-L{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/8.3.1/430"
// LIBS-NO-DFT-POS: "-L{{.*}}/Inputs/basic_msp430_tree{{/|\\\\}}msp430-elf{{/|\\\\}}lib/430"
// LIBS-NO-DFT-POS: "{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/8.3.1/430{{/|\\\\}}crtend_no_eh.o" "-lgcc"
// LIBS-NO-DFT-NEG-NOT: "-lc"
// LIBS-NO-DFT-NEG-NOT: "-lcrt"
// LIBS-NO-DFT-NEG-NOT: "-lsim"
// LIBS-NO-DFT-NEG-NOT: "-lnosys"
// LIBS-NO-DFT-NEG-NOT: "--start-group"
// LIBS-NO-DFT-NEG-NOT: "--end-group"

// RUN: %clang %s -### -no-canonical-prefixes -target msp430 -rtlib=libgcc -nolibc \
// RUN:   -fstack-protector --sysroot="%S/Inputs/basic_msp430_tree" > %t 2>&1
// RUN: FileCheck -check-prefix=LIBS-NO-LIBC-POS %s < %t
// RUN: FileCheck -check-prefix=LIBS-NO-LIBC-NEG %s < %t
// LIBS-NO-LIBC-POS: "{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/8.3.1/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}msp430-elf-ld"
// LIBS-NO-LIBC-POS: "{{.*}}/Inputs/basic_msp430_tree{{/|\\\\}}msp430-elf{{/|\\\\}}lib/430{{/|\\\\}}crt0.o"
// LIBS-NO-LIBC-POS: "{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/8.3.1/430{{/|\\\\}}crtbegin_no_eh.o"
// LIBS-NO-LIBC-POS: "-L{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/8.3.1/430"
// LIBS-NO-LIBC-POS: "-L{{.*}}/Inputs/basic_msp430_tree{{/|\\\\}}msp430-elf{{/|\\\\}}lib/430"
// LIBS-NO-LIBC-POS: "-lssp_nonshared" "-lssp" "-lgcc"
// LIBS-NO-LIBC-POS: "{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/8.3.1/430{{/|\\\\}}crtend_no_eh.o" "-lgcc"
// LIBS-NO-LIBC-NEG-NOT: "-lc"
// LIBS-NO-LIBC-NEG-NOT: "-lcrt"
// LIBS-NO-LIBC-NEG-NOT: "-lsim"
// LIBS-NO-LIBC-NEG-NOT: "-lnosys"
// LIBS-NO-LIBC-NEG-NOT: "--start-group"
// LIBS-NO-LIBC-NEG-NOT: "--end-group"

// RUN: %clang %s -### -no-canonical-prefixes -target msp430 -rtlib=libgcc -nostartfiles \
// RUN:   --sysroot="%S/Inputs/basic_msp430_tree" > %t 2>&1
// RUN: FileCheck -check-prefix=LIBS-NO-START-POS %s < %t
// RUN: FileCheck -check-prefix=LIBS-NO-START-NEG %s < %t
// LIBS-NO-START-POS: "{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/8.3.1/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}msp430-elf-ld"
// LIBS-NO-START-POS: "-L{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/8.3.1/430"
// LIBS-NO-START-POS: "-L{{.*}}/Inputs/basic_msp430_tree{{/|\\\\}}msp430-elf{{/|\\\\}}lib/430"
// LIBS-NO-START-POS: "-lgcc" "--start-group" "-lmul_none" "-lc" "-lgcc" "-lcrt" "-lnosys" "--end-group"
// LIBS-NO-START-NEG-NOT: crt0.o
// LIBS-NO-START-NEG-NOT: crtbegin
// LIBS-NO-START-NEG-NOT: crtend

// RUN: %clang %s -### -no-canonical-prefixes -target msp430 -nostdlib \
// RUN:   --sysroot="%S/Inputs/basic_msp430_tree" > %t 2>&1
// RUN: FileCheck -check-prefix=LIBS-NO-STD-POS %s < %t
// RUN: FileCheck -check-prefix=LIBS-NO-STD-NEG %s < %t
// LIBS-NO-STD-POS: "{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/8.3.1/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}msp430-elf-ld"
// LIBS-NO-STD-POS: "-L{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/8.3.1/430"
// LIBS-NO-STD-POS: "-L{{.*}}/Inputs/basic_msp430_tree{{/|\\\\}}msp430-elf{{/|\\\\}}lib/430"
// LIBS-NO-STD-NEG-NOT: crt0.o
// LIBS-NO-STD-NEG-NOT: crtbegin
// LIBS-NO-STD-NEG-NOT: crtend
// LIBS-NO-STD-NEG-NOT: "-lc"
// LIBS-NO-STD-NEG-NOT: "-lcrt"
// LIBS-NO-STD-NEG-NOT: "-lnosys"
// LIBS-NO-STD-NEG-NOT: "--start-group"
// LIBS-NO-STD-NEG-NOT: "--end-group"

// Test for linker script autodiscovery

// RUN: %clang %s -### -no-canonical-prefixes -target msp430 -mmcu=msp430g2553 \
// RUN:   --sysroot=%S/Inputs/basic_msp430_tree 2>&1 \
// RUN:   | FileCheck -check-prefix=LD-SCRIPT %s
// LD-SCRIPT: "{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/8.3.1/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}msp430-elf-ld"
// LD-SCRIPT: "-L{{.*}}/Inputs/basic_msp430_tree{{/|\\\\}}include"
// LD-SCRIPT: "-Tmsp430g2553.ld"

// RUN: %clang %s -### -no-canonical-prefixes -target msp430 -mmcu=msp430g2553 \
// RUN:   --sysroot=%S/Inputs/basic_msp430_tree \
// RUN:   -T custom_script.ld 2>&1 \
// RUN:   | FileCheck -check-prefix=CUSTOM-LD-SCRIPT %s
// CUSTOM-LD-SCRIPT: "{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/8.3.1/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}msp430-elf-ld"
// CUSTOM-LD_SCRIPT-NOT: "-Tmsp430g2553.ld"
// CUSTOM-LD-SCRIPT: "-T" "custom_script.ld"
// CUSTOM-LD_SCRIPT-NOT: "-Tmsp430g2553.ld"

// Test for compiling for simulator

// RUN: %clang %s -### -no-canonical-prefixes -target msp430 -mmcu=msp430g2553 \
// RUN:   -msim -rtlib=libgcc --sysroot=%S/Inputs/basic_msp430_tree > %t 2>&1
// RUN: FileCheck -check-prefix=SIMULATOR-POS %s < %t
// RUN: FileCheck -check-prefix=SIMULATOR-NEG %s < %t
// SIMULATOR-POS: "{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/8.3.1/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}msp430-elf-ld"
// SIMULATOR-POS: "{{.*}}/Inputs/basic_msp430_tree{{/|\\\\}}msp430-elf{{/|\\\\}}lib/430{{/|\\\\}}crt0.o"
// SIMULATOR-POS: "{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/8.3.1/430{{/|\\\\}}crtbegin_no_eh.o"
// SIMULATOR-POS: "-L{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/8.3.1/430"
// SIMULATOR-POS: "-L{{.*}}/Inputs/basic_msp430_tree{{/|\\\\}}msp430-elf{{/|\\\\}}lib/430"
// SIMULATOR-POS: "-lgcc" "--start-group" "-lmul_none" "-lc" "-lgcc" "-lcrt" "-lsim" "--undefined=__crt0_call_exit" "--end-group"
// SIMULATOR-POS: "-Tmsp430-sim.ld"
// SIMULATOR-POS: "{{.*}}/Inputs/basic_msp430_tree/lib/gcc/msp430-elf/8.3.1/430{{/|\\\\}}crtend_no_eh.o" "-lgcc"
// SIMULATOR-NEG-NOT: "-lnosys"

// Tests for HWMult

// RUN: %clang %s -### -no-canonical-prefixes -target msp430 -mmcu=msp430f147 --sysroot="" 2>&1 \
// RUN:   | FileCheck -check-prefix=HWMult-16BIT %s
// RUN: %clang %s -### -no-canonical-prefixes -target msp430 -mmcu=msp430f147 -mhwmult=auto --sysroot="" 2>&1 \
// RUN:   | FileCheck -check-prefix=HWMult-16BIT %s
// RUN: %clang %s -### -no-canonical-prefixes -target msp430 -mhwmult=16bit --sysroot="" 2>&1 \
// RUN:   | FileCheck -check-prefix=HWMult-16BIT %s
// HWMult-16BIT: "--start-group" "-lmul_16"

// RUN: %clang %s -### -no-canonical-prefixes -target msp430 -mmcu=msp430f4783 --sysroot="" 2>&1 \
// RUN:   | FileCheck -check-prefix=HWMult-32BIT %s
// RUN: %clang %s -### -no-canonical-prefixes -target msp430 -mmcu=msp430f4783 -mhwmult=auto --sysroot="" 2>&1 \
// RUN:   | FileCheck -check-prefix=HWMult-32BIT %s
// RUN: %clang %s -### -no-canonical-prefixes -target msp430 -mhwmult=32bit --sysroot="" 2>&1 \
// RUN:   | FileCheck -check-prefix=HWMult-32BIT %s
// HWMult-32BIT: "--start-group" "-lmul_32"

// RUN: %clang %s -### -no-canonical-prefixes -target msp430 -mhwmult=f5series --sysroot="" 2>&1 \
// RUN:   | FileCheck -check-prefix=HWMult-F5 %s
// HWMult-F5: "--start-group" "-lmul_f5"

// RUN: %clang %s -### -no-canonical-prefixes -target msp430 -mhwmult=none --sysroot="" 2>&1 \
// RUN:   | FileCheck -check-prefix=HWMult-NONE %s
// RUN: %clang %s -### -no-canonical-prefixes -target msp430 -mhwmult=none -mmcu=msp430f4783 --sysroot="" 2>&1 \
// RUN:   | FileCheck -check-prefix=HWMult-NONE %s
// HWMult-NONE: "--start-group" "-lmul_none"
