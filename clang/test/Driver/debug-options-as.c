// XFAIL: target={{.*}}-aix{{.*}}

// Check to make sure clang is somewhat picky about -g options.
// (Delived from debug-options.c)
// RUN: %clang -### -c -save-temps -integrated-as -g %s 2>&1 \
// RUN:   | FileCheck -check-prefix=SAVE %s
//
// SAVE: "-cc1"{{.*}}"-E"{{.*}}"-debug-info-kind=
// SAVE: "-cc1"{{.*}}"-emit-llvm-bc"{{.*}}"-debug-info-kind=
// SAVE: "-cc1"{{.*}}"-S"{{.*}}"-debug-info-kind=
// SAVE: "-cc1as"
// SAVE-NOT: -debug-info-kind=

// Make sure that '-ggdb0' is not accidentally mistaken for '-g'
// RUN: %clang -### -ggdb0 -c -integrated-as -x assembler %s 2>&1 \
// RUN:   | FileCheck -check-prefix=GGDB0 %s
//
// GGDB0: "-cc1as"
// GGDB0-NOT: -debug-info-kind=

// Check to make sure clang with -g on a .s file gets passed.
// This requires a target that defaults to DWARF.
// RUN: %clang -### --target=x86_64-linux-gnu -c -integrated-as -g -x assembler %s 2>&1 \
// RUN:   | FileCheck %s
//
// CHECK: "-cc1as"
// CHECK: "-debug-info-kind=constructor"

// Check that a plain -g, without any -gdwarf, for a MSVC target, doesn't
// trigger producing DWARF output.
// RUN: %clang -### --target=x86_64-windows-msvc -c -integrated-as -g -x assembler %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MSVC %s
//
// MSVC: "-cc1as"
// MSVC-NOT: "-debug-info-kind=constructor"

// Check that clang-cl with the -Z7 option works the same, not triggering
// any DWARF output.
//
// RUN: %clang_cl -### --target=x86_64-pc-windows-msvc -c -Z7 -x assembler -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MSVC %s

// Check to make sure clang with -g on a .s file gets passed -dwarf-debug-producer.
// RUN: %clang -### -c -integrated-as -g -x assembler %s 2>&1 \
// RUN:   | FileCheck -check-prefix=P %s
//
// P: "-cc1as"
// P: "-dwarf-debug-producer"

// Check that -gdwarf64 is passed to cc1as.
// RUN: %clang -### -c -gdwarf64 -gdwarf-5 -target x86_64 -integrated-as -x assembler %s 2>&1 \
// RUN:   | FileCheck -check-prefix=GDWARF64_ON %s
// RUN: %clang -### -c -gdwarf64 -gdwarf-4 -target x86_64 -integrated-as -x assembler %s 2>&1 \
// RUN:   | FileCheck -check-prefix=GDWARF64_ON %s
// RUN: %clang -### -c -gdwarf64 -gdwarf-3 -target x86_64 -integrated-as -x assembler %s 2>&1 \
// RUN:   | FileCheck -check-prefix=GDWARF64_ON %s
// GDWARF64_ON: "-cc1as"
// GDWARF64_ON: "-gdwarf64"

// Check that -gdwarf64 can be reverted with -gdwarf32.
// RUN: %clang -### -c -gdwarf64 -gdwarf32 -gdwarf-4 -target x86_64 -integrated-as -x assembler %s 2>&1 \
// RUN:   | FileCheck -check-prefix=GDWARF64_OFF %s
// GDWARF64_OFF: "-cc1as"
// GDWARF64_OFF-NOT: "-gdwarf64"

// Check that an error is reported if -gdwarf64 cannot be used.
// RUN: not %clang -### -c -gdwarf64 -gdwarf-2 --target=x86_64 -integrated-as -x assembler %s 2>&1 \
// RUN:   | FileCheck -check-prefix=GDWARF64_VER %s
// RUN: not %clang -### -c -gdwarf64 -gdwarf-4 --target=i386-linux-gnu %s 2>&1 \
// RUN:   | FileCheck -check-prefix=GDWARF64_32ARCH %s
// RUN: not %clang -### -c -gdwarf64 -gdwarf-4 -target x86_64-apple-darwin %s 2>&1 \
// RUN:   | FileCheck -check-prefix=GDWARF64_ELF %s
//
// GDWARF64_VER:  error: invalid argument '-gdwarf64' only allowed with 'DWARFv3 or greater'
// GDWARF64_32ARCH: error: invalid argument '-gdwarf64' only allowed with '64 bit architecture'
// GDWARF64_ELF: error: invalid argument '-gdwarf64' only allowed with 'ELF platforms'

// Check that -gdwarf-N can be placed before other options of the "-g" group.
// RUN: %clang -### -c -g -gdwarf-3 -target %itanium_abi_triple -fintegrated-as -x assembler %s 2>&1 \
// RUN:   | FileCheck -check-prefix=DWARF3 %s
// RUN: %clang -### -c -gdwarf-3 -g -target %itanium_abi_triple -fintegrated-as -x assembler %s 2>&1 \
// RUN:   | FileCheck -check-prefix=DWARF3 %s
// RUN: %clang -### -c -g -gdwarf-5 -target %itanium_abi_triple -fintegrated-as -x assembler %s 2>&1 \
// RUN:   | FileCheck -check-prefix=DWARF5 %s
// RUN: %clang -### -c -gdwarf-5 -g -target %itanium_abi_triple -fintegrated-as -x assembler %s 2>&1 \
// RUN:   | FileCheck -check-prefix=DWARF5 %s

// DWARF3: "-cc1as"
// DWARF3: "-dwarf-version=3"
// DWARF5: "-cc1as"
// DWARF5: "-dwarf-version=5"
