// RUN: %clang -### -target x86_64--- -c -integrated-as %s 2>&1 | FileCheck %s
// CHECK: cc1as
// CHECK-NOT: -relax-all

// RUN: %clang -### -target x86_64--- -c -integrated-as -Wa,-L %s 2>&1 | FileCheck --check-prefix=OPT_L %s
// OPT_L: msave-temp-labels

// Test that -I params in -Wa, and -Xassembler args are passed to integrated assembler
// RUN: %clang -### -target x86_64--- -c -integrated-as %s -Wa,-I,foo_dir 2>&1 | FileCheck --check-prefix=WA_INCLUDE1 %s
// WA_INCLUDE1: cc1as
// WA_INCLUDE1: "-I" "foo_dir"

// RUN: %clang -### -target x86_64--- -c -integrated-as %s -Wa,-Ifoo_dir 2>&1 | FileCheck --check-prefix=WA_INCLUDE2 %s
// WA_INCLUDE2: cc1as
// WA_INCLUDE2: "-Ifoo_dir"

// RUN: %clang -### -target x86_64--- -c -integrated-as %s -Wa,-I -Wa,foo_dir 2>&1 | FileCheck --check-prefix=WA_INCLUDE3 %s
// WA_INCLUDE3: cc1as
// WA_INCLUDE3: "-I" "foo_dir"

// RUN: %clang -### -target x86_64--- -c -integrated-as %s -Xassembler -I -Xassembler foo_dir 2>&1 | FileCheck --check-prefix=XA_INCLUDE1 %s
// XA_INCLUDE1: cc1as
// XA_INCLUDE1: "-I" "foo_dir"

// RUN: %clang -### -target x86_64--- -c -integrated-as %s -Xassembler -Ifoo_dir 2>&1 | FileCheck --check-prefix=XA_INCLUDE2 %s
// XA_INCLUDE2: cc1as
// XA_INCLUDE2: "-Ifoo_dir"

// RUN: %clang -### -target x86_64--- -c -integrated-as %s -gdwarf-4 -gdwarf-2 2>&1 | FileCheck --check-prefix=DWARF2 %s
// DWARF2: "-debug-info-kind=limited" "-dwarf-version=2"

// RUN: %clang -### -target x86_64--- -c -integrated-as %s -gdwarf-3 2>&1 | FileCheck --check-prefix=DWARF3 %s
// DWARF3: "-debug-info-kind=limited" "-dwarf-version=3"

// RUN: %clang -### -target x86_64--- -c -integrated-as %s -gdwarf-4 2>&1 | FileCheck --check-prefix=DWARF4 %s
// DWARF4: "-debug-info-kind=limited" "-dwarf-version=4"

// RUN: %clang -### -target x86_64--- -c -integrated-as %s -Xassembler -gdwarf-2 2>&1 | FileCheck --check-prefix=DWARF2XASSEMBLER %s
// DWARF2XASSEMBLER: "-debug-info-kind=limited" "-dwarf-version=2"

// RUN: %clang -### -target x86_64--- -c -integrated-as %s -Wa,-gdwarf-2 2>&1 | FileCheck --check-prefix=DWARF2WA %s
// DWARF2WA: "-debug-info-kind=limited" "-dwarf-version=2"

// A dwarf version number that driver can't parse is just stuffed in.
// RUN: %clang -### -target x86_64--- -c -integrated-as %s -Wa,-gdwarf-huh 2>&1 | FileCheck --check-prefix=BOGODWARF %s
// BOGODWARF: "-gdwarf-huh"

// RUN: %clang -### -target x86_64--- -x assembler -c -integrated-as %s -I myincludedir 2>&1 | FileCheck --check-prefix=INCLUDEPATH %s
// INCLUDEPATH: "-I" "myincludedir"

// RUN: %clang -### -target x86_64--- -x assembler -c -fPIC -integrated-as %s 2>&1 | FileCheck --check-prefix=PIC %s
// PIC: "-mrelocation-model" "pic"

// RUN: %clang -### -target x86_64--- -c -integrated-as %s -Wa,-fdebug-compilation-dir,. 2>&1 | FileCheck --check-prefix=WA_DEBUGDIR %s
// WA_DEBUGDIR: "-fdebug-compilation-dir" "."

// RUN: %clang -### -target x86_64--- -c -integrated-as %s -Xassembler -fdebug-compilation-dir -Xassembler . 2>&1 | FileCheck --check-prefix=XA_DEBUGDIR %s
// XA_DEBUGDIR: "-fdebug-compilation-dir" "."
