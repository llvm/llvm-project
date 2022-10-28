// RUN: %clang -### -Werror --target=x86_64 -fdebug-default-version=4 -gdwarf-2 -S -o - %s 2>&1 | FileCheck %s --check-prefix=DWARF2 --implicit-check-not=warning:
// RUN: %clang -### -Werror --target=x86_64 -gdwarf-3 -fdebug-default-version=4 -S -o - %s 2>&1 | FileCheck %s --check-prefix=DWARF3
// RUN: %clang -### -Werror --target=x86_64 -gdwarf-4 -fdebug-default-version=2 -S -o - %s 2>&1 | FileCheck %s --check-prefix=DWARF4
// RUN: %clang -### -Werror --target=x86_64 -gdwarf-5 -S -fdebug-default-version=2 -o - %s 2>&1 | FileCheck %s --check-prefix=DWARF5
// RUN: %clang -### -Werror --target=x86_64 -fdebug-default-version=5 -g -S -o - %s 2>&1 | FileCheck %s --check-prefix=DWARF5 --implicit-check-not=warning:
// RUN: %clang -### -Werror --target=x86_64 -gdwarf -fdebug-default-version=2 -S -o - %s 2>&1 | FileCheck %s --check-prefix=DWARF2
// RUN: %clang -### -Werror --target=x86_64 -fdebug-default-version=4 -S -o - %s 2>&1 | FileCheck %s --check-prefixes=NODEBUGINFO,NODWARF4

// Check which debug info formats we use on Windows. By default, in an MSVC
// environment, we should use codeview. You can enable dwarf, which implicitly
// disables codeview, or you can explicitly ask for both if you don't know how
// the app will be debugged.
// RUN: %clang -### -Werror -target i686-pc-windows-msvc -fdebug-default-version=2 -gdwarf -S -o - %s 2>&1 \
// RUN:     | FileCheck %s --check-prefixes=DWARF2,NOCODEVIEW
//     Explicitly request both.
// RUN: %clang -### -Werror -target i686-pc-windows-msvc -fdebug-default-version=4 -gdwarf -gcodeview -S -o - %s 2>&1 \
// RUN:     | FileCheck %s --check-prefixes=DWARF4,CODEVIEW

// Do Assembler testing most of the same test cases as those above.

// RUN: %clang -### -Werror -target x86_64-linux-gnu -fdebug-default-version=5 -g -x assembler -c -o - %s 2>&1 | FileCheck %s --check-prefix=DWARF5

// RUN: %clang -### -Werror -target x86_64-linux-gnu -fdebug-default-version=4 -gdwarf-2 -x assembler -c -o - %s 2>&1 | FileCheck %s --check-prefix=DWARF2
// RUN: %clang -### -Werror -target x86_64-linux-gnu -gdwarf-3 -fdebug-default-version=4 -x assembler -c -o - %s 2>&1 | FileCheck %s --check-prefix=DWARF3
// RUN: %clang -### -Werror -target x86_64-linux-gnu -gdwarf-4 -fdebug-default-version=2 -x assembler -c -o - %s 2>&1 | FileCheck %s --check-prefix=DWARF4
// RUN: %clang -### -Werror -target x86_64-linux-gnu -gdwarf-5 -x assembler -c -fdebug-default-version=2 -o - %s 2>&1 | FileCheck %s --check-prefix=DWARF5
// RUN: %clang -### -Werror -target x86_64-linux-gnu -fdebug-default-version=5 -g -x assembler -c -o - %s 2>&1 | FileCheck %s --check-prefix=DWARF5

int main(void) {
  return 0;
}

// NOCODEVIEW-NOT: -gcodeview
// CODEVIEW: "-gcodeview"

// NODEBUGINFO-NOT: -debug-info-kind=

// DWARF2: "-dwarf-version=2"
// DWARF3: "-dwarf-version=3"
// DWARF4: "-dwarf-version=4"
// DWARF5: "-dwarf-version=5"

// NOCODEVIEW-NOT: -gcodeview
// NODWARF4-NOT: -dwarf-version=4
