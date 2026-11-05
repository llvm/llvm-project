// clang-cl syntax is normalized independently of the target toolchain.
// RUN: %clang_cl --target=x86_64-pc-windows-msvc /c -### /DFOO#7 /DBAR=a#b /DBAZ#a=b /DEMPTY# \
// RUN:   /DQUX#a#b -- %s 2>&1 | FileCheck %s --check-prefix=MACROS
// RUN: %clang_cl --target=x86_64-unknown-linux-gnu /c -### /DFOO#7 /DBAR=a#b /DBAZ#a=b /DEMPTY# \
// RUN:   /DQUX#a#b -- %s 2>&1 | FileCheck %s --check-prefix=MACROS
// RUN: %clang_cl --target=x86_64-w64-windows-gnu /c -### /DFOO#7 /DBAR=a#b /DBAZ#a=b /DEMPTY# \
// RUN:   /DQUX#a#b -- %s 2>&1 | FileCheck %s --check-prefix=MACROS
// MACROS: "-D" "FOO=7" "-D" "BAR=a#b" "-D" "BAZ=a=b" "-D" "EMPTY=" "-D" "QUX=a#b"

// Preserve GNU-mode Clang's existing MSVC macro compatibility.
// RUN: %clang --target=x86_64-pc-windows-msvc -c -### -DFOO#7 -- %s 2>&1 | FileCheck %s --check-prefix=MSVC
// RUN: %clang --target=x86_64-unknown-linux-gnu -c -### -DFOO#7 -- %s 2>&1 | FileCheck %s --check-prefix=GNU
// MSVC: "-D" "FOO=7"
// GNU: "-D" "FOO#7"

// Filtering host arguments preserves explicit overrides of shared /O options.
// RUN: %clang_cl --target=x86_64-pc-windows-msvc /c -### /O2 /Ob0 \
// RUN:   -clang:-Xarch_host -clang:-O1 -clang:-Xarch_host -clang:-fno-builtin \
// RUN:   -- %s 2>&1 | FileCheck %s --check-prefix=FILTERED
// RUN: %clang_cl --target=x86_64-unknown-linux-gnu /c -### /O2 /Ob0 \
// RUN:   -clang:-Xarch_host -clang:-O1 -clang:-Xarch_host -clang:-fno-builtin \
// RUN:   -- %s 2>&1 | FileCheck %s --check-prefix=FILTERED
// RUN: %clang_cl --target=x86_64-w64-windows-gnu /c -### /O2 /Ob0 \
// RUN:   -clang:-Xarch_host -clang:-O1 -clang:-Xarch_host -clang:-fno-builtin \
// RUN:   -- %s 2>&1 | FileCheck %s --check-prefix=FILTERED
// FILTERED: "-cc1"
// FILTERED-NOT: "-O3"
// FILTERED: "-O1"
// FILTERED-NOT: "-O3"
// FILTERED: "-fno-builtin"
// FILTERED-SAME: "-fno-inline"
// FILTERED-NOT: argument unused during compilation

// Last-option precedence must include both slash and canonical options.
// RUN: %clang_cl --target=x86_64-pc-windows-msvc /c -x c++ -### /permissive- /permissive -- %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=PERMISSIVE
// RUN: %clang_cl --target=x86_64-unknown-linux-gnu /c -x c++ -### /permissive- /permissive -- %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=PERMISSIVE
// RUN: %clang_cl --target=x86_64-w64-windows-gnu /c -x c++ -### /permissive -- %s 2>&1 | FileCheck %s \
// RUN:   --check-prefix=PERMISSIVE
// PERMISSIVE: "-fno-operator-names"
// PERMISSIVE-SAME: "-fdelayed-template-parsing"

// RUN: %clang_cl --target=x86_64-pc-windows-msvc /c -x c++ -### /permissive /permissive- -- %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CONFORMING
// RUN: %clang_cl --target=x86_64-unknown-linux-gnu /c -x c++ -### /permissive /permissive- -- %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CONFORMING
// RUN: %clang_cl --target=x86_64-pc-windows-msvc /c -x c++ -### /permissive /Zc:twoPhase \
// RUN:   -clang:-foperator-names -- %s 2>&1 | FileCheck %s --check-prefix=CONFORMING
// /clang: arguments are appended after the other command-line arguments.
// RUN: %clang_cl --target=x86_64-unknown-linux-gnu /c -x c++ -### \
// RUN:   -clang:-fno-delayed-template-parsing -clang:-foperator-names /permissive -- %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CONFORMING
// CONFORMING: "-cc1"
// CONFORMING-NOT: "-fno-operator-names"
// CONFORMING-NOT: "-fdelayed-template-parsing"

// RUN: %clang --target=x86_64-pc-windows-msvc -c -### -Xarch_host -DFOO#7 -- %s 2>&1 | FileCheck %s \
// RUN:   --check-prefix=MSVC
// RUN: %clang --target=x86_64-windows-msvc -c -### -Xarch_host -DFOO#7 -- %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=MSVC
// RUN: %clang --target=x86_64-win32 -c -### -Xarch_host -DFOO#7 -- %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=MSVC
// RUN: %clang --target=x86_64-unknown-linux-gnu -c -### -Xarch_host -DFOO#7 -- %s 2>&1 | FileCheck %s \
// RUN:   --check-prefix=GNU
// RUN: %clang --target=x86_64-w64-windows-gnu -c -### -DFOO#7 -- %s 2>&1 | FileCheck %s --check-prefix=GNU

// RUN: %clang_cl --target=x86_64-unknown-linux-gnu /EP /DFOO#7 /DTRANSLATION_CHECK -- %s | FileCheck \
// RUN:   %s --check-prefix=PREPROCESS
// PREPROCESS: 7
#ifdef TRANSLATION_CHECK
#if FOO != 7
#error macro separator was not translated
#endif
FOO
#endif
