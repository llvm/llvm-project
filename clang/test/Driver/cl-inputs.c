// Note: %s must be preceded by --, otherwise it may be interpreted as a
// command-line option, e.g. on Mac where %s is commonly under /Users.

// RUN: %clang_cl /TC -### -- %s 2>&1 | FileCheck -check-prefix=TC %s
// TC:  "-x" "c"
// TC-NOT: warning
// TC-NOT: note

// RUN: %clang_cl /TP -### -- %s 2>&1 | FileCheck -check-prefix=TP %s
// TP:  "-x" "c++"
// TP-NOT: warning
// TP-NOT: note

// RUN: %clang_cl -### /Tc%s /TP -- %s 2>&1 | FileCheck -check-prefix=Tc %s
// RUN: %clang_cl -### /TP /Tc%s -- %s 2>&1 | FileCheck -check-prefix=Tc %s
// Tc:  "-x" "c"
// Tc:  "-x" "c++"
// Tc-NOT: warning
// Tc-NOT: note

// RUN: %clang_cl -### /Tp%s /TC -- %s 2>&1 | FileCheck -check-prefix=Tp %s
// RUN: %clang_cl -### /TC /Tp%s -- %s 2>&1 | FileCheck -check-prefix=Tp %s
// Tp:  "-x" "c++"
// Tp:  "-x" "c"
// Tp-NOT: warning
// Tp-NOT: note

// RUN: %clang_cl /TP /TC /TP -### -- %s 2>&1 | FileCheck -check-prefix=WARN %s
// WARN: warning: overriding '/TP' option with '/TC'
// WARN: warning: overriding '/TC' option with '/TP'
// WARN: note: the last '/TC' or '/TP' option takes precedence over earlier instances
// WARN-NOT: note

// MSYS2_ARG_CONV_EXCL tells MSYS2 to skip conversion of the specified argument.
// Add a dummy "other" entry to the path as well, to check that it's split
// around semicolons even on unix.
// RUN: env LIB="other;%S/Inputs/cl-libs" MSYS2_ARG_CONV_EXCL="/TP;/c" %clang_cl /c /TP cl-test.lib -### 2>&1 | FileCheck -check-prefix=TPlib %s
// TPlib: warning: cl-test.lib: 'linker' input unused
// TPlib: warning: argument unused during compilation: '/TP'
// TPlib-NOT: cl-test.lib

// RUN: env LIB=%S/Inputs/cl-libs MSYS2_ARG_CONV_EXCL="/TC;/c" %clang_cl /c /TC cl-test.lib -### 2>&1 | FileCheck -check-prefix=TClib %s
// TClib: warning: cl-test.lib: 'linker' input unused
// TClib: warning: argument unused during compilation: '/TC'
// TClib-NOT: cl-test.lib

// RUN: not %clang_cl - 2>&1 | FileCheck -check-prefix=STDIN %s
// STDIN: error: use /Tc or /Tp

// RUN: %clang_cl -### /Tc - 2>&1 | FileCheck -check-prefix=STDINTc %s
// STDINTc: "-x" "c"

// RUN: env LIB=%S/Inputs/cl-libs %clang_cl -fuse-ld=link -### -- %s cl-test.lib 2>&1 | FileCheck -check-prefix=LIBINPUT %s
// LIBINPUT: link.exe"
// LIBINPUT: "cl-test.lib"

// RUN: env LIB=%S/Inputs/cl-libs %clang_cl -fuse-ld=link -### -- %s cl-test2.lib 2>&1 | FileCheck -check-prefix=LIBINPUT2 %s
// LIBINPUT2: error: no such file or directory: 'cl-test2.lib'
// LIBINPUT2: link.exe"
// LIBINPUT2-NOT: "cl-test2.lib"

// RUN: %clang_cl -fuse-ld=link -### -- %s /nonexisting.lib 2>&1 | FileCheck -check-prefix=LIBINPUT3 %s
// LIBINPUT3: error: no such file or directory: '/nonexisting.lib'
// LIBINPUT3: link.exe"
// LIBINPUT3-NOT: "/nonexisting.lib"

void f();
