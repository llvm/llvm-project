// RUN: %clang -target x86_64-pc-win32 -### -no-integrated-as %s -c 2>&1 | FileCheck %s
// CHECK: there is no external assembler that can be used on this platform

// But there is for mingw.  The source file should only be mentioned once for
// the compile step.
// RUN: %clang -target i686-pc-mingw32 -### -no-integrated-as %s -c 2>&1 | FileCheck -check-prefix=MINGW %s
// MINGW: "-cc1"
// MINGW: "-main-file-name" "no-integrated-as-win.c"
// MINGW: "-x" "c" "{{.*}}no-integrated-as-win.c"
// The assembler goes here, but its name depends on PATH.
// MINGW-NOT: no-integrated-as-win.c
