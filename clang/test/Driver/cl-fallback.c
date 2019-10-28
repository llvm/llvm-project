// Note: %s must be preceded by --, otherwise it may be interpreted as a
// command-line option, e.g. on Mac where %s is commonly under /Users.

// RUN: %clang_cl --target=i686-pc-win32 /fallback /Dfoo=bar /Ubaz /Ifoo /O0 /Ox /GR /GR- /GS /GS- /Gy /Gy- \
// RUN:   /Gw /Gw- /LD /LDd /EHs /EHs- /Zl /MD /MDd /MTd /MT /guard:cf /guard:cf- /FImyheader.h /Zi \
// RUN:   -garbage -moregarbage \
// RUN:   -### -- %s 2>&1 \
// RUN:   | FileCheck %s
// CHECK: "-fdiagnostics-format" "msvc-fallback"
// CHECK: ||
// CHECK: cl.exe
// CHECK: "/nologo"
// CHECK: "/c"
// CHECK: "/W0"
// CHECK: "-D" "foo=bar"
// CHECK: "-U" "baz"
// CHECK: "-I" "foo"
// CHECK: "/Oi"
// CHECK: "/Og"
// CHECK: "/Ot"
// CHECK: "/Ob2"
// CHECK: "/Oy"
// CHECK: "/GF"
// CHECK: "/GR-"
// CHECK: "/GS-"
// CHECK: "/Gy-"
// CHECK: "/Gw-"
// CHECK: "/Z7"
// CHECK: "/FImyheader.h"
// CHECK: "/LD"
// CHECK: "/LDd"
// CHECK: "/EHs"
// CHECK: "/EHs-"
// CHECK: "/Zl"
// CHECK: "/MT"
// CHECK: "/guard:cf-"
// CHECK: "-garbage"
// CHECK: "-moregarbage"
// CHECK: "/Tc" "{{.*cl-fallback.c}}"
// CHECK: "/Fo{{.*cl-fallback.*.obj}}"

// RUN: %clang_cl /fallback /GR- -### -- %s 2>&1 | FileCheck -check-prefix=GR %s
// GR: cl.exe
// GR: "/GR-"

// RUN: %clang_cl /fallback /GS- -### -- %s 2>&1 | FileCheck -check-prefix=GS %s
// GS: cl.exe
// GS: "/GS-"

// RUN: %clang_cl /fallback /Zc:threadSafeInit -### -- %s 2>&1 | FileCheck -check-prefix=ThreadSafe %s
// ThreadSafe: /Zc:threadSafeInit

// RUN: %clang_cl /fallback /Zc:threadSafeInit- -### -- %s 2>&1 | FileCheck -check-prefix=NonThreadSafe %s
// NonThreadSafe: /Zc:threadSafeInit-

// RUN: %clang_cl /fallback /Od -### -- %s 2>&1 | FileCheck -check-prefix=O0 %s
// O0: cl.exe
// O0: "/Od"
// RUN: %clang_cl --target=i686-pc-win32 /fallback /O1 -### -- %s 2>&1 | FileCheck -check-prefix=O1 %s
// O1: cl.exe
// O1: "/Og" "/Os" "/Ob2" "/Oy" "/GF" "/Gy"
// RUN: %clang_cl --target=i686-pc-win32 /fallback /O2 -### -- %s 2>&1 | FileCheck -check-prefix=O2 %s
// O2: cl.exe
// O2: "/Oi" "/Og" "/Ot" "/Ob2" "/Oy" "/GF" "/Gy"
// RUN: %clang_cl --target=i686-pc-win32 /fallback /Os -### -- %s 2>&1 | FileCheck -check-prefix=Os %s
// Os: cl.exe
// Os: "/Os"
// RUN: %clang_cl --target=i686-pc-win32 /fallback /Ox -### -- %s 2>&1 | FileCheck -check-prefix=Ox %s
// Ox: cl.exe
// Ox: "/Oi" "/Og" "/Ot" "/Ob2" "/Oy" "/GF"

// Only fall back when actually compiling, not for e.g. /P (preprocess).
// RUN: %clang_cl /fallback /P -### -- %s 2>&1 | FileCheck -check-prefix=P %s
// P-NOT: ||
// P-NOT: "cl.exe"

// RUN: not %clang_cl /fallback /c -- %s 2>&1 | \
// RUN:     FileCheck -check-prefix=ErrWarn %s
// ErrWarn: warning: falling back to {{.*}}cl.exe

// RUN: %clang_cl /fallback /c /GR /GR- -### -- %s 2>&1 | \
// RUN:     FileCheck -check-prefix=NO_RTTI %s
// NO_RTTI: "-cc1"
// NO_RTTI: ||
// NO_RTTI: cl.exe
// NO_RTTI: "/GR-"

// Don't fall back on non-C or C++ files.
// RUN: %clang_cl /fallback -### -- %S/Inputs/file.ll 2>&1 | FileCheck -check-prefix=LL %s
// LL: file.ll
// LL-NOT: ||
// LL-NOT: "cl.exe"


#error "This fails to compile."
