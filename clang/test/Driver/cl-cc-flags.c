// Note: %s must be preceded by --, otherwise it may be interpreted as a
// command-line option, e.g. on Mac where %s is commonly under /Users.

// RUN: %clang_cl --target=i686-windows-msvc /Gd -### -- %s 2>&1 | FileCheck --check-prefix=CDECL %s
// CDECL: -fdefault-calling-conv=cdecl

// RUN: %clang_cl --target=i686-windows-msvc /Gr -### -- %s 2>&1 | FileCheck --check-prefix=FASTCALL %s
// FASTCALL: -fdefault-calling-conv=fastcall

// RUN: %clang_cl --target=i686-windows-msvc /Gz -### -- %s 2>&1 | FileCheck --check-prefix=STDCALL %s
// STDCALL: -fdefault-calling-conv=stdcall

// RUN: %clang_cl --target=i686-windows-msvc /Gv -### -- %s 2>&1 | FileCheck --check-prefix=VECTORCALL %s
// VECTORCALL: -fdefault-calling-conv=vectorcall

// Last one should win:

// RUN: %clang_cl --target=i686-windows-msvc /Gd /Gv -### -- %s 2>&1 | FileCheck --check-prefix=LASTWINS_VECTOR %s
// LASTWINS_VECTOR: -fdefault-calling-conv=vectorcall

// RUN: %clang_cl --target=i686-windows-msvc /Gv /Gd -### -- %s 2>&1 | FileCheck --check-prefix=LASTWINS_CDECL %s
// LASTWINS_CDECL: -fdefault-calling-conv=cdecl

// No fastcall or stdcall on x86_64:

// RUN: %clang_cl -Wno-msvc-not-found --target=x86_64-windows-msvc /Gr -### -- %s 2>&1 | FileCheck --check-prefix=UNSUPPORTED %s
// RUN: %clang_cl -Wno-msvc-not-found --target=x86_64-windows-msvc /Gz -### -- %s 2>&1 | FileCheck --check-prefix=UNSUPPORTED %s
// RUN: %clang_cl -Wno-msvc-not-found --target=thumbv7-windows-msvc /Gv -### -- %s 2>&1 | FileCheck --check-prefix=UNSUPPORTED %s

// UNSUPPORTED-NOT: error:
// UNSUPPORTED-NOT: warning:
// UNSUPPORTED-NOT: -fdefault-calling-conv=

