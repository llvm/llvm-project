// RUN: %clang -target i686-pc-windows-msvc -### %s 2>&1 | FileCheck --check-prefix=BASIC %s
// BASIC: link.exe"
// BASIC: "-out:a.exe"
// BASIC: "-defaultlib:libcmt"
// BASIC: "-nologo"
// BASIC-NOT: "-Brepro"

// RUN: %clang -target i686-pc-windows-msvc -shared -o a.dll -### %s 2>&1 | FileCheck --check-prefix=DLL %s
// DLL: link.exe"
// DLL: "-out:a.dll"
// DLL: "-defaultlib:libcmt"
// DLL: "-nologo"
// DLL: "-dll"

// RUN: %clang -target i686-pc-windows-msvc -L/var/empty -L/usr/lib -### %s 2>&1 | FileCheck --check-prefix LIBPATH %s
// LIBPATH: "-libpath:/var/empty"
// LIBPATH: "-libpath:/usr/lib"
// LIBPATH: "-nologo"

// RUN: %clang_cl /Brepro -### -- %s 2>&1 | FileCheck --check-prefix=REPRO %s
// REPRO: link.exe"
// REPRO: "-out:msvc-link.exe"
// REPRO: "-nologo"
// REPRO: "-Brepro"

// RUN: %clang_cl /Brepro- -### -- %s 2>&1 | FileCheck --check-prefix=NOREPRO %s
// NOREPRO: link.exe"
// NOREPRO: "-out:msvc-link.exe"
// NOREPRO: "-nologo"
// NOREPRO-NOT: "-Brepro"
