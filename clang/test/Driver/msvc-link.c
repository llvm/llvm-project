// RUN: %clang -target i686-pc-windows-msvc -### %s 2>&1 | FileCheck --check-prefix=BASIC %s
// BASIC: link.exe"
// BASIC: "-out:a.exe"
// BASIC: "-defaultlib:libcmt"
// BASIC: "-nologo"

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

