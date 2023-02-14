// RUN: %clang -target i686-pc-windows-msvc -fuse-ld=link -### %s 2>&1 | FileCheck --check-prefix=BASIC %s
// BASIC: link.exe"
// BASIC: "-out:a.exe"
// BASIC: "-defaultlib:libcmt"
// BASIC: "-defaultlib:oldnames"
// BASIC: "-nologo"
// BASIC-NOT: "-Brepro"

// RUN: %clang -target i686-pc-windows-msvc -shared -o a.dll -fuse-ld=link -### %s 2>&1 | FileCheck --check-prefix=DLL %s
// DLL: link.exe"
// DLL: "-out:a.dll"
// DLL: "-defaultlib:libcmt"
// DLL: "-defaultlib:oldnames"
// DLL: "-nologo"
// DLL: "-dll"

// RUN: %clang -target i686-pc-windows-msvc -L/var/empty -L/usr/lib -### %s 2>&1 | FileCheck --check-prefix LIBPATH %s
// LIBPATH: "-libpath:/var/empty"
// LIBPATH: "-libpath:/usr/lib"
// LIBPATH: "-nologo"

// RUN: %clang_cl /Brepro -fuse-ld=link -### -- %s 2>&1 | FileCheck --check-prefix=REPRO %s
// REPRO: link.exe"
// REPRO: "-out:msvc-link.exe"
// REPRO: "-nologo"
// REPRO: "-Brepro"

// RUN: %clang_cl /Brepro- -fuse-ld=link -### -- %s 2>&1 | FileCheck --check-prefix=NOREPRO %s
// NOREPRO: link.exe"
// NOREPRO: "-out:msvc-link.exe"
// NOREPRO: "-nologo"
// NOREPRO-NOT: "-Brepro"

// RUN: %clang_cl -fuse-ld=lld --vfsoverlay %s -### -- %s 2>&1 | FileCheck --check-prefix=VFSOVERLAY %s
// VFSOVERLAY: -cc1"
// VFSOVERLAY: "--vfsoverlay"
// VFSOVERLAY: lld-link"
// VFSOVERLAY: "/vfsoverlay:
