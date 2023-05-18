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
// VFSOVERLAY: lld-link
// VFSOVERLAY: "/vfsoverlay:{{.*}}" "{{.*}}.obj"

// RUN: %clang -target arm64ec-pc-windows-msvc -fuse-ld=link -### %s 2>&1 | FileCheck --check-prefix=ARM64EC %s
// RUN: %clang_cl -target arm64ec-pc-windows-msvc -fuse-ld=link -### -- %s 2>&1 | FileCheck --check-prefix=ARM64EC %s
// RUN: %clang_cl -arm64EC -fuse-ld=link -### -- %s 2>&1 | FileCheck --check-prefix=ARM64EC %s
// ARM64EC: "-machine:arm64ec"

// RUN: %clang -target arm64ec-pc-windows-msvc -fuse-ld=link -marm64x -### %s 2>&1 | \
// RUN:        FileCheck --check-prefix=ARM64X %s
// RUN: %clang -target aarch64-pc-windows-msvc -fuse-ld=link -marm64x -### %s 2>&1 | \
// RUN:        FileCheck --check-prefix=ARM64X %s
// RUN: %clang_cl -marm64x -fuse-ld=link -### -- %s 2>&1 | FileCheck --check-prefix=ARM64X %s
// RUN: %clang_cl -arm64EC -marm64x -fuse-ld=link -### -- %s 2>&1 | FileCheck --check-prefix=ARM64X %s
// ARM64X: "-machine:arm64x"

// RUN: not %clang -target x86_64-linux-gnu -marm64x -### %s 2>&1 | FileCheck --check-prefix=HYBRID-ERR %s
// HYBRID-ERR: error: unsupported option '-marm64x' for target 'x86_64-linux-gnu'

// RUN: %clang -c -marm64x  -target arm64ec-pc-windows-msvc -fuse-ld=link -### %s 2>&1 | \
// RUN:        FileCheck --check-prefix=HYBRID-WARN %s
// HYBRID-WARN: warning: argument unused during compilation: '-marm64x' [-Wunused-command-line-argument]
