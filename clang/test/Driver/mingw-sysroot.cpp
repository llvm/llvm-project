// REQUIRES: shell
// UNSUPPORTED: system-windows

// RUN: mkdir -p %T/testroot-gcc/bin
// RUN: [ ! -s %T/testroot-gcc/bin/x86_64-w64-mingw32-gcc ] || rm %T/testroot-gcc/bin/x86_64-w64-mingw32-gcc
// RUN: [ ! -s %T/testroot-gcc/bin/x86_64-w64-mingw32-clang ] || rm %T/testroot-gcc/bin/x86_64-w64-mingw32-clang
// RUN: [ ! -s %T/testroot-gcc/x86_64-w64-mingw32 ] || rm %T/testroot-gcc/x86_64-w64-mingw32
// RUN: [ ! -s %T/testroot-gcc/lib ] || rm %T/testroot-gcc/lib
// RUN: ln -s %clang %T/testroot-gcc/bin/x86_64-w64-mingw32-gcc
// RUN: ln -s %clang %T/testroot-gcc/bin/x86_64-w64-mingw32-clang
// RUN: ln -s %S/Inputs/mingw_ubuntu_posix_tree/usr/x86_64-w64-mingw32 %T/testroot-gcc/x86_64-w64-mingw32
// RUN: ln -s %S/Inputs/mingw_ubuntu_posix_tree/usr/lib %T/testroot-gcc/lib

// RUN: mkdir -p %T/testroot-clang/bin
// RUN: [ ! -s %T/testroot-clang/bin/x86_64-w64-mingw32-clang ] || rm %T/testroot-clang/bin/x86_64-w64-mingw32-clang
// RUN: [ ! -s %T/testroot-clang/x86_64-w64-mingw32 ] || rm %T/testroot-clang/x86_64-w64-mingw32
// RUN: ln -s %clang %T/testroot-clang/bin/x86_64-w64-mingw32-clang
// RUN: ln -s %S/Inputs/mingw_ubuntu_posix_tree/usr/x86_64-w64-mingw32 %T/testroot-clang/x86_64-w64-mingw32


// If we find a gcc in the path with the right triplet prefix, pick that as
// sysroot:

// RUN: env "PATH=%T/testroot-gcc/bin:%PATH%" %clang -target x86_64-w64-mingw32 -rtlib=platform -stdlib=libstdc++ -c -### %s 2>&1 | FileCheck -check-prefix=CHECK_TESTROOT_GCC %s
// CHECK_TESTROOT_GCC: "{{.*}}/testroot-gcc{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}x86_64-w64-mingw32{{/|\\\\}}5.3-posix{{/|\\\\}}include{{/|\\\\}}c++"
// CHECK_TESTROOT_GCC: "{{.*}}/testroot-gcc{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}x86_64-w64-mingw32{{/|\\\\}}5.3-posix{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}x86_64-w64-mingw32"
// CHECK_TESTROOT_GCC: "{{.*}}/testroot-gcc{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}x86_64-w64-mingw32{{/|\\\\}}5.3-posix{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}backward"
// CHECK_TESTROOT_GCC: "{{.*}}/testroot-gcc{{/|\\\\}}x86_64-w64-mingw32{{/|\\\\}}include"


// If there's a matching sysroot next to the clang binary itself, prefer that
// over a gcc in the path:

// RUN: env "PATH=%T/testroot-gcc/bin:%PATH%" %T/testroot-clang/bin/x86_64-w64-mingw32-clang -target x86_64-w64-mingw32 -rtlib=compiler-rt -stdlib=libstdc++ -c -### %s 2>&1 | FileCheck -check-prefix=CHECK_TESTROOT_CLANG %s
// CHECK_TESTROOT_CLANG: "{{.*}}/testroot-clang{{/|\\\\}}x86_64-w64-mingw32{{/|\\\\}}include"


// If we pick a root based on a sysroot next to the clang binary, which also
// happens to be in the same directory as gcc, make sure we still can pick up
// the libgcc directory:

// RUN: env "PATH=%T/testroot-gcc/bin:%PATH%" %T/testroot-gcc/bin/x86_64-w64-mingw32-clang -target x86_64-w64-mingw32 -rtlib=platform -stdlib=libstdc++ -c -### %s 2>&1 | FileCheck -check-prefix=CHECK_TESTROOT_GCC %s
