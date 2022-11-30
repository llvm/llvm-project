// REQUIRES: shell
// UNSUPPORTED: system-windows

// RUN: rm -rf %T/testroot-gcc
// RUN: mkdir -p %T/testroot-gcc/bin
// RUN: ln -s %clang %T/testroot-gcc/bin/x86_64-w64-mingw32-gcc
// RUN: ln -s %clang %T/testroot-gcc/bin/x86_64-w64-mingw32-clang
// RUN: ln -s %S/Inputs/mingw_ubuntu_posix_tree/usr/x86_64-w64-mingw32 %T/testroot-gcc/x86_64-w64-mingw32
// RUN: ln -s %S/Inputs/mingw_ubuntu_posix_tree/usr/lib %T/testroot-gcc/lib

// RUN: rm -rf %T/testroot-clang
// RUN: mkdir -p %T/testroot-clang/bin
// RUN: ln -s %clang %T/testroot-clang/bin/x86_64-w64-mingw32-clang
// RUN: ln -s %S/Inputs/mingw_ubuntu_posix_tree/usr/x86_64-w64-mingw32 %T/testroot-clang/x86_64-w64-mingw32
// RUN: ln -s %S/Inputs/mingw_arch_tree/usr/i686-w64-mingw32 %T/testroot-clang/i686-w64-mingw32

// RUN: rm -rf %T/testroot-custom-triple
// RUN: mkdir -p %T/testroot-custom-triple/bin
// RUN: ln -s %clang %T/testroot-custom-triple/bin/clang
// RUN: ln -s %S/Inputs/mingw_ubuntu_posix_tree/usr/x86_64-w64-mingw32 %T/testroot-custom-triple/x86_64-w64-mingw32foo

// If we find a gcc in the path with the right triplet prefix, pick that as
// sysroot:

// RUN: env "PATH=%T/testroot-gcc/bin:%PATH%" %clang -target x86_64-w64-mingw32 -rtlib=platform -stdlib=libstdc++ --sysroot="" -c -### %s 2>&1 | FileCheck -check-prefix=CHECK_TESTROOT_GCC %s
// CHECK_TESTROOT_GCC: "-internal-isystem" "[[BASE:[^"]+]]/testroot-gcc{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}x86_64-w64-mingw32{{/|\\\\}}10.2-posix{{/|\\\\}}include{{/|\\\\}}c++"
// CHECK_TESTROOT_GCC-SAME: {{^}} "-internal-isystem" "[[BASE]]/testroot-gcc{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}x86_64-w64-mingw32{{/|\\\\}}10.2-posix{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}x86_64-w64-mingw32"
// CHECK_TESTROOT_GCC-SAME: {{^}} "-internal-isystem" "[[BASE]]/testroot-gcc{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}x86_64-w64-mingw32{{/|\\\\}}10.2-posix{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}backward"
// CHECK_TESTROOT_GCC: "-internal-isystem" "[[BASE]]/testroot-gcc{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}x86_64-w64-mingw32{{/|\\\\}}10.2-posix{{/|\\\\}}include{{/|\\\\}}g++-v10.2-posix"
// CHECK_TESTROOT_GCC: "-internal-isystem" "[[BASE]]/testroot-gcc{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}x86_64-w64-mingw32{{/|\\\\}}10.2-posix{{/|\\\\}}include{{/|\\\\}}g++-v10.2"
// CHECK_TESTROOT_GCC: "-internal-isystem" "[[BASE]]/testroot-gcc{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}x86_64-w64-mingw32{{/|\\\\}}10.2-posix{{/|\\\\}}include{{/|\\\\}}g++-v10"
// CHECK_TESTROOT_GCC: "-internal-isystem" "[[BASE]]/testroot-gcc{{/|\\\\}}x86_64-w64-mingw32{{/|\\\\}}include"


// If there's a matching sysroot next to the clang binary itself, prefer that
// over a gcc in the path:

// RUN: env "PATH=%T/testroot-gcc/bin:%PATH%" %T/testroot-clang/bin/x86_64-w64-mingw32-clang -target x86_64-w64-mingw32 -rtlib=compiler-rt -stdlib=libstdc++ --sysroot="" -c -### %s 2>&1 | FileCheck -check-prefix=CHECK_TESTROOT_CLANG %s
// CHECK_TESTROOT_CLANG: "{{[^"]+}}/testroot-clang{{/|\\\\}}x86_64-w64-mingw32{{/|\\\\}}include"


// If we pick a root based on a sysroot next to the clang binary, which also
// happens to be in the same directory as gcc, make sure we still can pick up
// the libgcc directory:

// RUN: env "PATH=%T/testroot-gcc/bin:%PATH%" %T/testroot-gcc/bin/x86_64-w64-mingw32-clang -target x86_64-w64-mingw32 -rtlib=platform -stdlib=libstdc++ --sysroot="" -c -### %s 2>&1 | FileCheck -check-prefix=CHECK_TESTROOT_GCC %s


// If the user requests a different arch via the -m32 option, which changes
// x86_64 into i386, check that the driver notices that it can't find a
// sysroot for i386 but there is one for i686, and uses that one.
// (In practice, the real usecase is when using an unprefixed native clang
// that defaults to x86_64 mingw, but it's easier to test this in cross setups
// with symlinks, like the other tests here.)

// RUN: env "PATH=%T/testroot-gcc/bin:%PATH%" %T/testroot-clang/bin/x86_64-w64-mingw32-clang --target=x86_64-w64-mingw32 -m32 -rtlib=compiler-rt -stdlib=libstdc++ --sysroot="" -c -### %s 2>&1 | FileCheck -check-prefix=CHECK_TESTROOT_CLANG_I686 %s
// CHECK_TESTROOT_CLANG_I686: "{{[^"]+}}/testroot-clang{{/|\\\\}}i686-w64-mingw32{{/|\\\\}}include"


// If the user calls clang with a custom literal triple, make sure this maps
// to sysroots with the matching spelling.

// RUN: %T/testroot-custom-triple/bin/clang --target=x86_64-w64-mingw32foo -rtlib=compiler-rt -stdlib=libstdc++ --sysroot="" -c -### %s 2>&1 | FileCheck -check-prefix=CHECK_TESTROOT_CUSTOM_TRIPLE %s
// CHECK_TESTROOT_CUSTOM_TRIPLE: "{{[^"]+}}/testroot-custom-triple{{/|\\\\}}x86_64-w64-mingw32foo{{/|\\\\}}include"
