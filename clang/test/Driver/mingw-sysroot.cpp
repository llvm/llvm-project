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

// RUN: rm -rf %T/testroot-clang-native
// RUN: mkdir -p %T/testroot-clang-native/bin
// RUN: ln -s %clang %T/testroot-clang-native/bin/clang
// RUN: mkdir -p %T/testroot-clang-native/include/_mingw.h
// RUN: mkdir -p %T/testroot-clang-native/lib/libkernel32.a

// RUN: rm -rf %T/testroot-custom-triple
// RUN: mkdir -p %T/testroot-custom-triple/bin
// RUN: ln -s %clang %T/testroot-custom-triple/bin/clang
// RUN: ln -s %S/Inputs/mingw_ubuntu_posix_tree/usr/x86_64-w64-mingw32 %T/testroot-custom-triple/x86_64-w64-mingw32foo

// If we find a gcc in the path with the right triplet prefix, pick that as
// sysroot:

// This test is only executed on non-Windows systems, i.e. only when
// cross compiling. Check that we don't add the tool root's plain include
// directory to the path - this would end up including /usr/include for
// cross toolchains installed in /usr.

// RUN: env "PATH=%T/testroot-gcc/bin:%PATH%" %clang -target x86_64-w64-mingw32 -rtlib=platform -stdlib=libstdc++ --sysroot="" -c -### %s 2>&1 | FileCheck -check-prefix=CHECK_TESTROOT_GCC %s --implicit-check-not="\"{{.*}}/testroot-gcc{{/|\\\\}}include\""
// CHECK_TESTROOT_GCC: "-internal-isystem" "[[BASE:[^"]+]]/testroot-gcc{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}x86_64-w64-mingw32{{/|\\\\}}10.2-posix{{/|\\\\}}include{{/|\\\\}}c++"
// CHECK_TESTROOT_GCC-SAME: {{^}} "-internal-isystem" "[[BASE]]/testroot-gcc{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}x86_64-w64-mingw32{{/|\\\\}}10.2-posix{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}x86_64-w64-mingw32"
// CHECK_TESTROOT_GCC-SAME: {{^}} "-internal-isystem" "[[BASE]]/testroot-gcc{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}x86_64-w64-mingw32{{/|\\\\}}10.2-posix{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}backward"
// CHECK_TESTROOT_GCC: "-internal-isystem" "[[BASE]]/testroot-gcc{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}x86_64-w64-mingw32{{/|\\\\}}10.2-posix{{/|\\\\}}include{{/|\\\\}}g++-v10.2-posix"
// CHECK_TESTROOT_GCC: "-internal-isystem" "[[BASE]]/testroot-gcc{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}x86_64-w64-mingw32{{/|\\\\}}10.2-posix{{/|\\\\}}include{{/|\\\\}}g++-v10.2"
// CHECK_TESTROOT_GCC: "-internal-isystem" "[[BASE]]/testroot-gcc{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}x86_64-w64-mingw32{{/|\\\\}}10.2-posix{{/|\\\\}}include{{/|\\\\}}g++-v10"
// CHECK_TESTROOT_GCC: "-internal-isystem" "[[BASE]]/testroot-gcc{{/|\\\\}}x86_64-w64-mingw32{{/|\\\\}}include"


// If we pass --sysroot explicitly, then we do include <sysroot>/include
// even when cross compiling.
// RUN: %clang -target x86_64-w64-mingw32 -rtlib=platform -stdlib=libstdc++ --sysroot="%T/testroot-gcc" -c -### %s 2>&1 | FileCheck -check-prefix=CHECK_TESTROOT_GCC_EXPLICIT %s

// CHECK_TESTROOT_GCC_EXPLICIT: "-internal-isystem" "{{[^"]+}}/testroot-gcc{{/|\\\\}}include"


// If -no-canonical-prefixes and there's a matching sysroot next to the clang binary itself, prefer that
// over a gcc in the path:

// RUN: env "PATH=%T/testroot-gcc/bin:%PATH%" %T/testroot-clang/bin/x86_64-w64-mingw32-clang --target=x86_64-w64-mingw32 -rtlib=compiler-rt -stdlib=libstdc++ --sysroot="" -c -### %s 2>&1 | FileCheck -check-prefix=CHECK_TESTROOT_GCC2 %s
// RUN: env "PATH=%T/testroot-gcc/bin:%PATH%" %T/testroot-clang/bin/x86_64-w64-mingw32-clang --target=x86_64-w64-mingw32 -rtlib=compiler-rt -stdlib=libstdc++ --sysroot="" -c -### %s -no-canonical-prefixes 2>&1 | FileCheck -check-prefix=CHECK_TESTROOT_CLANG %s
// CHECK_TESTROOT_GCC2: "{{[^"]+}}/testroot-gcc{{/|\\\\}}x86_64-w64-mingw32{{/|\\\\}}include"
// CHECK_TESTROOT_CLANG: "{{[^"]+}}/testroot-clang{{/|\\\\}}x86_64-w64-mingw32{{/|\\\\}}include"


// If we pick a root based on a sysroot next to the clang binary, which also
// happens to be in the same directory as gcc, make sure we still can pick up
// the libgcc directory:

// RUN: env "PATH=%T/testroot-gcc/bin:%PATH%" %T/testroot-gcc/bin/x86_64-w64-mingw32-clang -target x86_64-w64-mingw32 -rtlib=platform -stdlib=libstdc++ --sysroot="" -c -### %s 2>&1 | FileCheck -check-prefix=CHECK_TESTROOT_GCC %s


// If we're executing clang from a directory with what looks like a mingw sysroot,
// with headers in <base>/include and libs in <base>/lib, use that rather than looking
// for another GCC in the path.
//
// Note, this test has a surprising quirk: We're testing with an install directory,
// testroot-clang-native, which lacks the "x86_64-w64-mingw32" subdirectory, it only
// has the include and lib subdirectories without any triple prefix.
//
// Since commit fd15cb935d7aae25ad62bfe06fe9f17cea585978, we avoid using the
// <base>/include and <base>/lib directories when cross compiling. So technically, this
// case testcase only works exactly as expected when running on x86_64 Windows, when
// this target isn't considered cross compiling.
//
// However we do still pass the include directory <base>/x86_64-w64-mingw32/include to
// the -cc1 interface, even if it is missing. Thus, this test looks for this path name,
// that indicates that we did choose the right base, even if this particular directory
// actually doesn't exist here.

// RUN: env "PATH=%T/testroot-gcc/bin:%PATH%" %T/testroot-clang-native/bin/clang -no-canonical-prefixes --target=x86_64-w64-mingw32 -rtlib=compiler-rt -stdlib=libstdc++ --sysroot="" -c -### %s 2>&1 | FileCheck -check-prefix=CHECK_TESTROOT_CLANG_NATIVE %s
// CHECK_TESTROOT_CLANG_NATIVE: "{{[^"]+}}/testroot-clang-native{{/|\\\\}}x86_64-w64-mingw32{{/|\\\\}}include"


// If the user requests a different arch via the -m32 option, which changes
// x86_64 into i386, check that the driver notices that it can't find a
// sysroot for i386 but there is one for i686, and uses that one.
// (In practice, the real usecase is when using an unprefixed native clang
// that defaults to x86_64 mingw, but it's easier to test this in cross setups
// with symlinks, like the other tests here.)

// RUN: env "PATH=%T/testroot-gcc/bin:%PATH%" %T/testroot-clang/bin/x86_64-w64-mingw32-clang -no-canonical-prefixes --target=x86_64-w64-mingw32 -m32 -rtlib=compiler-rt -stdlib=libstdc++ --sysroot="" -c -### %s 2>&1 | FileCheck -check-prefix=CHECK_TESTROOT_CLANG_I686 %s
// CHECK_TESTROOT_CLANG_I686: "{{[^"]+}}/testroot-clang{{/|\\\\}}i686-w64-mingw32{{/|\\\\}}include"


// If the user calls clang with a custom literal triple, make sure this maps
// to sysroots with the matching spelling.

// RUN: %T/testroot-custom-triple/bin/clang -no-canonical-prefixes --target=x86_64-w64-mingw32foo -rtlib=compiler-rt -stdlib=libstdc++ --sysroot="" -c -### %s 2>&1 | FileCheck -check-prefix=CHECK_TESTROOT_CUSTOM_TRIPLE %s
// CHECK_TESTROOT_CUSTOM_TRIPLE: "{{[^"]+}}/testroot-custom-triple{{/|\\\\}}x86_64-w64-mingw32foo{{/|\\\\}}include"
