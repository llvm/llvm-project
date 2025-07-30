// REQUIRES: default-cxx-stdlib=libc++
// UNSUPPORTED: system-windows
//   Windows is unsupported because we use the Unix path separator `/` in the test.

// Unlike the Darwin driver, the MachO driver doesn't add any framework search paths,
// only the normal header ones.
// RUN: %clang -x c -target arm64-apple-none-macho -isysroot %S/Inputs/MacOSX15.1.sdk -### -c %s 2>&1 \
// RUN: | FileCheck --check-prefixes=CC1,NO-CXX,ULI,CI,UI,NO-FW -DSDKROOT=%S/Inputs/MacOSX15.1.sdk %s

// Unlike the Darwin driver, the MachO driver doesn't default to libc++, but when
// CLANG_DEFAULT_CXX_STDLIB is libc++ then the MachO driver should find the search path.
// RUN: %clang -x c++ -target arm64-apple-none-macho -isysroot %S/Inputs/MacOSX15.1.sdk -### -c %s 2>&1 \
// RUN: | FileCheck --check-prefixes=CC1,CXX,ULI,CI,UI,NO-FW -DSDKROOT=%S/Inputs/MacOSX15.1.sdk %s

// If the user requests libc++, the MachO driver should still find the search path.
// RUN: %clang -x c++ -stdlib=libc++ -target arm64-apple-none-macho -isysroot %S/Inputs/MacOSX15.1.sdk -### -c %s 2>&1 \
// RUN: | FileCheck --check-prefixes=CC1,CXX,ULI,CI,UI,NO-FW -DSDKROOT=%S/Inputs/MacOSX15.1.sdk %s

// Verify that embedded uses can swap in alternate usr/include and usr/local/include directories.
// usr/local/include is specified in the driver as -internal-isystem, however, the driver generated
// paths come before the paths in the driver arguments. In order to keep usr/local/include in the
// same position, -isystem has to be used instead of -Xclang -internal-isystem. There isn't an
// -externc-isystem, but it's ok to use -Xclang -internal-externc-isystem since the driver doesn't
// use that if -nostdlibinc or -nostdinc is passed.
// RUN: %clang -x c++ -stdlib=libc++ -target arm64-apple-none-macho -isysroot %S/Inputs/MacOSX15.1.sdk \
// RUN:        -nostdlibinc -isystem %S/Inputs/MacOSX15.1.sdk/embedded/usr/local/include \
// RUN:        -Xclang -internal-externc-isystem -Xclang %S/Inputs/MacOSX15.1.sdk/embedded/usr/include \
// RUN:        -### -c %s 2>&1 | FileCheck --check-prefixes=CC1,NO-CXX,EULI,CI,EUI,NO-FW -DSDKROOT=%S/Inputs/MacOSX15.1.sdk %s


// The ordering of these flags doesn't matter, and so this test is a little
// fragile. i.e. all of the -internal-isystem paths will be searched before the
// -internal-externc-isystem ones, and their order on the command line doesn't
// matter. The line order here is just the current order that the driver writes
// the cc1 arguments.

// CC1: "-cc1"
// NO-CXX-NOT: "-internal-isystem" "{{.*}}/include/c++/v1"
// CXX-SAME: "-internal-isystem" "{{.*}}/include/c++/v1"
// ULI-SAME: "-internal-isystem" "[[SDKROOT]]/usr/local/include"
// EULI-SAME: "-isystem" "[[SDKROOT]]/embedded/usr/local/include"
// CI-SAME: "-internal-isystem" "{{.*}}/clang/{{[[:digit:].]*}}/include"
// UI-SAME: "-internal-externc-isystem" "[[SDKROOT]]/usr/include"
// EUI-SAME: "-internal-externc-isystem" "[[SDKROOT]]/embedded/usr/include"
// NO-FW-NOT: "-internal-iframework"
