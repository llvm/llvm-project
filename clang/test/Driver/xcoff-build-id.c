// RUN: touch %t.o
// RUN: %clang --target=powerpc64-ibm-aix -### %t.o -mxcoff-build-id=0x12 2>&1 | FileCheck %s
// Test that:
//  1) our ldrinfo flag comes before any user specified ldrinfo flag;
//  2) upper case hex digits are converted to lower case;
//  3) a zero is added when odd number of digits is specified in the HEXSTRING.
// RUN: %clang --target=powerpc-ibm-aix -### %t.o -Wl,-bdbg:ldrinfo:FOO -mxcoff-build-id=0x011ffFF 2>&1 | FileCheck %s --check-prefix=OTHER

// RUN: not %clang --target=powerpc-ibm-aix -### %t.o -mxcoff-build-id=ff 2>&1 | FileCheck %s --check-prefix=BAD_INPUT
// RUN: not %clang --target=powerpc-ibm-aix -### %t.o -mxcoff-build-id=0x0z  2>&1 | FileCheck %s --check-prefix=BAD_INPUT

CHECK: "-bdbg:ldrinfo:xcoff_binary_id:0x12"
OTHER: "-bdbg:ldrinfo:xcoff_binary_id:0x0011ffff" {{.*}} "-bdbg:ldrinfo:FOO"
BAD_INPUT: clang: error: unsupported argument {{.*}} to option '-mxcoff-build-id='
