// Check that BOLT patches the build ID of the output binary so that it cannot
// be mistaken for the input. The high bit of the first byte and the low bit of
// the last byte are flipped.
//
// REQUIRES: system-linux

// RUN: %clang %cflags -Wl,-q %s -o %t.exe \
// RUN:   -Wl,--build-id=0x0123456789abcdef0123456789abcdef01234567
// RUN: llvm-readelf -n %t.exe | FileCheck %s --check-prefix=CHECK-INPUT
// RUN: llvm-bolt %t.exe -o %t.bolt | FileCheck %s --check-prefix=CHECK-BOLT
// RUN: llvm-readelf -n %t.bolt | FileCheck %s --check-prefix=CHECK-OUTPUT

// CHECK-INPUT: Build ID: 0123456789abcdef0123456789abcdef01234567
// CHECK-BOLT: BOLT-INFO: patched build-id
// CHECK-OUTPUT: Build ID: 8123456789abcdef0123456789abcdef01234566

int main() { return 0; }
