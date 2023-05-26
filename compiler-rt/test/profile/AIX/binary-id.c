// RUN: %clang_pgogen -c -o %t.o %s
//
// Test valid IDs:
// (20-byte, ends with 2 zeroes, upper case)
// RUN: %clang_pgogen -mxcoff-build-id=0x8d7AEC8b900dce6c14afe557dc8889230518be00 -o %t %t.o
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t
// RUN: llvm-profdata show --binary-ids %t.profraw | FileCheck %s --check-prefix=LONG

// (all zeroes)
// RUN: %clang_pgogen -mxcoff-build-id=0x00 -o %t %t.o
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t
// RUN: llvm-profdata show --binary-ids %t.profraw | FileCheck %s --check-prefix=00

// (starts with one zero)
// RUN: %clang_pgogen -mxcoff-build-id=0x0120 -o %t %t.o
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t
// RUN: llvm-profdata show --binary-ids %t.profraw | FileCheck %s --check-prefix=0120

// (starts with 8 zeroes == 4 bytes)
// RUN: %clang_pgogen -mxcoff-build-id=0x0000000012 -o %t %t.o
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t
// RUN: llvm-profdata show --binary-ids %t.profraw | FileCheck %s --check-prefix=0000000012

// (starts with 16 zeroes == 8 bytes)
// RUN: %clang_pgogen -mxcoff-build-id=0x0000000000000000ff -o %t %t.o
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t
// RUN: llvm-profdata show --binary-ids %t.profraw | FileCheck %s --check-prefix=0000000000000000ff

// (starts with 17 zeroes)
// RUN: %clang_pgogen -mxcoff-build-id=0x00000000000000000f -o %t %t.o
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t
// RUN: llvm-profdata show --binary-ids %t.profraw | FileCheck %s --check-prefix=00000000000000000f

// LONG: Binary IDs:
// LONG-NEXT:  8d7aec8b900dce6c14afe557dc8889230518be00
// 00: Binary IDs:
// 00-NEXT:  {{^}}00{{$}}
// 0120: Binary IDs:
// 0120-NEXT:  {{^}}0120{{$}}
// 0000000012: Binary IDs:
// 0000000012-NEXT:  {{^}}0000000012{{$}}
// 0000000000000000ff: Binary IDs:
// 0000000000000000ff-NEXT:  {{^}}0000000000000000ff{{$}}
// 00000000000000000f: Binary IDs:
// 00000000000000000f-NEXT:  {{^}}00000000000000000f{{$}}

int main() { return 0; }
