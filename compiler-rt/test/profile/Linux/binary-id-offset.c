// REQUIRES: linux, lld-available
//
// Make sure the build-id can be found in both EXEC and DYN (PIE) files,
// even when the note has been loaded at an offset address in memory.
// (The DYN case would also apply to libraries, not explicitly tested here.)

// DEFINE: %{cflags} =

// REDEFINE: %{cflags} = -no-pie
// RUN: %clang_profgen -fuse-ld=lld -Wl,--build-id -o %t %s %{cflags}
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t
// RUN: llvm-readelf --notes %t > %t2 && llvm-profdata show --binary-ids %t.profraw >> %t2 && FileCheck %s < %t2

// REDEFINE: %{cflags} = -pie -fPIE
// RUN: %clang_profgen -fuse-ld=lld -Wl,--build-id -o %t %s %{cflags}
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t
// RUN: llvm-readelf --notes %t > %t2 && llvm-profdata show --binary-ids %t.profraw >> %t2 && FileCheck %s < %t2

// Moving the note after .bss also gives it extra LOAD segment padding,
// making its memory offset different than its file offset.
// RUN: echo "SECTIONS { .note.gnu.build-id : {} } INSERT AFTER .bss;" >%t.script

// REDEFINE: %{cflags} = -no-pie -Wl,--script=%t.script
// RUN: %clang_profgen -fuse-ld=lld -Wl,--build-id -o %t %s %{cflags}
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t
// RUN: llvm-readelf --notes %t > %t2 && llvm-profdata show --binary-ids %t.profraw >> %t2 && FileCheck %s < %t2

// REDEFINE: %{cflags} = -pie -fPIE -Wl,--script=%t.script
// RUN: %clang_profgen -fuse-ld=lld -Wl,--build-id -o %t %s %{cflags}
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t
// RUN: llvm-readelf --notes %t > %t2 && llvm-profdata show --binary-ids %t.profraw >> %t2 && FileCheck %s < %t2

// CHECK-LABEL{LITERAL}: .note.gnu.build-id
// CHECK: Build ID: [[ID:[0-9a-f]+]]

// CHECK-LABEL{LITERAL}: Instrumentation level: Front-end
// CHECK-LABEL{LITERAL}: Binary IDs:
// CHECK-NEXT: [[ID]]

int main() { return 0; }
