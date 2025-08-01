// REQUIRES: linux, lld-available
//
// Make sure the build-id can be found in both EXEC and DYN (PIE) files,
// even when the note has been loaded at an offset address in memory.
// (The DYN case would also apply to libraries, not explicitly tested here.)

// DEFINE: %{cflags} =
// DEFINE: %{check} = (                                                           \
// DEFINE:     %clang_profgen -fuse-ld=lld -Wl,--build-id -o %t %s %{cflags}   && \
// DEFINE:     env LLVM_PROFILE_FILE=%t.profraw %run %t                        && \
// DEFINE:     llvm-readelf --notes %t                                         && \
// DEFINE:     llvm-profdata show --binary-ids %t.profraw                         \
// DEFINE:   ) | FileCheck %s

// REDEFINE: %{cflags} = -no-pie
// RUN: %{check}

// REDEFINE: %{cflags} = -pie -fPIE
// RUN: %{check}

// Moving the note after .bss also gives it extra LOAD segment padding,
// making its memory offset different than its file offset.
// RUN: echo "SECTIONS { .note.gnu.build-id : {} } INSERT AFTER .bss;" >%t.script

// REDEFINE: %{cflags} = -no-pie -Wl,--script=%t.script
// RUN: %{check}

// REDEFINE: %{cflags} = -pie -fPIE -Wl,--script=%t.script
// RUN: %{check}

// CHECK-LABEL{LITERAL}: .note.gnu.build-id
// CHECK: Build ID: [[ID:[0-9a-f]+]]

// CHECK-LABEL{LITERAL}: Instrumentation level: Front-end
// CHECK-LABEL{LITERAL}: Binary IDs:
// CHECK-NEXT: [[ID]]

int main() { return 0; }
