// REQUIRES: linux
//
// Make sure the build-id can be found in both EXEC and DYN (PIE) files,
// even when the note's section-start is forced to a weird address.
// (The DYN case would also apply to libraries, not explicitly tested here.)

// DEFINE: %{cflags} =
// DEFINE: %{check} = (                                             \
// DEFINE:     %clang_profgen -Wl,--build-id -o %t %s %{cflags}  && \
// DEFINE:     env LLVM_PROFILE_FILE=%t.profraw %run %t          && \
// DEFINE:     llvm-readelf --notes %t                           && \
// DEFINE:     llvm-profdata show --binary-ids %t.profraw           \
// DEFINE:   ) | FileCheck %s

// REDEFINE: %{cflags} = -no-pie
// RUN: %{check}

// REDEFINE: %{cflags} = -pie -fPIE
// RUN: %{check}

// REDEFINE: %{cflags} = -no-pie -Wl,--section-start=.note.gnu.build-id=0x1000000
// RUN: %{check}

// REDEFINE: %{cflags} = -pie -fPIE -Wl,--section-start=.note.gnu.build-id=0x1000000
// RUN: %{check}

// CHECK-LABEL{LITERAL}: .note.gnu.build-id
// CHECK: Build ID: [[ID:[0-9a-f]+]]

// CHECK-LABEL{LITERAL}: Binary IDs:
// CHECK-NEXT: [[ID]]

int main() { return 0; }
