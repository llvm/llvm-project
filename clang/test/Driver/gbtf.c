// Test the -gbtf flag behavior in the Clang driver.
// -gbtf is claimed and ignored for assembler input.
// RUN: %clang --target=x86_64-pc-windows-msvc -gbtf -x assembler -c -### /dev/null 2>&1 \
// RUN:   | FileCheck -check-prefix=ASM %s
// RUN: %clang --target=x86_64-linux-gnu -gbtf -x assembler -c -### /dev/null 2>&1 \
// RUN:   | FileCheck -check-prefix=ASM %s
// ASM: "-cc1as"
// ASM-NOT: warning: argument unused
// ASM-NOT: -debug-info-kind
//
// -gbtf forwards to cc1.
// RUN: %clang --target=x86_64-linux-gnu -gbtf -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=BTF %s
// RUN: %clang --target=x86_64-pc-windows-msvc -g -gbtf -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WINDOWS %s
// RUN: %clang --target=x86_64-pc-windows-msvc -gbtf -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=WINDOWS-NODEBUG %s
// RUN: %clang --target=x86_64-apple-darwin -gbtf -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=DARWIN %s
// RUN: %clang --target=x86_64-linux-gnu -gbtf -g0 -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=BTF-DEBUGINFO %s
// RUN: %clang --target=x86_64-linux-gnu -g0 -gbtf -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=BTF-DEBUGINFO %s
// -gbtf independently enables metadata on ELF, regardless of -g0 order.
//
// -gbtf alone implies debug info (LimitedDebugInfo).
// RUN: %clang --target=x86_64-linux-gnu -gbtf -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=BTF-DEBUGINFO %s
//
// -gbtf with explicit -g also works.
// RUN: %clang --target=x86_64-linux-gnu -g -gbtf -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=BTF %s
//
// Without -gbtf, no -gbtf in cc1 args.
// RUN: %clang --target=x86_64-linux-gnu -g -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=NO-BTF %s
//
// -gbtf works on aarch64 too (any ELF target).
// RUN: %clang --target=aarch64-linux-gnu -gbtf -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=BTF %s

// BTF: "-gbtf"
// BTF-DEBUGINFO: "-debug-info-kind=limited"
// NO-BTF-NOT: "-gbtf"
// WINDOWS-NOT: "-gbtf"
// WINDOWS: "-gcodeview"
// WINDOWS-NOT: "-gbtf"
// WINDOWS-NODEBUG-NOT: "-gbtf"
// WINDOWS-NODEBUG-NOT: "-debug-info-kind=
// WINDOWS-NODEBUG-NOT: "-debug"
// DARWIN-NOT: dsymutil

int main(void) { return 0; }
