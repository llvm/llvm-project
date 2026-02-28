// Test the -gbtf flag behavior in the Clang driver.
//
// -gbtf forwards to cc1.
// RUN: %clang --target=x86_64-linux-gnu -gbtf -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=BTF %s
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
// BTF-DEBUGINFO: "-debug-info-kind=
// NO-BTF-NOT: "-gbtf"

int main(void) { return 0; }
