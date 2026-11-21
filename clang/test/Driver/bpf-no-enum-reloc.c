// RUN: %clang -### -c --target=bpf -fbpf-no-enum-reloc=e1,e2 \
// RUN:   -fbpf-no-enum-reloc=e3 %s 2>&1 | FileCheck %s
// CHECK: "-cc1"
// CHECK-SAME: "-fbpf-no-enum-reloc=e1,e2" "-fbpf-no-enum-reloc=e3"

// RUN: not %clang -### -c --target=x86_64 -fbpf-no-enum-reloc=e1 %s 2>&1 | \
// RUN:   FileCheck --check-prefix=ERR %s
// ERR: error: unsupported option '-fbpf-no-enum-reloc=' for target 'x86_64'
