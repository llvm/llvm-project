// RUN: %clang -### -c --target=x86_64 -Wa,--reloc-section-sym=internal %s -Werror 2>&1 | FileCheck %s
// RUN: %clang -### -c --target=aarch64 -Wa,--reloc-section-sym=internal,--reloc-section-sym=none %s -Werror 2>&1 | FileCheck %s --check-prefix=NONE
// RUN: not %clang -### -c --target=arm64-apple-darwin -Wa,--reloc-section-sym=internal %s 2>&1 | FileCheck %s --check-prefix=ERR
// RUN: not %clang -### -c --target=x86_64 -Wa,--reloc-section-sym=x %s 2>&1 | FileCheck %s --check-prefix=BADVAL

// CHECK:      "--reloc-section-sym=internal"
// NONE:       "--reloc-section-sym=none"
// ERR:        error: unsupported option '-Wa,--reloc-section-sym' for target
// BADVAL:     error: invalid value 'x' in '-Wa,--reloc-section-sym=x'
