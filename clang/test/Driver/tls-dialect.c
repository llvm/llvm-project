/// Options for ELF
// RUN: %clang -### -target aarch64-linux-gnu -mtls-dialect=trad %s 2>&1 | FileCheck -check-prefix=TRAD %s
// RUN: %clang -### -target aarch64-linux-gnu -mtls-dialect=desc %s 2>&1 | FileCheck -check-prefix=DESC %s

/// Unsupported target
// RUN: not %clang -target aarch64-unknown-windows-msvc -mtls-dialect=trad %s 2>&1 | FileCheck -check-prefix=UNSUPPORTED-TARGET %s
// RUN: not %clang -target aarch64-unknown-windows-msvc -mtls-dialect=desc %s 2>&1 | FileCheck -check-prefix=UNSUPPORTED-TARGET %s

/// Invalid option value
// RUN: not %clang -target x86_64-linux-gnu -mtls-dialect=foo %s 2>&1 | FileCheck -check-prefix=INVALID-VALUE %s

// TRAD: "-cc1" {{.*}}"-mtls-dialect=trad"
// DESC: "-cc1" {{.*}}"-mtls-dialect=desc"
// UNSUPPORTED-TARGET: error: unsupported option
// INVALID-VALUE: error: invalid value 'foo' in '-mtls-dialect=foo'
