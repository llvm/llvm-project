// This test is intended to validate whether decimal floating-point (DFP)
// extensions are supported and working for a given target and to ensure
// that an appropriate diagnostic is issued when DFP features are used
// otherwise.

// FIXME: Remove all uses of -fexperimental-decimal-floating-point once -std=c23 implies DFP enablement.
// RUN: %clang -target aarch64-unknown-freebsd       -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target aarch64-unknown-linux         -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target aarch64-unknown-linux-android -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target aarch64-unknown-windows-gnu   -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target aarch64-unknown-windows-msvc  -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target arm64-apple-darwin            -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target arm64-apple-ios               -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target arm64-apple-macos             -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target arm64-apple-tvos              -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target arm64-apple-watchos           -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target arm64-unknown-linux           -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target arm64-unknown-linux-android   -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target armv7-unknown-linux           -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target armv7-unknown-linux-android   -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target armv7-unknown-netbsd          -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target armv7-unknown-openbsd         -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target armv7-unknown-windows-gnu     -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target armv7-unknown-windows-msvc    -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target hexagon-unknown-linux         -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target i686-unknown-linux            -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify %s
// RUN: %clang -target i686-unknown-linux-android    -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target i686-unknown-windows-gnu      -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target i686-unknown-windows-msvc     -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target loongarch32-unknown-linux     -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target loongarch64-unknown-linux     -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target m68k-unknown-linux            -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target mips-unknown-freebsd          -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target mips-unknown-linux            -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target mips64-unknown-freebsd        -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target mips64-unknown-linux          -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target powerpc64-ibm-aix             -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target powerpc64-unknown-linux       -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target powerpc-ibm-aix               -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target powerpc-unknown-linux         -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target riscv32-unknown-linux         -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target riscv64-unknown-freebsd       -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target riscv64-unknown-fuchsia       -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target riscv64-unknown-linux         -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target riscv64-unknown-openbsd       -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target s390x-ibm-linux               -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target s390x-ibm-zos                 -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target sparc-sun-solaris             -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target sparc-unknown-freebsd         -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target sparc-unknown-linux           -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target sparc-unknown-netbsd          -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target sparcv9-sun-solaris           -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target sparc64-unknown-freebsd       -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target sparc64-unknown-linux         -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target sparc64-unknown-netbsd        -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target x86_64-apple-darwin           -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target x86_64-apple-ios              -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target x86_64-apple-macos            -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target x86_64-apple-tvos             -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target x86_64-apple-watchos          -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target x86_64-unknown-freebsd        -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target x86_64-unknown-fuchsia        -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target x86_64-unknown-haiku          -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target x86_64-unknown-linux          -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify %s
// RUN: %clang -target x86_64-unknown-netbsd         -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target x86_64-unknown-openbsd        -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target x86_64-unknown-windows-gnu    -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s
// RUN: %clang -target x86_64-unknown-windows-msvc   -std=c23 -fsyntax-only -fexperimental-decimal-floating-point -Xclang -verify=unsupported %s

// expected-no-diagnostics

_Decimal32 d32;   // unsupported-error {{decimal floating-point extensions are not supported on the current target}}
_Decimal64 d64;   // unsupported-error {{decimal floating-point extensions are not supported on the current target}}
_Decimal128 d128; // unsupported-error {{decimal floating-point extensions are not supported on the current target}}

typedef float __attribute__((mode(SD))) D32;  // unsupported-error {{decimal floating-point extensions are not supported on the current target}}
typedef float __attribute__((mode(DD))) D64;  // unsupported-error {{decimal floating-point extensions are not supported on the current target}}
typedef float __attribute__((mode(TD))) D128; // unsupported-error {{decimal floating-point extensions are not supported on the current target}}

float __attribute__((mode(SD))) famsd; // unsupported-error {{decimal floating-point extensions are not supported on the current target}}
float __attribute__((mode(DD))) famdd; // unsupported-error {{decimal floating-point extensions are not supported on the current target}}
float __attribute__((mode(TD))) famtd; // unsupported-error {{decimal floating-point extensions are not supported on the current target}}
