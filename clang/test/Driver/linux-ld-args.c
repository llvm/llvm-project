/// Tests that gnutools::Linker::ConstructJob forwards user command-line options
/// to the ld invocation. Sysroot-dependent toolchain discovery and
/// crt*/library-path selection live in linux-ld.c.

/// Mixed -Xlinker, -Wl, and standalone forwarding. -T is reordered to the end
/// so that earlier -L paths take precedence.
// RUN: %clang --target=x86_64-linux-gnu -### \
// RUN:   -e _start -T a.lds -t -Xlinker one -Xlinker --no-demangle \
// RUN:   -Wl,two,--no-demangle,three -Xlinker four -z five -r %s 2>&1 | \
// RUN:   FileCheck --check-prefix=MIXED %s
// MIXED: "--no-demangle" "-e" "_start" "one" "two" "three" "four" "-z" "five" "-r" {{.*}} "-T" "a.lds" "-t"

/// Passthrough flags. -rdynamic becomes -export-dynamic; -Wl,-z keyword pairs
/// are forwarded as separate "-z" KEYWORD tokens.
// RUN: %clang --target=x86_64-linux-gnu -### \
// RUN:   -s -u my_sym -rdynamic -Wl,-z,relro,-z,now %s 2>&1 | \
// RUN:   FileCheck --check-prefix=PASS %s
// PASS:      "-s"
// PASS-SAME: "-export-dynamic"
// PASS-SAME: "-u" "my_sym"
// PASS-SAME: "-z" "relro" "-z" "now"

/// -nopie is OpenBSD-specific.
// RUN: not %clang -### --target=x86_64-unknown-linux-gnu %s -nopie 2>&1 | FileCheck %s --check-prefix=CHECK-NOPIE
// CHECK-NOPIE: error: unsupported option '-nopie' for target 'x86_64-unknown-linux-gnu'
