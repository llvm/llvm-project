// Check that -fapple-kext links a kext bundle, without start files or runtime
// libraries. The linker looks up undefined symbols of kexts dynamically on its
// own, so no -undefined is passed.

// RUN: %clang --target=x86_64-apple-macos11 -fapple-kext -### %s 2>&1 | \
// RUN:   FileCheck %s
// CHECK:      {{ld(.exe)?"}}
// CHECK-SAME: "-static" "-arch" "x86_64" "-kext"
// CHECK-NOT:  "-kext"
// CHECK-NOT:  "-undefined"
// CHECK-NOT:  crt
// CHECK-NOT:  "-lSystem"

// RUN: %clang --target=x86_64-apple-macos11 -fapple-kext -bundle \
// RUN:   -bundle_loader foo -### %s 2>&1 | FileCheck %s --check-prefix=BUNDLE
// BUNDLE:     {{ld(.exe)?"}}
// BUNDLE-NOT: "-bundle

// RUN: not %clang --target=x86_64-apple-macos11 -fapple-kext -dynamiclib \
// RUN:   -### %s 2>&1 | FileCheck %s --check-prefix=DYLIB
// DYLIB: error: invalid argument '-fapple-kext' not allowed with '-dynamiclib'
