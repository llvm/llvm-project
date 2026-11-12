// By default, we don't pass any -fauto-import to -cc1, as that's the default.
//
// RUN: %clang --target=x86_64-w64-windows-gnu -### %s 2>&1 | FileCheck --check-prefixes=DEFAULT %s
// RUN: %clang --target=x86_64-w64-windows-gnu -fno-auto-import -fauto-import -### %s 2>&1 | FileCheck --check-prefixes=DEFAULT %s
// DEFAULT: "-cc1"
// DEFAULT-NOT: no-auto-import
// DEFAULT-NOT: --disable-auto-import

// When compiling with -fno-auto-import, we pass -fno-auto-import to -cc1
// and --disable-auto-import to the linker.
//
// RUN: %clang --target=x86_64-w64-windows-gnu -fauto-import -fno-auto-import -### %s 2>&1 | FileCheck --check-prefixes=NO_AUTOIMPORT %s
// NO_AUTOIMPORT: "-cc1"
// NO_AUTOIMPORT: "-fno-auto-import"
// NO_AUTOIMPORT: "--disable-auto-import"
