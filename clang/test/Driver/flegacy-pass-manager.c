// RUN: %clang -### -c -flegacy-pass-manager -fno-legacy-pass-manager %s 2>&1 | FileCheck --check-prefixes=NOWARN,NEW %s
// RUN: %clang -### -c -fno-legacy-pass-manager -flegacy-pass-manager %s 2>&1 | FileCheck --check-prefixes=NOWARN,LEGACY %s

/// -fexperimental-new-pass-manager is a legacy alias. -fno-experimental-new-pass-manager has been removed.
// RUN: %clang -### -c -fexperimental-new-pass-manager %s 2>&1 | FileCheck --check-prefixes=NOWARN,NEW %s
// RUN: not %clang -### -fno-experimental-new-pass-manager %s

// NOWARN-NOT: warning: argument unused

// NEW:        -fno-legacy-pass-manager
// NEW-NOT:    -flegacy-pass-manager

// LEGACY:     -flegacy-pass-manager
// LEGACY-NOT: -fno-legacy-pass-manager

/// For full/Thin LTO, -fno-legacy-pass-manager passes -plugin-opt=new-pass-manager to the linker (which may not be LLD).
// RUN: %clang -### -target x86_64-linux -flto -fno-legacy-pass-manager %s 2>&1 | FileCheck --check-prefix=LTO_NEW %s
// RUN: %clang -### -target x86_64-linux -flto=thin -fexperimental-new-pass-manager %s 2>&1 | FileCheck --check-prefix=LTO_NEW %s

// LTO_NEW:    "-plugin-opt=new-pass-manager"

// RUN: %clang -### -target x86_64-linux -flto -flegacy-pass-manager %s 2>&1 | FileCheck --check-prefix=LTO_LEGACY %s

// LTO_LEGACY: "-plugin-opt=legacy-pass-manager"

// RUN: %clang -### -target x86_64-linux -flto %s 2>&1 | FileCheck --check-prefix=DEFAULT %s
//
// DEFAULT-NOT: "-plugin-opt=new-pass-manager"
// DEFAULT-NOT: "-plugin-opt=legacy-pass-manager"
