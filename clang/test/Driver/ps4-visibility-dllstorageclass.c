// Check behaviour of -fvisibility-from-dllstorageclass options for PS4

// RUN: %clang -### -target x86_64-scei-ps4 %s -Werror -o - 2>&1 | \
// RUN:   FileCheck %s --check-prefix=DEFAULTS \
// RUN:     --implicit-check-not=-fvisibility-from-dllstorageclass \
// RUN:     --implicit-check-not=-fvisibility-dllexport \
// RUN:     --implicit-check-not=-fvisibility-nodllstorageclass \
// RUN:     --implicit-check-not=-fvisibility-externs-dllimport \
// RUN:     --implicit-check-not=-fvisibility-externs-nodllstorageclass

// RUN: %clang -### -target x86_64-scei-ps4 \
// RUN:     -fno-visibility-from-dllstorageclass \
// RUN:     -fvisibility-from-dllstorageclass \
// RUN:     -Werror \
// RUN:     %s -o - 2>&1 | \
// RUN:   FileCheck %s --check-prefix=DEFAULTS \
// RUN:     --implicit-check-not=-fvisibility-from-dllstorageclass \
// RUN:     --implicit-check-not=-fvisibility-dllexport \
// RUN:     --implicit-check-not=-fvisibility-nodllstorageclass \
// RUN:     --implicit-check-not=-fvisibility-externs-dllimport \
// RUN:     --implicit-check-not=-fvisibility-externs-nodllstorageclass

// DEFAULTS:      "-fvisibility-from-dllstorageclass"
// DEFAULTS-SAME: "-fvisibility-dllexport=protected"
// DEFAULTS-SAME: "-fvisibility-nodllstorageclass=hidden"
// DEFAULTS-SAME: "-fvisibility-externs-dllimport=default"
// DEFAULTS-SAME: "-fvisibility-externs-nodllstorageclass=default"

// RUN: %clang -### -target x86_64-scei-ps4 \
// RUN:     -fvisibility-from-dllstorageclass \
// RUN:     -fvisibility-dllexport=hidden \
// RUN:     -fvisibility-nodllstorageclass=protected \
// RUN:     -fvisibility-externs-dllimport=hidden \
// RUN:     -fvisibility-externs-nodllstorageclass=protected \
// RUN:     -fno-visibility-from-dllstorageclass \
// RUN:     %s -o - 2>&1 | \
// RUN:   FileCheck %s -check-prefix=UNUSED \
// RUN:     --implicit-check-not=-fvisibility-from-dllstorageclass \
// RUN:     --implicit-check-not=-fvisibility-dllexport \
// RUN:     --implicit-check-not=-fvisibility-nodllstorageclass \
// RUN:     --implicit-check-not=-fvisibility-externs-dllimport \
// RUN:     --implicit-check-not=-fvisibility-externs-nodllstorageclass \
// RUN:     --implicit-check-not=warning:

// UNUSED:      warning: argument unused during compilation: '-fvisibility-dllexport=hidden'
// UNUSED-NEXT: warning: argument unused during compilation: '-fvisibility-nodllstorageclass=protected'
// UNUSED-NEXT: warning: argument unused during compilation: '-fvisibility-externs-dllimport=hidden'
// UNUSED-NEXT: warning: argument unused during compilation: '-fvisibility-externs-nodllstorageclass=protected'

// RUN: %clang -### -target x86_64-scei-ps4 \
// RUN:     -fvisibility-nodllstorageclass=protected \
// RUN:     -fvisibility-externs-dllimport=hidden \
// RUN:     -Werror \
// RUN:     %s -o - 2>&1 | \
// RUN:   FileCheck %s -check-prefix=SOME \
// RUN:     --implicit-check-not=-fvisibility-from-dllstorageclass \
// RUN:     --implicit-check-not=-fvisibility-dllexport \
// RUN:     --implicit-check-not=-fvisibility-nodllstorageclass \
// RUN:     --implicit-check-not=-fvisibility-externs-dllimport \
// RUN:     --implicit-check-not=-fvisibility-externs-nodllstorageclass

// RUN: %clang -### -target x86_64-scei-ps4 \
// RUN:     -fvisibility-from-dllstorageclass \
// RUN:     -fvisibility-nodllstorageclass=protected \
// RUN:     -fvisibility-externs-dllimport=hidden \
// RUN:     -Werror \
// RUN:     %s -o - 2>&1 | \
// RUN:   FileCheck %s -check-prefix=SOME \
// RUN:     --implicit-check-not=-fvisibility-from-dllstorageclass \
// RUN:     --implicit-check-not=-fvisibility-dllexport \
// RUN:     --implicit-check-not=-fvisibility-nodllstorageclass \
// RUN:     --implicit-check-not=-fvisibility-externs-dllimport \
// RUN:     --implicit-check-not=-fvisibility-externs-nodllstorageclass

// SOME:      "-fvisibility-from-dllstorageclass"
// SOME-SAME: "-fvisibility-dllexport=protected"
// SOME-SAME: "-fvisibility-nodllstorageclass=protected"
// SOME-SAME: "-fvisibility-externs-dllimport=hidden"
// SOME-SAME: "-fvisibility-externs-nodllstorageclass=default"

// RUN: %clang -### -target x86_64-scei-ps4 \
// RUN:     -fvisibility-dllexport=default \
// RUN:     -fvisibility-dllexport=hidden \
// RUN:     -fvisibility-nodllstorageclass=default \
// RUN:     -fvisibility-nodllstorageclass=protected \
// RUN:     -fvisibility-externs-dllimport=default \
// RUN:     -fvisibility-externs-dllimport=hidden \
// RUN:     -fvisibility-externs-nodllstorageclass=default \
// RUN:     -fvisibility-externs-nodllstorageclass=protected \
// RUN:     -Werror \
// RUN:     %s -o - 2>&1 | \
// RUN:   FileCheck %s -check-prefix=ALL \
// RUN:     --implicit-check-not=-fvisibility-from-dllstorageclass \
// RUN:     --implicit-check-not=-fvisibility-dllexport \
// RUN:     --implicit-check-not=-fvisibility-nodllstorageclass \
// RUN:     --implicit-check-not=-fvisibility-externs-dllimport \
// RUN:     --implicit-check-not=-fvisibility-externs-nodllstorageclass

// RUN: %clang -### -target x86_64-scei-ps4 \
// RUN:     -fvisibility-from-dllstorageclass \
// RUN:     -fvisibility-dllexport=default \
// RUN:     -fvisibility-dllexport=hidden \
// RUN:     -fvisibility-nodllstorageclass=default \
// RUN:     -fvisibility-nodllstorageclass=protected \
// RUN:     -fvisibility-externs-dllimport=default \
// RUN:     -fvisibility-externs-dllimport=hidden \
// RUN:     -fvisibility-externs-nodllstorageclass=default \
// RUN:     -fvisibility-externs-nodllstorageclass=protected \
// RUN:     -Werror \
// RUN:     %s -o - 2>&1 | \
// RUN:   FileCheck %s -check-prefix=ALL \
// RUN:     --implicit-check-not=-fvisibility-from-dllstorageclass \
// RUN:     --implicit-check-not=-fvisibility-dllexport \
// RUN:     --implicit-check-not=-fvisibility-nodllstorageclass \
// RUN:     --implicit-check-not=-fvisibility-externs-dllimport \
// RUN:     --implicit-check-not=-fvisibility-externs-nodllstorageclass

// ALL:      "-fvisibility-from-dllstorageclass"
// ALL-SAME: "-fvisibility-dllexport=hidden"
// ALL-SAME: "-fvisibility-nodllstorageclass=protected"
// ALL-SAME: "-fvisibility-externs-dllimport=hidden"
// ALL-SAME: "-fvisibility-externs-nodllstorageclass=protected"
