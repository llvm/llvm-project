// Check behaviour of -fvisibility-from-dllstorageclass options

// RUN: %clang -target x86_64-unknown-windows-itanium -fdeclspec \
// RUN:     -Werror -S -### %s 2>&1 | \
// RUN:   FileCheck %s \
// RUN:     --implicit-check-not=-fvisibility-from-dllstorageclass \
// RUN:     --implicit-check-not=-fvisibility-dllexport \
// RUN:     --implicit-check-not=-fvisibility-nodllstorageclass \
// RUN:     --implicit-check-not=-fvisibility-externs-dllimport \
// RUN:     --implicit-check-not=-fvisibility-externs-nodllstorageclass

// RUN: %clang -target x86_64-unknown-windows-itanium -fdeclspec \
// RUN:     -fvisibility-from-dllstorageclass \
// RUN:     -fno-visibility-from-dllstorageclass \
// RUN:     -Werror -S -### %s 2>&1 | \
// RUN:   FileCheck %s \
// RUN:     --implicit-check-not=-fvisibility-from-dllstorageclass \
// RUN:     --implicit-check-not=-fvisibility-dllexport \
// RUN:     --implicit-check-not=-fvisibility-nodllstorageclass \
// RUN:     --implicit-check-not=-fvisibility-externs-dllimport \
// RUN:     --implicit-check-not=-fvisibility-externs-nodllstorageclass

// RUN: %clang -target x86_64-unknown-windows-itanium -fdeclspec \
// RUN:     -fno-visibility-from-dllstorageclass \
// RUN:     -fvisibility-from-dllstorageclass \
// RUN:     -Werror -S -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=SET \
// RUN:     --implicit-check-not=-fvisibility-from-dllstorageclass \
// RUN:     --implicit-check-not=-fvisibility-dllexport \
// RUN:     --implicit-check-not=-fvisibility-nodllstorageclass \
// RUN:     --implicit-check-not=-fvisibility-externs-dllimport \
// RUN:     --implicit-check-not=-fvisibility-externs-nodllstorageclass

// RUN: %clang -target x86_64-unknown-windows-itanium -fdeclspec \
// RUN:     -fvisibility-dllexport=hidden \
// RUN:     -fvisibility-nodllstorageclass=protected \
// RUN:     -fvisibility-externs-dllimport=hidden \
// RUN:     -fvisibility-externs-nodllstorageclass=protected \
// RUN:     -S -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=UNUSED \
// RUN:     --implicit-check-not=-fvisibility-from-dllstorageclass \
// RUN:     --implicit-check-not=-fvisibility-dllexport \
// RUN:     --implicit-check-not=-fvisibility-nodllstorageclass \
// RUN:     --implicit-check-not=-fvisibility-externs-dllimport \
// RUN:     --implicit-check-not=-fvisibility-externs-nodllstorageclass \
// RUN:     --implicit-check-not=error: \
// RUN:     --implicit-check-not=warning:

// RUN: %clang -target x86_64-unknown-windows-itanium -fdeclspec \
// RUN:     -fno-visibility-from-dllstorageclass \
// RUN:     -fvisibility-dllexport=hidden \
// RUN:     -fvisibility-nodllstorageclass=protected \
// RUN:     -fvisibility-externs-dllimport=hidden \
// RUN:     -fvisibility-externs-nodllstorageclass=protected \
// RUN:     -S -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=UNUSED \
// RUN:     --implicit-check-not=-fvisibility-from-dllstorageclass \
// RUN:     --implicit-check-not=-fvisibility-dllexport \
// RUN:     --implicit-check-not=-fvisibility-nodllstorageclass \
// RUN:     --implicit-check-not=-fvisibility-externs-dllimport \
// RUN:     --implicit-check-not=-fvisibility-externs-nodllstorageclass \
// RUN:     --implicit-check-not=error: \
// RUN:     --implicit-check-not=warning:

// UNUSED:      warning: argument unused during compilation: '-fvisibility-dllexport=hidden'
// UNUSED-NEXT: warning: argument unused during compilation: '-fvisibility-nodllstorageclass=protected'
// UNUSED-NEXT: warning: argument unused during compilation: '-fvisibility-externs-dllimport=hidden'
// UNUSED-NEXT: warning: argument unused during compilation: '-fvisibility-externs-nodllstorageclass=protected'

// RUN: %clang -target x86_64-unknown-windows-itanium -fdeclspec \
// RUN:     -fvisibility-from-dllstorageclass \
// RUN:     -fvisibility-dllexport=default \
// RUN:     -fvisibility-dllexport=hidden \
// RUN:     -fvisibility-nodllstorageclass=default \
// RUN:     -fvisibility-nodllstorageclass=protected \
// RUN:     -fvisibility-externs-dllimport=default \
// RUN:     -fvisibility-externs-dllimport=hidden \
// RUN:     -fvisibility-externs-nodllstorageclass=default \
// RUN:     -fvisibility-externs-nodllstorageclass=protected \
// RUN:     -Werror -S -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=SET,ALL

// SET:      "-fvisibility-from-dllstorageclass"
// ALL-SAME: "-fvisibility-dllexport=hidden"
// ALL-SAME: "-fvisibility-nodllstorageclass=protected"
// ALL-SAME: "-fvisibility-externs-dllimport=hidden"
// ALL-SAME: "-fvisibility-externs-nodllstorageclass=protected"
