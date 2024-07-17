/// Check PS5 specific interactions between visibility options.
/// Detailed testing of -fvisibility-from-dllstorageclass is covered elsewhere.

/// Check defaults.
// RUN: %clang -### -target x86_64-sie-ps5 -x cl -c -emit-llvm %s 2>&1 | \
// RUN:   FileCheck -check-prefixes=VDEFAULT,VGND_DEFAULT,DEFAULT %s --implicit-check-not=fvisibility --implicit-check-not=ftype-visibility --implicit-check-not=dllstorageclass
// VDEFAULT-DAG:     "-fvisibility=hidden"
// VGND_DEFAULT-DAG: "-fvisibility-global-new-delete=source"
// DEFAULT-DAG:      "-fvisibility-from-dllstorageclass"
// DEFAULT-DAG:      "-fvisibility-dllexport=protected"
// DEFAULT-DAG:      "-fvisibility-nodllstorageclass=keep"
// DEFAULT-DAG:      "-fvisibility-externs-dllimport=default"
// DEFAULT-DAG:      "-fvisibility-externs-nodllstorageclass=keep"

/// -fvisibility= specified explicitly.
// RUN: %clang -### -target x86_64-sie-ps5 -x cl -c -emit-llvm -fvisibility=protected %s 2>&1 | \
// RUN:   FileCheck -check-prefixes=VPROTECTED,VGND_DEFAULT,DEFAULT %s --implicit-check-not=fvisibility --implicit-check-not=ftype-visibility --implicit-check-not=dllstorageclass
// VPROTECTED-DAG: "-fvisibility=protected"

/// -fvisibility-ms-compat added explicitly.
// RUN: %clang -### -target x86_64-sie-ps5 -x cl -c -emit-llvm -fvisibility-ms-compat %s 2>&1 | \
// RUN:   FileCheck -check-prefixes=MSCOMPT,VGND_DEFAULT,DEFAULT %s --implicit-check-not=fvisibility --implicit-check-not=ftype-visibility --implicit-check-not=dllstorageclass
// MSCOMPT-DAG: "-fvisibility=hidden"
// MSCOMPT-DAG: "-ftype-visibility=default"

/// -fvisibility-from-dllstorageclass added explicitly.
// RUN: %clang -### -target x86_64-sie-ps5 -x cl -c -emit-llvm -fvisibility-from-dllstorageclass %s 2>&1 | \
// RUN:   FileCheck -check-prefixes=VDEFAULT,VGND_DEFAULT,DEFAULT %s --implicit-check-not=fvisibility --implicit-check-not=ftype-visibility --implicit-check-not=dllstorageclass

/// -fvisibility-from-dllstorageclass disabled explicitly.
// RUN: %clang -### -target x86_64-sie-ps5 -x cl -c -emit-llvm -fno-visibility-from-dllstorageclass %s 2>&1 | \
// RUN:   FileCheck -check-prefixes=VDEFAULT,VGND_DEFAULT,NOVISFROM %s --implicit-check-not=fvisibility --implicit-check-not=ftype-visibility --implicit-check-not=dllstorageclass
// NOVISFROM-NOT: "-fvisibility-from-dllstorageclass"
