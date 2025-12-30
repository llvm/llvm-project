/// Check PS4 specific interactions between visibility options.
/// Detailed testing of -fvisibility-from-dllstorageclass is covered elsewhere.

/// Check defaults.
// RUN: %clang -### -target x86_64-scei-ps4 -x cl -c -emit-llvm %s 2>&1 | \
// RUN:   FileCheck -check-prefix=DEFAULT %s --implicit-check-not=fvisibility --implicit-check-not=ftype-visibility --implicit-check-not=dllstorageclass
// DEFAULT-DAG: "-fvisibility-from-dllstorageclass"
// DEFAULT-DAG: "-fvisibility-dllexport=protected"
// DEFAULT-DAG: "-fvisibility-nodllstorageclass=hidden"
// DEFAULT-DAG: "-fvisibility-externs-dllimport=default"
// DEFAULT-DAG: "-fvisibility-externs-nodllstorageclass=default"

/// Check that -fvisibility-from-dllstorageclass is added in the presence of -fvisibility=.
// RUN: %clang -### -target x86_64-scei-ps4 -x cl -c -emit-llvm -fvisibility=default  %s 2>&1 | \
// RUN:   FileCheck -check-prefixes=DEFAULT,VISEQUALS %s --implicit-check-not=fvisibility --implicit-check-not=ftype-visibility --implicit-check-not=dllstorageclass
// VISEQUALS-DAG: "-fvisibility=default"

/// Check that -fvisibility-from-dllstorageclass is added in the presence of -fvisibility-ms-compat.
// RUN: %clang -### -target x86_64-scei-ps4 -x cl -c -emit-llvm -fvisibility-ms-compat %s 2>&1 | \
// RUN:   FileCheck -check-prefixes=DEFAULT,MSCOMPT %s --implicit-check-not=fvisibility --implicit-check-not=ftype-visibility --implicit-check-not=dllstorageclass
// MSCOMPT-DAG: "-fvisibility=hidden"
// MSCOMPT-DAG: "-ftype-visibility=default"

/// -fvisibility-from-dllstorageclass added explicitly.
// RUN: %clang -### -target x86_64-scei-ps4 -x cl -c -emit-llvm -fvisibility-from-dllstorageclass %s 2>&1 | \
// RUN:   FileCheck -check-prefixes=DEFAULT %s --implicit-check-not=fvisibility --implicit-check-not=ftype-visibility --implicit-check-not=dllstorageclass

/// -fvisibility-from-dllstorageclass disabled explicitly.
// RUN: %clang -### -target x86_64-scei-ps4 -x cl -c -emit-llvm -fno-visibility-from-dllstorageclass %s 2>&1 | \
// RUN:   FileCheck -check-prefixes=NOVISFROM %s --implicit-check-not=fvisibility --implicit-check-not=ftype-visibility --implicit-check-not=dllstorageclass
// NOVISFROM-NOT: "-fvisibility-from-dllstorageclass"

