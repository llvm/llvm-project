// Check behaviour of -fvisibility-from-dllstorageclass options for PS4/PS5.

// DEFINE: %{triple} =
// DEFINE: %{run} = \
// DEFINE: %clang -### -target %{triple} %s -Werror -o - 2>&1 | \
// DEFINE:   FileCheck %s --check-prefix=DEFAULTS \
// DEFINE:     --implicit-check-not=-fvisibility-from-dllstorageclass \
// DEFINE:     --implicit-check-not=-fvisibility-dllexport \
// DEFINE:     --implicit-check-not=-fvisibility-nodllstorageclass \
// DEFINE:     --implicit-check-not=-fvisibility-externs-dllimport \
// DEFINE:     --implicit-check-not=-fvisibility-externs-nodllstorageclass
// REDEFINE: %{triple} = x86_64-scei-ps4
// RUN: %{run}
// REDEFINE: %{triple} = x86_64-sie-ps5
// RUN: %{run}
//
// REDEFINE: %{run} = \
// REDEFINE: %clang -### -target %{triple} \
// REDEFINE:     -fno-visibility-from-dllstorageclass \
// REDEFINE:     -fvisibility-from-dllstorageclass \
// REDEFINE:     -Werror \
// REDEFINE:     %s -o - 2>&1 | \
// REDEFINE:   FileCheck %s --check-prefix=DEFAULTS \
// REDEFINE:     --implicit-check-not=-fvisibility-from-dllstorageclass \
// REDEFINE:     --implicit-check-not=-fvisibility-dllexport \
// REDEFINE:     --implicit-check-not=-fvisibility-nodllstorageclass \
// REDEFINE:     --implicit-check-not=-fvisibility-externs-dllimport \
// REDEFINE:     --implicit-check-not=-fvisibility-externs-nodllstorageclass
// REDEFINE: %{triple} = x86_64-scei-ps4
// RUN: %{run}
// REDEFINE: %{triple} = x86_64-sie-ps5
// RUN: %{run}

// DEFAULTS:      "-fvisibility-from-dllstorageclass"
// DEFAULTS-SAME: "-fvisibility-dllexport=protected"
// DEFAULTS-SAME: "-fvisibility-nodllstorageclass=hidden"
// DEFAULTS-SAME: "-fvisibility-externs-dllimport=default"
// DEFAULTS-SAME: "-fvisibility-externs-nodllstorageclass=default"

// REDEFINE: %{run} = \
// REDEFINE: %clang -### -target x86_64-scei-ps4 \
// REDEFINE:     -fvisibility-from-dllstorageclass \
// REDEFINE:     -fvisibility-dllexport=hidden \
// REDEFINE:     -fvisibility-nodllstorageclass=protected \
// REDEFINE:     -fvisibility-externs-dllimport=hidden \
// REDEFINE:     -fvisibility-externs-nodllstorageclass=protected \
// REDEFINE:     -fno-visibility-from-dllstorageclass \
// REDEFINE:     %s -o - 2>&1 | \
// REDEFINE:   FileCheck %s -check-prefix=UNUSED \
// REDEFINE:     --implicit-check-not=-fvisibility-from-dllstorageclass \
// REDEFINE:     --implicit-check-not=-fvisibility-dllexport \
// REDEFINE:     --implicit-check-not=-fvisibility-nodllstorageclass \
// REDEFINE:     --implicit-check-not=-fvisibility-externs-dllimport \
// REDEFINE:     --implicit-check-not=-fvisibility-externs-nodllstorageclass \
// REDEFINE:     --implicit-check-not=warning:
// REDEFINE: %{triple} = x86_64-scei-ps4
// RUN: %{run}
// REDEFINE: %{triple} = x86_64-sie-ps5
// RUN: %{run}

// UNUSED:      warning: argument unused during compilation: '-fvisibility-dllexport=hidden'
// UNUSED-NEXT: warning: argument unused during compilation: '-fvisibility-nodllstorageclass=protected'
// UNUSED-NEXT: warning: argument unused during compilation: '-fvisibility-externs-dllimport=hidden'
// UNUSED-NEXT: warning: argument unused during compilation: '-fvisibility-externs-nodllstorageclass=protected'

// REDEFINE: %{run} = \
// REDEFINE: %clang -### -target x86_64-scei-ps4 \
// REDEFINE:     -fvisibility-nodllstorageclass=protected \
// REDEFINE:     -fvisibility-externs-dllimport=hidden \
// REDEFINE:     -Werror \
// REDEFINE:     %s -o - 2>&1 | \
// REDEFINE:   FileCheck %s -check-prefix=SOME \
// REDEFINE:     --implicit-check-not=-fvisibility-from-dllstorageclass \
// REDEFINE:     --implicit-check-not=-fvisibility-dllexport \
// REDEFINE:     --implicit-check-not=-fvisibility-nodllstorageclass \
// REDEFINE:     --implicit-check-not=-fvisibility-externs-dllimport \
// REDEFINE:     --implicit-check-not=-fvisibility-externs-nodllstorageclass
// REDEFINE: %{triple} = x86_64-scei-ps4
// RUN: %{run}
// REDEFINE: %{triple} = x86_64-sie-ps5
// RUN: %{run}

// REDEFINE: %{run} = \
// REDEFINE: %clang -### -target x86_64-scei-ps4 \
// REDEFINE:     -fvisibility-from-dllstorageclass \
// REDEFINE:     -fvisibility-nodllstorageclass=protected \
// REDEFINE:     -fvisibility-externs-dllimport=hidden \
// REDEFINE:     -Werror \
// REDEFINE:     %s -o - 2>&1 | \
// REDEFINE:   FileCheck %s -check-prefix=SOME \
// REDEFINE:     --implicit-check-not=-fvisibility-from-dllstorageclass \
// REDEFINE:     --implicit-check-not=-fvisibility-dllexport \
// REDEFINE:     --implicit-check-not=-fvisibility-nodllstorageclass \
// REDEFINE:     --implicit-check-not=-fvisibility-externs-dllimport \
// REDEFINE:     --implicit-check-not=-fvisibility-externs-nodllstorageclass
// REDEFINE: %{triple} = x86_64-scei-ps4
// RUN: %{run}
// REDEFINE: %{triple} = x86_64-sie-ps5
// RUN: %{run}

// SOME:      "-fvisibility-from-dllstorageclass"
// SOME-SAME: "-fvisibility-dllexport=protected"
// SOME-SAME: "-fvisibility-nodllstorageclass=protected"
// SOME-SAME: "-fvisibility-externs-dllimport=hidden"
// SOME-SAME: "-fvisibility-externs-nodllstorageclass=default"

// REDEFINE: %{run} = \
// REDEFINE: %clang -### -target x86_64-scei-ps4 \
// REDEFINE:     -fvisibility-dllexport=default \
// REDEFINE:     -fvisibility-dllexport=hidden \
// REDEFINE:     -fvisibility-nodllstorageclass=default \
// REDEFINE:     -fvisibility-nodllstorageclass=protected \
// REDEFINE:     -fvisibility-externs-dllimport=default \
// REDEFINE:     -fvisibility-externs-dllimport=hidden \
// REDEFINE:     -fvisibility-externs-nodllstorageclass=default \
// REDEFINE:     -fvisibility-externs-nodllstorageclass=protected \
// REDEFINE:     -Werror \
// REDEFINE:     %s -o - 2>&1 | \
// REDEFINE:   FileCheck %s -check-prefix=ALL \
// REDEFINE:     --implicit-check-not=-fvisibility-from-dllstorageclass \
// REDEFINE:     --implicit-check-not=-fvisibility-dllexport \
// REDEFINE:     --implicit-check-not=-fvisibility-nodllstorageclass \
// REDEFINE:     --implicit-check-not=-fvisibility-externs-dllimport \
// REDEFINE:     --implicit-check-not=-fvisibility-externs-nodllstorageclass
// REDEFINE: %{triple} = x86_64-scei-ps4
// RUN: %{run}
// REDEFINE: %{triple} = x86_64-sie-ps5
// RUN: %{run}

// REDEFINE: %{run} = \
// REDEFINE: %clang -### -target x86_64-scei-ps4 \
// REDEFINE:     -fvisibility-from-dllstorageclass \
// REDEFINE:     -fvisibility-dllexport=default \
// REDEFINE:     -fvisibility-dllexport=hidden \
// REDEFINE:     -fvisibility-nodllstorageclass=default \
// REDEFINE:     -fvisibility-nodllstorageclass=protected \
// REDEFINE:     -fvisibility-externs-dllimport=default \
// REDEFINE:     -fvisibility-externs-dllimport=hidden \
// REDEFINE:     -fvisibility-externs-nodllstorageclass=default \
// REDEFINE:     -fvisibility-externs-nodllstorageclass=protected \
// REDEFINE:     -Werror \
// REDEFINE:     %s -o - 2>&1 | \
// REDEFINE:   FileCheck %s -check-prefix=ALL \
// REDEFINE:     --implicit-check-not=-fvisibility-from-dllstorageclass \
// REDEFINE:     --implicit-check-not=-fvisibility-dllexport \
// REDEFINE:     --implicit-check-not=-fvisibility-nodllstorageclass \
// REDEFINE:     --implicit-check-not=-fvisibility-externs-dllimport \
// REDEFINE:     --implicit-check-not=-fvisibility-externs-nodllstorageclass
// REDEFINE: %{triple} = x86_64-scei-ps4
// RUN: %{run}
// REDEFINE: %{triple} = x86_64-sie-ps5
// RUN: %{run}

// ALL:      "-fvisibility-from-dllstorageclass"
// ALL-SAME: "-fvisibility-dllexport=hidden"
// ALL-SAME: "-fvisibility-nodllstorageclass=protected"
// ALL-SAME: "-fvisibility-externs-dllimport=hidden"
// ALL-SAME: "-fvisibility-externs-nodllstorageclass=protected"
