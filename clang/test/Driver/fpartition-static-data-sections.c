// RUN: %clang -### --target=x86_64 -fpartition-static-data-sections %s 2>&1 | FileCheck %s --check-prefixes=OPT
// RUN: %clang -### --target=aarch64 -fpartition-static-data-sections %s 2>&1 | FileCheck %s --check-prefixes=OPT

// RUN: not %clang -### --target=arm -fpartition-static-data-sections %s 2>&1 | FileCheck %s --check-prefixes=ERR

// RUN: %clang -### --target=x86_64 -fpartition-static-data-sections -fno-partition-static-data-sections %s 2>&1 | FileCheck %s --implicit-check-not="-fpartition-static-data-sections"

// RUN: %clang -### --target=x86_64-linux -flto -fpartition-static-data-sections %s 2>&1 | FileCheck %s --check-prefix=LTO
// RUN: %clang -### --target=x86_64-linux -flto -fpartition-static-data-sections -fno-partition-static-data-sections %s 2>&1 | FileCheck %s --implicit-check-not="-plugin-opt=-fpartition-static-data-sections"

// OPT: "-fpartition-static-data-sections"
// OPT: "-mllvm" "-memprof-annotate-static-data-prefix"

// ERR: error: unsupported option '-fpartition-static-data-sections' for target

// LTO: "-plugin-opt=-partition-static-data-sections"
