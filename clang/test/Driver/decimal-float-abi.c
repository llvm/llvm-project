// DEFINE: %{common_opts_x86} = -###  \
// DEFINE: -fexperimental-decimal-floating-point

// X86
// RUN: %clang %{common_opts_x86} --target=x86_64 \
// RUN: -mdecimal-float-abi=libgcc:bid %s 2>&1 | FileCheck -check-prefix=BID %s

// RUN: not %clang %{common_opts_x86} --target=x86_64 \
// RUN: -mdecimal-float-abi=libgcc:dpd %s 2>&1 \
// RUN: | FileCheck -check-prefix=ERROR_DPD %s

// RUN: not %clang %{common_opts_x86} --target=x86_64 \
// RUN: -mdecimal-float-abi=hard %s 2>&1 \
// RUN: | FileCheck -check-prefix=ERROR_HARD %s

// RUN: not %clang %{common_opts_x86} --target=mips64-unknown-linux \
// RUN: -mdecimal-float-abi=hard %s 2>&1 \
// RUN: | FileCheck -check-prefix=ERROR_MIPS_TRGT %s

// BID: "-mdecimal-float-abi=libgcc:bid" {{.*}} "-target-feature" "+bid-encoding"

// ERROR_DPD: unsupported decimal floating-point encoding 'libgcc:dpd' for target 'X86'
// ERROR_HARD: unsupported decimal floating-point encoding 'hard' for target 'X86'
// ERROR_MIPS_TRGT: unsupported option '-mdecimal-float-abi=' for target 'mips64-unknown-linux'

// PPC
// S390
// ARM
