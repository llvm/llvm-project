// Make sure SparcV9 does not use the integrated assembler by default.

// RUN: %clang -target sparcv9-linux -### -c %s 2>&1 \
// RUN: | FileCheck -check-prefix=NO-IAS %s

// RUN: %clang -target sparcv9-linux -fintegrated-as -### -c %s 2>&1 \
// RUN: | FileCheck -check-prefix=IAS %s

// RUN: %clang -target sparcv9-linux -fno-integrated-as -### -c %s 2>&1 \
// RUN: | FileCheck -check-prefix=NO-IAS %s

// IAS-NOT: "-no-integrated-as"
// NO-IAS: "-no-integrated-as"

// RUN: %clang -no-canonical-prefixes -target sparcv9--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC %s

// RUN: %clang -mcpu=v9 -no-canonical-prefixes -target sparcv9--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-V9 %s

// RUN: %clang -mcpu=ultrasparc -no-canonical-prefixes -target sparcv9--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-V9 %s

// RUN: %clang -mcpu=ultrasparc3 -no-canonical-prefixes -target sparcv9--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-V9 %s

// RUN: %clang -mcpu=niagara -no-canonical-prefixes -target sparcv9--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-V9B %s

// RUN: %clang -mcpu=niagara2 -no-canonical-prefixes -target sparcv9--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-V9B %s

// RUN: %clang -mcpu=niagara3 -no-canonical-prefixes -target sparcv9--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-V9D %s

// RUN: %clang -mcpu=niagara4 -no-canonical-prefixes -target sparcv9--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-V9D %s

// SPARC: as{{.*}}" "-64" "-Av9" "-o"
// SPARC-V9: as{{.*}}" "-64" "-Av9" "-o"
// SPARC-V9B: as{{.*}}" "-64" "-Av9b" "-o"
// SPARC-V9D: as{{.*}}" "-64" "-Av9d" "-o"

// RUN: not %clang -mcpu=v8 -no-canonical-prefixes -target sparcv9--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -c 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-V8 %s

// RUN: not %clang -mcpu=supersparc -no-canonical-prefixes -target sparcv9--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -c 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-V8 %s

// RUN: not %clang -mcpu=sparclite -no-canonical-prefixes -target sparcv9--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -c 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-V8 %s

// RUN: not %clang -mcpu=f934 -no-canonical-prefixes -target sparcv9--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -c 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-V8 %s

// RUN: not %clang -mcpu=hypersparc -no-canonical-prefixes -target sparcv9--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -c 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-V8 %s

// RUN: not %clang -mcpu=sparclite86x -no-canonical-prefixes -target sparcv9--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -c 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-V8 %s

// RUN: not %clang -mcpu=sparclet -no-canonical-prefixes -target sparcv9--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -c 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-V8 %s

// RUN: not %clang -mcpu=tsc701 -no-canonical-prefixes -target sparcv9--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -c 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-V8 %s

// SPARC-V8: error: unknown target CPU

int x;
