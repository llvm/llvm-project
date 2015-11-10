// RUN: %clang -no-canonical-prefixes -target sparc--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC %s

// RUN: %clang -mcpu=v8 -no-canonical-prefixes -target sparc--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-V8 %s

// RUN: %clang -mcpu=supersparc -no-canonical-prefixes -target sparc--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-V8 %s

// RUN: %clang -mcpu=sparclite -no-canonical-prefixes -target sparc--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-SPARCLITE %s

// RUN: %clang -mcpu=f934 -no-canonical-prefixes -target sparc--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-SPARCLITE %s

// RUN: %clang -mcpu=hypersparc -no-canonical-prefixes -target sparc--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-V8 %s

// RUN: %clang -mcpu=sparclite86x -no-canonical-prefixes -target sparc--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-SPARCLITE %s

// RUN: %clang -mcpu=sparclet -no-canonical-prefixes -target sparc--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-SPARCLET %s

// RUN: %clang -mcpu=tsc701 -no-canonical-prefixes -target sparc--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-SPARCLET %s

// RUN: %clang -mcpu=v9 -no-canonical-prefixes -target sparc--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-V8PLUS %s

// RUN: %clang -mcpu=ultrasparc -no-canonical-prefixes -target sparc--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-V8PLUS %s

// RUN: %clang -mcpu=ultrasparc3 -no-canonical-prefixes -target sparc--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-V8PLUS %s

// RUN: %clang -mcpu=niagara -no-canonical-prefixes -target sparc--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-V8PLUSB %s

// RUN: %clang -mcpu=niagara2 -no-canonical-prefixes -target sparc--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-V8PLUSB %s

// RUN: %clang -mcpu=niagara3 -no-canonical-prefixes -target sparc--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-V8PLUSD %s

// RUN: %clang -mcpu=niagara4 -no-canonical-prefixes -target sparc--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-V8PLUSD %s

// SPARC: as{{.*}}" "-32" "-Av8" "-o"
// SPARC-V8: as{{.*}}" "-32" "-Av8" "-o"
// SPARC-SPARCLITE: as{{.*}}" "-32" "-Asparclite" "-o"
// SPARC-SPARCLET: as{{.*}}" "-32" "-Asparclet" "-o"
// SPARC-V8PLUS: as{{.*}}" "-32" "-Av8plus" "-o"
// SPARC-V8PLUSB: as{{.*}}" "-32" "-Av8plusb" "-o"
// SPARC-V8PLUSD: as{{.*}}" "-32" "-Av8plusd" "-o"
