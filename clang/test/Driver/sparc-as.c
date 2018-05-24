// Make sure Sparc does not use the integrated assembler by default.

// RUN: %clang -target sparc-linux -### -c %s 2>&1 \
// RUN: | FileCheck -check-prefix=NO-IAS %s

// RUN: %clang -target sparc-linux -fintegrated-as -### -c %s 2>&1 \
// RUN: | FileCheck -check-prefix=IAS %s

// RUN: %clang -target sparc-linux -fno-integrated-as -### -c %s 2>&1 \
// RUN: | FileCheck -check-prefix=NO-IAS %s

// IAS-NOT: "-no-integrated-as"
// NO-IAS: "-no-integrated-as"

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

// RUN: %clang -mcpu=ma2100 -no-canonical-prefixes -target sparc \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-LEON %s

// RUN: %clang -mcpu=ma2150 -no-canonical-prefixes -target sparc \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-LEON %s

// RUN: %clang -mcpu=ma2155 -no-canonical-prefixes -target sparc \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-LEON %s

// RUN: %clang -mcpu=ma2450 -no-canonical-prefixes -target sparc \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-LEON %s

// RUN: %clang -mcpu=ma2455 -no-canonical-prefixes -target sparc \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-LEON %s

// RUN: %clang -mcpu=ma2x5x -no-canonical-prefixes -target sparc \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-LEON %s

// RUN: %clang -mcpu=ma2080 -no-canonical-prefixes -target sparc \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-LEON %s

// RUN: %clang -mcpu=ma2085 -no-canonical-prefixes -target sparc \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-LEON %s

// RUN: %clang -mcpu=ma2480 -no-canonical-prefixes -target sparc \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-LEON %s

// RUN: %clang -mcpu=ma2485 -no-canonical-prefixes -target sparc \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-LEON %s

// RUN: %clang -mcpu=ma2x8x -no-canonical-prefixes -target sparc \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-LEON %s

// RUN: %clang -mcpu=myriad2 -no-canonical-prefixes -target sparc \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-LEON %s

// RUN: %clang -mcpu=myriad2.1 -no-canonical-prefixes -target sparc \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-LEON %s

// RUN: %clang -mcpu=myriad2.2 -no-canonical-prefixes -target sparc \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-LEON %s

// RUN: %clang -mcpu=myriad2.3 -no-canonical-prefixes -target sparc \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-LEON %s

// RUN: %clang -mcpu=leon2 -no-canonical-prefixes -target sparc \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-V8 %s

// RUN: %clang -mcpu=at697e -no-canonical-prefixes -target sparc \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-V8 %s

// RUN: %clang -mcpu=at697f -no-canonical-prefixes -target sparc \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-V8 %s

// RUN: %clang -mcpu=leon3 -no-canonical-prefixes -target sparc \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-LEON %s

// RUN: %clang -mcpu=ut699 -no-canonical-prefixes -target sparc \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-V8 %s

// RUN: %clang -mcpu=gr712rc -no-canonical-prefixes -target sparc \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-LEON %s

// RUN: %clang -mcpu=leon4 -no-canonical-prefixes -target sparc \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-LEON %s

// RUN: %clang -mcpu=gr740 -no-canonical-prefixes -target sparc \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-LEON %s

// SPARC: as{{.*}}" "-32" "-Av8" "-o"
// SPARC-V8: as{{.*}}" "-32" "-Av8" "-o"
// SPARC-LEON: as{{.*}}" "-32" "-Aleon" "-o"
// SPARC-SPARCLITE: as{{.*}}" "-32" "-Asparclite" "-o"
// SPARC-SPARCLET: as{{.*}}" "-32" "-Asparclet" "-o"
// SPARC-V8PLUS: as{{.*}}" "-32" "-Av8plus" "-o"
// SPARC-V8PLUSB: as{{.*}}" "-32" "-Av8plusb" "-o"
// SPARC-V8PLUSD: as{{.*}}" "-32" "-Av8plusd" "-o"
