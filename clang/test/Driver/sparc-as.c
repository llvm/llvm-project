// Make sure Sparc does use the integrated assembler by default.

// RUN: %clang --target=sparc-linux -### -c %s 2>&1 \
// RUN: | FileCheck -check-prefix=IAS %s

// RUN: %clang --target=sparc-linux -fintegrated-as -### -c %s 2>&1 \
// RUN: | FileCheck -check-prefix=IAS %s

// RUN: %clang --target=sparc-linux -fno-integrated-as -### -c %s 2>&1 \
// RUN: | FileCheck -check-prefix=NO-IAS %s

// IAS-NOT: "-no-integrated-as"
// NO-IAS: "-no-integrated-as"

// RUN: %clang --target=sparc--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC %s

// RUN: %clang -mcpu=v8 --target=sparc--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-V8 %s

// RUN: %clang -mcpu=supersparc --target=sparc--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-V8 %s

// RUN: %clang -mcpu=sparclite --target=sparc--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-SPARCLITE %s

// RUN: %clang -mcpu=f934 --target=sparc--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-SPARCLITE %s

// RUN: %clang -mcpu=hypersparc --target=sparc--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-V8 %s

// RUN: %clang -mcpu=sparclite86x --target=sparc--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-SPARCLITE %s

// RUN: %clang -mcpu=sparclet --target=sparc--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-SPARCLET %s

// RUN: %clang -mcpu=tsc701 --target=sparc--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-SPARCLET %s

// RUN: %clang -mcpu=v9 --target=sparc--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-V8PLUS %s

// RUN: %clang -mcpu=ultrasparc --target=sparc--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-V8PLUS %s

// RUN: %clang -mcpu=ultrasparc3 --target=sparc--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-V8PLUS %s

// RUN: %clang -mcpu=niagara --target=sparc--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-V8PLUSB %s

// RUN: %clang -mcpu=niagara2 --target=sparc--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-V8PLUSB %s

// RUN: %clang -mcpu=niagara3 --target=sparc--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-V8PLUSD %s

// RUN: %clang -mcpu=niagara4 --target=sparc--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-V8PLUSD %s

// RUN: %clang -mcpu=ma2100 --target=sparc \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-LEON %s

// RUN: %clang -mcpu=ma2150 --target=sparc \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-LEON %s

// RUN: %clang -mcpu=ma2155 --target=sparc \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-LEON %s

// RUN: %clang -mcpu=ma2450 --target=sparc \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-LEON %s

// RUN: %clang -mcpu=ma2455 --target=sparc \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-LEON %s

// RUN: %clang -mcpu=ma2x5x --target=sparc \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-LEON %s

// RUN: %clang -mcpu=ma2080 --target=sparc \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-LEON %s

// RUN: %clang -mcpu=ma2085 --target=sparc \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-LEON %s

// RUN: %clang -mcpu=ma2480 --target=sparc \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-LEON %s

// RUN: %clang -mcpu=ma2485 --target=sparc \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-LEON %s

// RUN: %clang -mcpu=ma2x8x --target=sparc \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-LEON %s

// RUN: %clang -mcpu=myriad2 --target=sparc \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-LEON %s

// RUN: %clang -mcpu=myriad2.1 --target=sparc \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-LEON %s

// RUN: %clang -mcpu=myriad2.2 --target=sparc \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-LEON %s

// RUN: %clang -mcpu=myriad2.3 --target=sparc \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-LEON %s

// RUN: %clang -mcpu=leon2 --target=sparc \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-V8 %s

// RUN: %clang -mcpu=at697e --target=sparc \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-V8 %s

// RUN: %clang -mcpu=at697f --target=sparc \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-V8 %s

// RUN: %clang -mcpu=leon3 --target=sparc \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-LEON %s

// RUN: %clang -mcpu=ut699 --target=sparc \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-V8 %s

// RUN: %clang -mcpu=gr712rc --target=sparc \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-LEON %s

// RUN: %clang -mcpu=leon4 --target=sparc \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-LEON %s

// RUN: %clang -mcpu=gr740 --target=sparc \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-LEON %s

// SPARC: as{{.*}}" "-32" "-Av8" "-o"
// SPARC-V8: as{{.*}}" "-32" "-Av8" "-o"
// SPARC-LEON: as{{.*}}" "-32" "-Aleon" "-o"
// SPARC-SPARCLITE: as{{.*}}" "-32" "-Asparclite" "-o"
// SPARC-SPARCLET: as{{.*}}" "-32" "-Asparclet" "-o"
// SPARC-V8PLUS: as{{.*}}" "-32" "-Av8plus" "-o"
// SPARC-V8PLUSB: as{{.*}}" "-32" "-Av8plusb" "-o"
// SPARC-V8PLUSD: as{{.*}}" "-32" "-Av8plusd" "-o"
