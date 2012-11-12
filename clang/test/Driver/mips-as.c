// REQUIRES: mips-registered-target
//
// Check passing options to the assembler for MIPS targets.
//
// RUN: %clang -target mips-linux-gnu -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS32-EB-AS %s
// MIPS32-EB-AS: as{{(.exe)?}}" "-march" "mips32" "-mabi" "32" "-EB"
// MIPS32-EB-AS-NOT: "-KPIC"
//
// RUN: %clang -target mips-linux-gnu -### \
// RUN:   -no-integrated-as -fPIC -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS32-EB-PIC %s
// MIPS32-EB-PIC: as{{(.exe)?}}" "-march" "mips32" "-mabi" "32" "-EB"
// MIPS32-EB-PIC: "-KPIC"
//
// RUN: %clang -target mipsel-linux-gnu -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS32-EL-AS %s
// MIPS32-EL-AS: as{{(.exe)?}}" "-march" "mips32" "-mabi" "32" "-EL"
//
// RUN: %clang -target mips64-linux-gnu -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS64-EB-AS %s
// MIPS64-EB-AS: as{{(.exe)?}}" "-march" "mips64" "-mabi" "64" "-EB"
//
// RUN: %clang -target mips64el-linux-gnu -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS64-EL-AS %s
// MIPS64-EL-AS: as{{(.exe)?}}" "-march" "mips64" "-mabi" "64" "-EL"
//
// RUN: %clang -target mips-linux-gnu -mabi=eabi -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-EABI %s
// MIPS-EABI: as{{(.exe)?}}" "-march" "mips32" "-mabi" "eabi" "-EB"
//
// RUN: %clang -target mips64-linux-gnu -mabi=n32 -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-N32 %s
// MIPS-N32: as{{(.exe)?}}" "-march" "mips64" "-mabi" "n32" "-EB"
//
// RUN: %clang -target mips-linux-gnu -march=mips32r2 -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-32R2 %s
// MIPS-32R2: as{{(.exe)?}}" "-march" "mips32r2" "-mabi" "32" "-EB"
//
// RUN: %clang -target mips-linux-gnu -mips32 -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-ALIAS-32 %s
// MIPS-ALIAS-32: as{{(.exe)?}}" "-march" "mips32" "-mabi" "32" "-EB"
//
// RUN: %clang -target mips-linux-gnu -mips32r2 -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-ALIAS-32R2 %s
// MIPS-ALIAS-32R2: as{{(.exe)?}}" "-march" "mips32r2" "-mabi" "32" "-EB"
//
// RUN: %clang -target mips-linux-gnu -mips64 -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-ALIAS-64 %s
// MIPS-ALIAS-64: as{{(.exe)?}}" "-march" "mips64" "-mabi" "64" "-EB"
//
// RUN: %clang -target mips-linux-gnu -mips64r2 -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-ALIAS-64R2 %s
// MIPS-ALIAS-64R2: as{{(.exe)?}}" "-march" "mips64r2" "-mabi" "64" "-EB"
