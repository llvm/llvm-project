// Check passing options to the assembler for MIPS targets.
//
// RUN: %clang --target=mips-unknown-freebsd -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS32-EB-AS %s
// MIPS32-EB-AS: as{{(.exe)?}}" "-march" "mips2" "-mabi" "32" "-EB"
// MIPS32-EB-AS-NOT: "-KPIC"
//
// RUN: %clang --target=mips-unknown-freebsd -### \
// RUN:   -no-integrated-as -fPIC -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS32-EB-PIC %s
// MIPS32-EB-PIC: as{{(.exe)?}}" "-march" "mips2" "-mabi" "32" "-EB"
// MIPS32-EB-PIC: "-KPIC"
//
// RUN: %clang --target=mips-unknown-freebsd -### \
// RUN:   -no-integrated-as -fpic -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS32-EB-PIC-SMALL %s
// MIPS32-EB-PIC-SMALL: as{{(.exe)?}}" "-march" "mips2" "-mabi" "32" "-EB"
// MIPS32-EB-PIC-SMALL: "-KPIC"
//
// RUN: %clang --target=mips-unknown-freebsd -### \
// RUN:   -no-integrated-as -fPIE -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS32-EB-PIE %s
// MIPS32-EB-PIE: as{{(.exe)?}}" "-march" "mips2" "-mabi" "32" "-EB"
// MIPS32-EB-PIE: "-KPIC"
//
// RUN: %clang --target=mips-unknown-freebsd -### \
// RUN:   -no-integrated-as -fpie -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS32-EB-PIE-SMALL %s
// MIPS32-EB-PIE-SMALL: as{{(.exe)?}}" "-march" "mips2" "-mabi" "32" "-EB"
// MIPS32-EB-PIE-SMALL: "-KPIC"
//
// RUN: %clang --target=mipsel-unknown-freebsd -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS32-DEF-EL-AS %s
// MIPS32-DEF-EL-AS: as{{(.exe)?}}" "-march" "mips2" "-mabi" "32" "-EL"
//
// RUN: %clang --target=mips64-unknown-freebsd -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS64-EB-AS %s
// MIPS64-EB-AS: as{{(.exe)?}}" "-march" "mips3" "-mabi" "64" "-EB"
//
// RUN: %clang --target=mips64el-unknown-freebsd -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS64-DEF-EL-AS %s
// MIPS64-DEF-EL-AS: as{{(.exe)?}}" "-march" "mips3" "-mabi" "64" "-EL"
//
// RUN: %clang --target=mips64-unknown-freebsd -mabi=n32 -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-N32 %s
// MIPS-N32: as{{(.exe)?}}" "-march" "mips3" "-mabi" "n32" "-EB"
//
// RUN: %clang --target=mipsel-unknown-freebsd -mabi=32 -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS32-EL-AS %s
// MIPS32-EL-AS: as{{(.exe)?}}" "-march" "mips2" "-mabi" "32" "-EL"
//
// RUN: %clang --target=mips64el-unknown-freebsd -mabi=64 -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS64-EL-AS %s
// MIPS64-EL-AS: as{{(.exe)?}}" "-march" "mips3" "-mabi" "64" "-EL"
//
// RUN: %clang --target=mips-linux-freebsd -march=mips32r2 -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-32R2 %s
// MIPS-32R2: as{{(.exe)?}}" "-march" "mips32r2" "-mabi" "32" "-EB"
//
// RUN: %clang --target=mips-unknown-freebsd -mips32 -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-ALIAS-32 %s
// MIPS-ALIAS-32: as{{(.exe)?}}" "-march" "mips32" "-mabi" "32" "-EB"
//
// RUN: %clang --target=mips-unknown-freebsd -mips32r2 -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-ALIAS-32R2 %s
// MIPS-ALIAS-32R2: as{{(.exe)?}}" "-march" "mips32r2" "-mabi" "32" "-EB"
//
// RUN: %clang --target=mips64-unknown-freebsd -mips64 -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-ALIAS-64 %s
// MIPS-ALIAS-64: as{{(.exe)?}}" "-march" "mips64" "-mabi" "64" "-EB"
//
// RUN: %clang --target=mips64-unknown-freebsd -mips64r2 -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-ALIAS-64R2 %s
// MIPS-ALIAS-64R2: as{{(.exe)?}}" "-march" "mips64r2" "-mabi" "64" "-EB"
//
// RUN: %clang --target=mips-unknown-freebsd -### \
// RUN:   -no-integrated-as -G0 -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS32-EB-AS-G0 %s
// MIPS32-EB-AS-G0: as{{(.exe)?}}" "-march" "mips2" "-mabi" "32" "-EB" "-G0"
// MIPS32-EB-AS-G0-NOT: "-KPIC"
