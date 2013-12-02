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
// RUN:   | FileCheck -check-prefix=MIPS32-DEF-EL-AS %s
// MIPS32-DEF-EL-AS: as{{(.exe)?}}" "-march" "mips32" "-mabi" "32" "-EL"
//
// RUN: %clang -target mips64-linux-gnu -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS64-EB-AS %s
// MIPS64-EB-AS: as{{(.exe)?}}" "-march" "mips64" "-mabi" "64" "-EB"
//
// RUN: %clang -target mips64el-linux-gnu -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS64-DEF-EL-AS %s
// MIPS64-DEF-EL-AS: as{{(.exe)?}}" "-march" "mips64" "-mabi" "64" "-EL"
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
// RUN: %clang -target mipsel-linux-gnu -mabi=32 -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS32-EL-AS %s
// MIPS32-EL-AS: as{{(.exe)?}}" "-march" "mips32" "-mabi" "32" "-EL"
//
// RUN: %clang -target mips64el-linux-gnu -mabi=64 -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS64-EL-AS %s
// MIPS64-EL-AS: as{{(.exe)?}}" "-march" "mips64" "-mabi" "64" "-EL"
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
//
// RUN: %clang -target mips-linux-gnu -mno-mips16 -mips16 -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-16 %s
// MIPS-16: as{{(.exe)?}}" "-march" "mips32" "-mabi" "32" "-EB" "-mips16"
//
// RUN: %clang -target mips-linux-gnu -mips16 -mno-mips16 -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-N16 %s
// MIPS-N16: as{{(.exe)?}}"
// MIPS-N16-NOT: "-mips16"
//
// RUN: %clang -target mips-linux-gnu -mno-micromips -mmicromips -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-MICRO %s
// MIPS-MICRO: as{{(.exe)?}}" "-march" "mips32" "-mabi" "32" "-EB" "-mmicromips"
//
// RUN: %clang -target mips-linux-gnu -mmicromips -mno-micromips -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-NMICRO %s
// MIPS-NMICRO: as{{(.exe)?}}"
// MIPS-NMICRO-NOT: "-mmicromips"
//
// RUN: %clang -target mips-linux-gnu -mno-dsp -mdsp -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-DSP %s
// MIPS-DSP: as{{(.exe)?}}" "-march" "mips32" "-mabi" "32" "-EB" "-mdsp"
//
// RUN: %clang -target mips-linux-gnu -mdsp -mno-dsp -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-NDSP %s
// MIPS-NDSP: as{{(.exe)?}}"
// MIPS-NDSP-NOT: "-mdsp"
//
// RUN: %clang -target mips-linux-gnu -mno-dspr2 -mdspr2 -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-DSPR2 %s
// MIPS-DSPR2: as{{(.exe)?}}" "-march" "mips32" "-mabi" "32" "-EB" "-mdspr2"
//
// RUN: %clang -target mips-linux-gnu -mdspr2 -mno-dspr2 -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-NDSPR2 %s
// MIPS-NDSPR2: as{{(.exe)?}}"
// MIPS-NDSPR2-NOT: "-mdspr2"
//
// RUN: %clang -target mips-linux-gnu -mnan=legacy -mnan=2008 -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-NAN2008 %s
// MIPS-NAN2008: as{{(.exe)?}}" "-march" "mips32" "-mabi" "32" "-EB" "-mnan=2008"
//
// RUN: %clang -target mips-linux-gnu -mnan=2008 -mnan=legacy -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-NAN-LEGACY %s
// MIPS-NAN-LEGACY: as{{(.exe)?}}"
// MIPS-NAN_LEGACY-NOT: "-mnan={{.*}}"
//
// RUN: %clang -target mips-linux-gnu -mfp64 -mfp32 -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-MFP32 %s
// MIPS-MFP32: as{{(.exe)?}}" "-march" "mips32" "-mabi" "32" "-EB" "-mfp32"
//
// RUN: %clang -target mips-linux-gnu -mfp32 -mfp64 -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-MFP64 %s
// MIPS-MFP64: as{{(.exe)?}}" "-march" "mips32" "-mabi" "32" "-EB" "-mfp64"
//
// RUN: %clang -target mips-linux-gnu -mno-msa -mmsa -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-MSA %s
// MIPS-MSA: as{{(.exe)?}}" "-march" "mips32" "-mabi" "32" "-EB" "-mmsa"
//
// RUN: %clang -target mips-linux-gnu -mmsa -mno-msa -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-NMSA %s
// MIPS-NMSA: as{{(.exe)?}}"
// MIPS-NMSA-NOT: "-mmsa"
