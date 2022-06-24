// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s 2>&1 | \
// RUN:   FileCheck -check-prefix=ABI-O32 %s
// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -mabi=32 2>&1 | \
// RUN:   FileCheck -check-prefix=ABI-O32 %s
// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -mabi=o32 2>&1 | \
// RUN:   FileCheck -check-prefix=ABI-O32 %s
// ABI-O32: -cc1as
// ABI-O32: "-target-abi" "o32"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -mabi=eabi 2>&1 | \
// RUN:   FileCheck -check-prefix=ABI-EABI32 %s
// ABI-EABI32: -cc1as
// ABI-EABI32: "-target-abi" "eabi"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -mips64 -mabi=n32 2>&1 | \
// RUN:   FileCheck -check-prefix=ABI-N32 %s
// RUN: %clang -target mips64-linux-gnu -### -fintegrated-as -c %s -mabi=n32 2>&1 | \
// RUN:   FileCheck -check-prefix=ABI-N32 %s
// ABI-N32: -cc1as
// ABI-N32: "-target-abi" "n32"

// FIXME: We should also test '-target mips-linux-gnu -mips64' defaults to the
//        default 64-bit ABI (N64 but GCC uses N32). It currently selects O32
//        because of the triple.
// RUN: %clang -target mips64-linux-gnu -### -fintegrated-as -c %s -mips64 2>&1 | \
// RUN:   FileCheck -check-prefix=ABI-N64 %s

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -mips64 -mabi=64 2>&1 | \
// RUN:   FileCheck -check-prefix=ABI-N64 %s
// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -mips64 -mabi=n64 2>&1 | \
// RUN:   FileCheck -check-prefix=ABI-N64 %s
// RUN: %clang -target mips64-linux-gnu -### -fintegrated-as -c %s -mips64 -mabi=64 2>&1 | \
// RUN:   FileCheck -check-prefix=ABI-N64 %s
// RUN: %clang -target mips64-linux-gnu -### -fintegrated-as -c %s -mips64 -mabi=n64 2>&1 | \
// RUN:   FileCheck -check-prefix=ABI-N64 %s
// ABI-N64: -cc1as
// ABI-N64: "-target-abi" "n64"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -msoft-float 2>&1 | \
// RUN:   FileCheck -check-prefix=SOFTFLOAT %s
// SOFTFLOAT: -cc1as
// SOFTFLOAT: "-target-feature" "+soft-float"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s 2>&1 | \
// RUN:   FileCheck -check-prefix=HARDFLOAT %s
// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -mhard-float 2>&1 | \
// RUN:   FileCheck -check-prefix=HARDFLOAT %s
// HARDFLOAT: -cc1as
// HARDFLOAT-NOT: "-target-feature" "+soft-float"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s 2>&1 | \
// RUN:   FileCheck -check-prefix=NAN-DEFAULT %s
// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -mips32r6 2>&1 | \
// RUN:   FileCheck -check-prefix=NAN-DEFAULT %s
// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -mips64r6 2>&1 | \
// RUN:   FileCheck -check-prefix=NAN-DEFAULT %s
// NAN-DEFAULT: -cc1as
// NAN-DEFAULT-NOT: "-target-feature" "{{[-+]}}nan2008"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -mnan=legacy 2>&1 | \
// RUN:   FileCheck -check-prefix=NAN-LEGACY %s
// NAN-LEGACY: -cc1as
// NAN-LEGACY: "-target-feature" "-nan2008"

// RUN: %clang -target mips-linux-gnu -march=mips32r6 -### -fintegrated-as -c %s -mnan=2008 2>&1 | \
// RUN:   FileCheck -check-prefix=NAN-2008 %s
// NAN-2008: -cc1as
// NAN-2008: "-target-feature" "+nan2008"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s 2>&1 | \
// RUN:   FileCheck -check-prefix=DEFAULT-FLOAT %s
// DEFAULT-FLOAT: -cc1as
// DEFAULT-FLOAT-NOT: "-target-feature" "{{[+-]}}single-float"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -msingle-float 2>&1 | \
// RUN:   FileCheck -check-prefix=SINGLE-FLOAT %s
// SINGLE-FLOAT: -cc1as
// SINGLE-FLOAT: "-target-feature" "+single-float"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -mdouble-float 2>&1 | \
// RUN:   FileCheck -check-prefix=DOUBLE-FLOAT %s
// DOUBLE-FLOAT: -cc1as
// DOUBLE-FLOAT: "-target-feature" "-single-float"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s 2>&1 | \
// RUN:   FileCheck -check-prefix=MIPS16-DEFAULT %s
// MIPS16-DEFAULT: -cc1as
// MIPS16-DEFAULT-NOT: "-target-feature" "{{[+-]}}mips16"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -mips16 2>&1 | \
// RUN:   FileCheck -check-prefix=MIPS16-ON %s
// MIPS16-ON: -cc1as
// MIPS16-ON: "-target-feature" "+mips16"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -mno-mips16 2>&1 | \
// RUN:   FileCheck -check-prefix=MIPS16-OFF %s
// MIPS16-OFF: -cc1as
// MIPS16-OFF: "-target-feature" "-mips16"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s 2>&1 | \
// RUN:   FileCheck -check-prefix=MICROMIPS-DEFAULT %s
// MICROMIPS-DEFAULT: -cc1as
// MICROMIPS-DEFAULT-NOT: "-target-feature" "{{[+-]}}micromips"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -mmicromips 2>&1 | \
// RUN:   FileCheck -check-prefix=MICROMIPS-ON %s
// MICROMIPS-ON: -cc1as
// MICROMIPS-ON: "-target-feature" "+micromips"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -mno-micromips 2>&1 | \
// RUN:   FileCheck -check-prefix=MICROMIPS-OFF %s
// MICROMIPS-OFF: -cc1as
// MICROMIPS-OFF: "-target-feature" "-micromips"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s 2>&1 | \
// RUN:   FileCheck -check-prefix=DSP-DEFAULT %s
// DSP-DEFAULT: -cc1as
// DSP-DEFAULT-NOT: "-target-feature" "{{[+-]}}dsp"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -mdsp 2>&1 | \
// RUN:   FileCheck -check-prefix=DSP-ON %s
// DSP-ON: -cc1as
// DSP-ON: "-target-feature" "+dsp"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -mno-dsp 2>&1 | \
// RUN:   FileCheck -check-prefix=DSP-OFF %s
// DSP-OFF: -cc1as
// DSP-OFF: "-target-feature" "-dsp"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s 2>&1 | \
// RUN:   FileCheck -check-prefix=DSPR2-DEFAULT %s
// DSPR2-DEFAULT: -cc1as
// DSPR2-DEFAULT-NOT: "-target-feature" "{{[+-]}}dspr2"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -mdspr2 2>&1 | \
// RUN:   FileCheck -check-prefix=DSPR2-ON %s
// DSPR2-ON: -cc1as
// DSPR2-ON: "-target-feature" "+dspr2"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -mno-dspr2 2>&1 | \
// RUN:   FileCheck -check-prefix=DSPR2-OFF %s
// DSPR2-OFF: -cc1as
// DSPR2-OFF: "-target-feature" "-dspr2"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s 2>&1 | \
// RUN:   FileCheck -check-prefix=MSA-DEFAULT %s
// MSA-DEFAULT: -cc1as
// MSA-DEFAULT-NOT: "-target-feature" "{{[+-]}}msa"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -mmsa 2>&1 | \
// RUN:   FileCheck -check-prefix=MSA-ON %s
// MSA-ON: -cc1as
// MSA-ON: "-target-feature" "+msa"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -mno-msa 2>&1 | \
// RUN:   FileCheck -check-prefix=MSA-OFF %s
// MSA-OFF: -cc1as
// MSA-OFF: "-target-feature" "-msa"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s 2>&1 | \
// RUN:   FileCheck -check-prefix=FPXX-DEFAULT %s
// FPXX-DEFAULT: -cc1as
// FPXX-DEFAULT-NOT: "-target-feature" "+fpxx"
// FPXX-DEFAULT-NOT: "-target-feature" "+nooddspreg"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -mfp32 2>&1 | \
// RUN:   FileCheck -check-prefix=FP32 %s
// FP32: -cc1as
// FP32: "-target-feature" "-fp64"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -mfpxx 2>&1 | \
// RUN:   FileCheck -check-prefix=FPXX %s
// FPXX: -cc1as
// FPXX: "-target-feature" "+fpxx"
// FPXX: "-target-feature" "+nooddspreg"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -mfp64 2>&1 | \
// RUN:   FileCheck -check-prefix=FP64 %s
// FP64: -cc1as
// FP64: "-target-feature" "+fp64"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s 2>&1 | \
// RUN:   FileCheck -check-prefix=ODDSPREG-DEFAULT %s
// ODDSPREG-DEFAULT: -cc1as
// ODDSPREG-DEFAULT-NOT: "-target-feature" "{{[+-]}}nooddspreg"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -modd-spreg 2>&1 | \
// RUN:   FileCheck -check-prefix=ODDSPREG-ON %s
// ODDSPREG-ON: -cc1as
// ODDSPREG-ON: "-target-feature" "-nooddspreg"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -mno-odd-spreg 2>&1 | \
// RUN:   FileCheck -check-prefix=ODDSPREG-OFF %s
// ODDSPREG-OFF: -cc1as
// ODDSPREG-OFF: "-target-feature" "+nooddspreg"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -mfpxx -modd-spreg 2>&1 | \
// RUN:   FileCheck -check-prefix=FPXX-ODDSPREG %s
// FPXX-ODDSPREG: -cc1as
// FPXX-ODDSPREG: "-target-feature" "+fpxx"
// FPXX-ODDSPREG: "-target-feature" "-nooddspreg"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -mabicalls 2>&1 | \
// RUN:   FileCheck -check-prefix=ABICALLS-ON %s
// ABICALLS-ON: -cc1as
// ABICALLS-ON: "-target-feature" "-noabicalls"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -mno-abicalls 2>&1 | \
// RUN:   FileCheck -check-prefix=ABICALLS-OFF %s
// ABICALLS-OFF: -cc1as
// ABICALLS-OFF: "-target-feature" "+noabicalls"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -msoft-float -c %s 2>&1 | \
// RUN:   FileCheck -check-prefix=SOFTFLOAT-IMPLICIT-FPXX --implicit-check-not=-mfpxx %s
// SOFTFLOAT-IMPLICIT-FPXX: -cc1as
// SOFTFLOAT-IMPLICIT-FPXX: "-target-feature" "+soft-float"
// SOFTFLOAT-IMPLICIT-FPXX-NOT: "-target-feature" "+fpxx"
// SOFTFLOAT-IMPLICIT-FPXX-NOT: "-target-feature" "+nooddspreg"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -msoft-float -mfpxx -c %s 2>&1 | \
// RUN:   FileCheck -check-prefix=SOFTFLOAT-EXPLICIT-FPXX %s
// SOFTFLOAT-EXPLICIT-FPXX: -cc1as
// SOFTFLOAT-EXPLICIT-FPXX: "-target-feature" "+soft-float"
// SOFTFLOAT-EXPLICIT-FPXX: "-target-feature" "+fpxx"
// SOFTFLOAT-EXPLICIT-FPXX: "-target-feature" "+nooddspreg"

// RUN: %clang -target mips-mti-linux-gnu -### -fintegrated-as -msoft-float -c %s 2>&1 | \
// RUN:   FileCheck -check-prefix=MTI-SOFTFLOAT-IMPLICIT-FPXX --implicit-check-not=-mfpxx %s
// MTI-SOFTFLOAT-IMPLICIT-FPXX: -cc1as
// MTI-SOFTFLOAT-IMPLICIT-FPXX: "-target-feature" "+soft-float"
// MTI-SOFTFLOAT-IMPLICIT-FPXX-NOT: "-target-feature" "+fpxx"
// MTI-SOFTFLOAT-IMPLICIT-FPXX-NOT: "-target-feature" "+nooddspreg"

// RUN: %clang -target mips-mti-linux-gnu -### -fintegrated-as -msoft-float -mfpxx -c %s 2>&1 | \
// RUN:   FileCheck -check-prefix=MTI-SOFTFLOAT-EXPLICIT-FPXX %s
// MTI-SOFTFLOAT-EXPLICIT-FPXX: -cc1as
// MTI-SOFTFLOAT-EXPLICIT-FPXX: "-target-feature" "+soft-float"
// MTI-SOFTFLOAT-EXPLICIT-FPXX: "-target-feature" "+fpxx"
// MTI-SOFTFLOAT-EXPLICIT-FPXX: "-target-feature" "+nooddspreg"

// RUN: %clang -target mips-img-linux-gnu -### -fintegrated-as -msoft-float -c %s 2>&1 | \
// RUN:   FileCheck -check-prefix=IMG-SOFTFLOAT-IMPLICIT-FPXX --implicit-check-not=-mfpxx %s
// IMG-SOFTFLOAT-IMPLICIT-FPXX: -cc1as
// IMG-SOFTFLOAT-IMPLICIT-FPXX: "-target-feature" "+soft-float"
// IMG-SOFTFLOAT-IMPLICIT-FPXX-NOT: "-target-feature" "+fpxx"
// IMG-SOFTFLOAT-IMPLICIT-FPXX-NOT: "-target-feature" "+nooddspreg"

// RUN: %clang -target mips-img-linux-gnu -### -fintegrated-as -msoft-float -mfpxx -c %s 2>&1 | \
// RUN:   FileCheck -check-prefix=IMG-SOFTFLOAT-EXPLICIT-FPXX %s
// IMG-SOFTFLOAT-EXPLICIT-FPXX: -cc1as
// IMG-SOFTFLOAT-EXPLICIT-FPXX: "-target-feature" "+soft-float"
// IMG-SOFTFLOAT-EXPLICIT-FPXX: "-target-feature" "+fpxx"
// IMG-SOFTFLOAT-EXPLICIT-FPXX: "-target-feature" "+nooddspreg"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -msingle-float -c %s 2>&1 | \
// RUN:   FileCheck -check-prefix=SINGLEFLOAT-IMPLICIT-FPXX --implicit-check-not=-mfpxx %s
// SINGLEFLOAT-IMPLICIT-FPXX: -cc1as
// SINGLEFLOAT-IMPLICIT-FPXX: "-target-feature" "+single-float"
// SINGLEFLOAT-IMPLICIT-FPXX-NOT: "-target-feature" "+fpxx"
// SINGLEFLOAT-IMPLICIT-FPXX-NOT: "-target-feature" "+nooddspreg"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -msingle-float -mfpxx -c %s 2>&1 | \
// RUN:   FileCheck -check-prefix=SINGLEFLOAT-EXPLICIT-FPXX %s
// SINGLEFLOAT-EXPLICIT-FPXX: -cc1as
// SINGLEFLOAT-EXPLICIT-FPXX: "-target-feature" "+single-float"
// SINGLEFLOAT-EXPLICIT-FPXX: "-target-feature" "+fpxx"
// SINGLEFLOAT-EXPLICIT-FPXX: "-target-feature" "+nooddspreg"

// RUN: %clang -target mips-mti-linux-gnu -### -fintegrated-as -msingle-float -c %s 2>&1 | \
// RUN:   FileCheck -check-prefix=MTI-SINGLEFLOAT-IMPLICIT-FPXX --implicit-check-not=-mfpxx %s
// MTI-SINGLEFLOAT-IMPLICIT-FPXX: -cc1as
// MTI-SINGLEFLOAT-IMPLICIT-FPXX: "-target-feature" "+single-float"
// MTI-SINGLEFLOAT-IMPLICIT-FPXX-NOT: "-target-feature" "+fpxx"
// MTI-SINGLEFLOAT-IMPLICIT-FPXX-NOT: "-target-feature" "+nooddspreg"

// RUN: %clang -target mips-mti-linux-gnu -### -fintegrated-as -msingle-float -mfpxx -c %s 2>&1 | \
// RUN:   FileCheck -check-prefix=MTI-SINGLEFLOAT-EXPLICIT-FPXX %s
// MTI-SINGLEFLOAT-EXPLICIT-FPXX: -cc1as
// MTI-SINGLEFLOAT-EXPLICIT-FPXX: "-target-feature" "+single-float"
// MTI-SINGLEFLOAT-EXPLICIT-FPXX: "-target-feature" "+fpxx"
// MTI-SINGLEFLOAT-EXPLICIT-FPXX: "-target-feature" "+nooddspreg"

// RUN: %clang -target mips-img-linux-gnu -### -fintegrated-as -msingle-float -c %s 2>&1 | \
// RUN:   FileCheck -check-prefix=IMG-SINGLEFLOAT-IMPLICIT-FPXX --implicit-check-not=-mfpxx %s
// IMG-SINGLEFLOAT-IMPLICIT-FPXX: -cc1as
// IMG-SINGLEFLOAT-IMPLICIT-FPXX: "-target-feature" "+single-float"
// IMG-SINGLEFLOAT-IMPLICIT-FPXX-NOT: "-target-feature" "+fpxx"
// IMG-SINGLEFLOAT-IMPLICIT-FPXX-NOT: "-target-feature" "+nooddspreg"

// RUN: %clang -target mips-img-linux-gnu -### -fintegrated-as -msingle-float -mfpxx -c %s 2>&1 | \
// RUN:   FileCheck -check-prefix=IMG-SINGLEFLOAT-EXPLICIT-FPXX %s
// IMG-SINGLEFLOAT-EXPLICIT-FPXX: -cc1as
// IMG-SINGLEFLOAT-EXPLICIT-FPXX: "-target-feature" "+single-float"
// IMG-SINGLEFLOAT-EXPLICIT-FPXX: "-target-feature" "+fpxx"
// IMG-SINGLEFLOAT-EXPLICIT-FPXX: "-target-feature" "+nooddspreg"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -mxgot -c %s 2>&1 | \
// RUN:   FileCheck -check-prefix=XGOT %s
// XGOT: -cc1as
// XGOT: "-target-feature" "+xgot"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -mno-xgot -c %s 2>&1 | \
// RUN:   FileCheck -check-prefix=NOXGOT %s
// NOXGOT: -cc1as
// NOXGOT: "-target-feature" "-xgot"
