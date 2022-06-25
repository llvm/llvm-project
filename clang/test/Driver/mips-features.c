// Check handling MIPS specific features options.
//
// -mabicalls
// RUN: %clang -target mips-linux-gnu -### -c %s -mno-abicalls -mabicalls 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MABICALLS %s
// CHECK-MABICALLS: "-target-feature" "-noabicalls"
//
// -mno-abicalls
// RUN: %clang -target mips-linux-gnu -### -c %s -mabicalls -mno-abicalls 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MNOABICALLS %s
// CHECK-MNOABICALLS: "-target-feature" "+noabicalls"
//
// -mno-abicalls non-PIC N64
// RUN: %clang -target mips64-linux-gnu -### -c -fno-PIC -mno-abicalls %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MNOABICALLS-N64NPIC %s
// CHECK-MNOABICALLS-N64NPIC: "-target-feature" "+noabicalls"
//
// -mgpopt
// RUN: %clang -target mips-linux-gnu -### -c %s -mno-gpopt -mgpopt -Wno-unsupported-gpopt 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MGPOPT-DEF-ABICALLS %s
// CHECK-MGPOPT-DEF-ABICALLS-NOT: "-mllvm" "-mgpopt"
//
// -mabicalls -mgpopt
// RUN: %clang -target mips-linux-gnu -### -c %s -mabicalls -mno-gpopt -mgpopt -Wno-unsupported-gpopt 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MGPOPT-EXPLICIT-ABICALLS %s
// CHECK-MGPOPT-EXPLICIT-ABICALLS-NOT: "-mllvm" "-mgpopt"
//
// -mno-abicalls -mgpopt
// RUN: %clang -target mips-linux-gnu -### -c %s -mno-abicalls -mno-gpopt -mgpopt 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MGPOPT %s
// CHECK-MGPOPT: "-mllvm" "-mgpopt"
//
// -mno-abicalls -mno-gpopt
// RUN: %clang -target mips-linux-gnu -### -c %s -mno-abicalls -mgpopt -mno-gpopt 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MNOGPOPT %s
// CHECK-MNOGPOPT-NOT: "-mllvm" "-mgpopt"
//
// -mno-abicalls
// RUN: %clang -target mips-linux-gnu -### -c %s -mno-abicalls 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MGPOPTDEF %s
// CHECK-MGPOPTDEF: "-mllvm" "-mgpopt"
//
// -mgpopt -mno-abicalls -mlocal-sdata
// RUN: %clang -target mips-linux-gnu -### -c %s -mno-abicalls -mno-gpopt -mgpopt -mno-local-sdata -mlocal-sdata 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MLOCALSDATA %s
// CHECK-MLOCALSDATA: "-mllvm" "-mlocal-sdata=1"
//
// -mgpopt -mno-abicalls -mno-local-sdata
// RUN: %clang -target mips-linux-gnu -### -c %s -mno-abicalls -mno-gpopt -mgpopt -mlocal-sdata -mno-local-sdata 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MNOLOCALSDATA %s
// CHECK-MNOLOCALSDATA: "-mllvm" "-mlocal-sdata=0"
//
// -mgpopt -mno-abicalls
// RUN: %clang -target mips-linux-gnu -### -c %s -mno-abicalls -mgpopt 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MLOCALSDATADEF %s
// CHECK-MLOCALSDATADEF-NOT: "-mllvm" "-mlocal-sdata"
//
// -mno-abicalls -mgpopt -mextern-sdata
// RUN: %clang -target mips-linux-gnu -### -c %s -mno-abicalls -mgpopt -mno-extern-sdata -mextern-sdata 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MEXTERNSDATA %s
// CHECK-MEXTERNSDATA: "-mllvm" "-mextern-sdata=1"
//
// -mno-abicalls -mgpopt -mno-extern-sdata
// RUN: %clang -target mips-linux-gnu -### -c %s -mno-abicalls -mgpopt -mextern-sdata -mno-extern-sdata 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MNOEXTERNSDATA %s
// CHECK-MNOEXTERNSDATA: "-mllvm" "-mextern-sdata=0"
//
// -mno-abicalls -mgpopt
// RUN: %clang -target mips-linux-gnu -### -c %s -mno-abicalls -mgpopt 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MEXTERNSDATADEF %s
// CHECK-MEXTERNSDATADEF-NOT: "-mllvm" "-mextern-sdata"
//
// -mno-abicalls -mgpopt -membedded-data
// RUN: %clang -target mips-linux-gnu -### -c %s -mno-abicalls -mgpopt -mno-embedded-data -membedded-data 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MEMBEDDEDDATA %s
// CHECK-MEMBEDDEDDATA: "-mllvm" "-membedded-data=1"
//
// -mno-abicalls -mgpopt -mno-embedded-data
// RUN: %clang -target mips-linux-gnu -### -c %s -mno-abicalls -mgpopt -membedded-data -mno-embedded-data 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MNOEMBEDDEDDATA %s
// CHECK-MNOEMBEDDEDDATA: "-mllvm" "-membedded-data=0"
//
// -mno-abicalls -mgpopt
// RUN: %clang -target mips-linux-gnu -### -c %s -mno-abicalls -mgpopt 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MEMBEDDEDDATADEF %s
// CHECK-MEMBEDDEDDATADEF-NOT: "-mllvm" "-membedded-data"
//
// MIPS64 + N64: -fno-pic -> -mno-abicalls -mgpopt
// RUN: %clang -target mips64-mti-elf -mabi=64 -### -c %s -fno-pic -mno-abicalls 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-N64-GPOPT %s
// CHECK-N64-GPOPT: "-target-feature" "+noabicalls"
// CHECK-N64-GPOPT: "-mllvm" "-mgpopt"
//
// MIPS64 + N64: -fno-pic -mno-gpopt
// RUN: %clang -target mips64-mti-elf -mabi=64 -### -c %s -fno-pic -mno-abicalls -mno-gpopt 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-N64-MNO-GPOPT %s
// CHECK-N64-MNO-GPOPT: "-target-feature" "+noabicalls"
// CHECK-N64-MNO-GPOPT-NOT: "-mllvm" "-mgpopt"
//
// MIPS64 + N64: -mgpopt (-fpic is implicit)
// RUN: %clang -target mips64-mti-linux-gnu -mabi=64 -### -c %s -mgpopt 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-N64-PIC-GPOPT %s
// CHECK-N64-PIC-GPOPT-NOT: "-mllvm" "-mgpopt"
// CHECK-N64-PIC-GPOPT: ignoring '-mgpopt' option as it cannot be used with the implicit usage of -mabicalls
//
// -mips16
// RUN: %clang -target mips-linux-gnu -### -c %s \
// RUN:     -mno-mips16 -mips16 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS16 %s
// CHECK-MIPS16: "-target-feature" "+mips16"
//
// -mno-mips16
// RUN: %clang -target mips-linux-gnu -### -c %s \
// RUN:     -mips16 -mno-mips16 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NOMIPS16 %s
// CHECK-NOMIPS16: "-target-feature" "-mips16"
//
// -mmicromips
// RUN: %clang -target mips-linux-gnu -### -c %s \
// RUN:     -mno-micromips -mmicromips 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MICROMIPS %s
// CHECK-MICROMIPS: "-target-feature" "+micromips"
//
// -mno-micromips
// RUN: %clang -target mips-linux-gnu -### -c %s \
// RUN:     -mmicromips -mno-micromips 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NOMICROMIPS %s
// CHECK-NOMICROMIPS: "-target-feature" "-micromips"
//
// -mdsp
// RUN: %clang -target mips-linux-gnu -### -c %s \
// RUN:     -mno-dsp -mdsp 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MDSP %s
// CHECK-MDSP: "-target-feature" "+dsp"
//
// -mno-dsp
// RUN: %clang -target mips-linux-gnu -### -c %s \
// RUN:     -mdsp -mno-dsp 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NOMDSP %s
// CHECK-NOMDSP: "-target-feature" "-dsp"
//
// -mdspr2
// RUN: %clang -target mips-linux-gnu -### -c %s \
// RUN:     -mno-dspr2 -mdspr2 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MDSPR2 %s
// CHECK-MDSPR2: "-target-feature" "+dspr2"
//
// -mno-dspr2
// RUN: %clang -target mips-linux-gnu -### -c %s \
// RUN:     -mdspr2 -mno-dspr2 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NOMDSPR2 %s
// CHECK-NOMDSPR2: "-target-feature" "-dspr2"
//
// -mmsa
// RUN: %clang -target mips-linux-gnu -### -c %s \
// RUN:     -mno-msa -mmsa 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MMSA %s
// CHECK-MMSA: "-target-feature" "+msa"
//
// -mno-msa
// RUN: %clang -target mips-linux-gnu -### -c %s \
// RUN:     -mmsa -mno-msa 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NOMMSA %s
// CHECK-NOMMSA: "-target-feature" "-msa"
//
// -mmt
// RUN: %clang -target mips-linux-gnu -### -c %s \
// RUN:     -mno-mt -mmt 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MMT %s
// CHECK-MMT: "-target-feature" "+mt"
//
// -mno-mt
// RUN: %clang -target mips-linux-gnu -### -c %s \
// RUN:     -mmt -mno-mt 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NOMMT %s
// CHECK-NOMMT: "-target-feature" "-mt"
//
// -modd-spreg
// RUN: %clang -target mips-linux-gnu -### -c %s -mno-odd-spreg -modd-spreg 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MODDSPREG %s
// CHECK-MODDSPREG: "-target-feature" "-nooddspreg"
//
// -mno-odd-spreg
// RUN: %clang -target mips-linux-gnu -### -c %s -modd-spreg -mno-odd-spreg 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NOMODDSPREG %s
// CHECK-NOMODDSPREG: "-target-feature" "+nooddspreg"
//
// -mfpxx
// RUN: %clang -target mips-linux-gnu -### -c %s -mfpxx 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MFPXX %s
// CHECK-MFPXX: "-target-feature" "+fpxx"
// CHECK-MFPXX: "-target-feature" "+nooddspreg"
//
// -mfpxx -modd-spreg
// RUN: %clang -target mips-linux-gnu -### -c %s -mfpxx -modd-spreg 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MFPXX-ODDSPREG %s
// CHECK-MFPXX-ODDSPREG: "-target-feature" "+fpxx"
// CHECK-MFPXX-ODDSPREG: "-target-feature" "-nooddspreg"
//
// -mfp64
// RUN: %clang -target mips-linux-gnu -### -c %s \
// RUN:     -mfp32 -mfp64 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MFP64 %s
// CHECK-MFP64: "-target-feature" "+fp64"
//
// -mfp32
// RUN: %clang -target mips-linux-gnu -### -c %s \
// RUN:     -mfp64 -mfp32 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NOMFP64 %s
// CHECK-NOMFP64: "-target-feature" "-fp64"
//
// -mnan=2008
// RUN: %clang -target mips-linux-gnu -march=mips32r3 -### -c %s \
// RUN:     -mnan=legacy -mnan=2008 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NAN2008 %s
// CHECK-NAN2008: "-target-feature" "+nan2008"
//
// -mnan=legacy
// RUN: %clang -target mips-linux-gnu -march=mips32r3 -### -c %s \
// RUN:     -mnan=2008 -mnan=legacy 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NANLEGACY %s
// CHECK-NANLEGACY: "-target-feature" "-nan2008"
//
// -mabs=2008 on pre R2
// RUN: %clang -target mips-linux-gnu -march=mips32 -### -c %s \
// RUN:     -mabs=2008 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-ABSLEGACY %s
//
// -mabs=2008
// RUN: %clang -target mips-linux-gnu -march=mips32r3 -### -c %s \
// RUN:     -mabs=2008 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-ABS2008 %s
//
// -mabs=legacy
// RUN: %clang -target mips-linux-gnu -march=mips32r3 -### -c %s \
// RUN:     -mabs=legacy 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-ABSLEGACY %s
//
// -mabs=legacy on R6
// RUN: %clang -target mips-linux-gnu -march=mips32r6 -### -c %s \
// RUN:     -mabs=legacy 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-ABS2008 %s
//
// CHECK-ABSLEGACY: "-target-feature" "-abs2008"
// CHECK-ABSLEGACY-NOT: "-target-feature" "+abs2008"
// CHECK-ABS2008: "-target-feature" "+abs2008"
// CHECK-ABS2008-NOT: "-target-feature" "-abs2008"
//
// -mcompact-branches=never
// RUN: %clang -target mips-linux-gnu -march=mips32r6 -### -c %s \
// RUN:     -mcompact-branches=never 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CBNEVER %s
// CHECK-CBNEVER: "-mllvm" "-mips-compact-branches=never"
//
// -mcompact-branches=optimal
// RUN: %clang -target mips-linux-gnu -march=mips32r6 -### -c %s \
// RUN:     -mcompact-branches=optimal 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CBOPTIMAL %s
// CHECK-CBOPTIMAL: "-mllvm" "-mips-compact-branches=optimal"
//
// -mcompact-branches=always
// RUN: %clang -target mips-linux-gnu -march=mips32r6 -### -c %s \
// RUN:     -mcompact-branches=always 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CBALWAYS %s
// CHECK-CBALWAYS: "-mllvm" "-mips-compact-branches=always"
//
// -mxgot
// RUN: %clang -target mips-linux-gnu -### -c %s \
// RUN:     -mno-xgot -mxgot 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-XGOT %s
// CHECK-XGOT: "-target-feature" "+xgot"
//
// -mno-xgot
// RUN: %clang -target mips-linux-gnu -### -c %s \
// RUN:     -mxgot -mno-xgot 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NOXGOT %s
// CHECK-NOXGOT: "-target-feature" "-xgot"
//
// -mldc1-sdc1
// RUN: %clang -target mips-linux-gnu -### -c %s \
// RUN:     -mno-ldc1-sdc1 -mldc1-sdc1 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LDC1SDC1 %s
// CHECK-LDC1SDC1-NOT: "-mllvm" "-mno-ldc1-sdc1"
//
// -mno-ldc1-sdc1
// RUN: %clang -target mips-linux-gnu -### -c %s \
// RUN:     -mldc1-sdc1 -mno-ldc1-sdc1 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NOLDC1SDC1 %s
// CHECK-NOLDC1SDC1: "-mllvm" "-mno-ldc1-sdc1"
//
// -mcheck-zero-division
// RUN: %clang -target mips-linux-gnu -### -c %s \
// RUN:     -mno-check-zero-division -mcheck-zero-division 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-ZERODIV %s
// CHECK-ZERODIV-NOT: "-mllvm" "-mno-check-zero-division"
//
// -mno-check-zero-division
// RUN: %clang -target mips-linux-gnu -### -c %s \
// RUN:     -mcheck-zero-division -mno-check-zero-division 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NOZERODIV %s
// CHECK-NOZERODIV: "-mllvm" "-mno-check-zero-division"
//
// -G
// RUN: %clang -target mips-linux-gnu -### -c %s \
// RUN:     -G 16 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS-G %s
// CHECK-MIPS-G: "-mllvm" "-mips-ssection-threshold=16"
//
// -msoft-float (unknown vendor)
// RUN: %clang -target mips-linux-gnu -### -c %s -msoft-float 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-SOFTFLOAT %s
// CHECK-SOFTFLOAT: "-target-feature" "+soft-float"
// CHECK-SOFTFLOAT-NOT: "-target-feature" "+fpxx"
//
// -msoft-float -mfpxx (unknown vendor)
// RUN: %clang -target mips-linux-gnu -### -c %s -msoft-float -mfpxx 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-SOFTFLOAT-FPXX %s
// CHECK-SOFTFLOAT-FPXX: "-target-feature" "+soft-float"
// CHECK-SOFTFLOAT-FPXX: "-target-feature" "+fpxx"
//
// -msoft-float (MTI)
// RUN: %clang -target mips-mti-linux-gnu -### -c %s -msoft-float 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MTI-SOFTFLOAT %s
// CHECK-MTI-SOFTFLOAT: "-target-feature" "+soft-float"
// CHECK-MTI-SOFTFLOAT-NOT: "-target-feature" "+fpxx"
//
// -msoft-float -mfpxx (MTI)
// RUN: %clang -target mips-mti-linux-gnu -### -c %s -msoft-float -mfpxx 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MTI-SOFTFLOAT-FPXX %s
// CHECK-MTI-SOFTFLOAT-FPXX: "-target-feature" "+soft-float"
// CHECK-MTI-SOFTFLOAT-FPXX: "-target-feature" "+fpxx"
//
// -msoft-float (IMG)
// RUN: %clang -target mips-img-linux-gnu -### -c %s -msoft-float 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-IMG-SOFTFLOAT %s
// CHECK-IMG-SOFTFLOAT: "-target-feature" "+soft-float"
// CHECK-IMG-SOFTFLOAT-NOT: "-target-feature" "+fpxx"
//
// -msoft-float -mfpxx (IMG)
// RUN: %clang -target mips-img-linux-gnu -### -c %s -msoft-float -mfpxx 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-IMG-SOFTFLOAT-FPXX %s
// CHECK-IMG-SOFTFLOAT-FPXX: "-target-feature" "+soft-float"
// CHECK-IMG-SOFTFLOAT-FPXX: "-target-feature" "+fpxx"
//
// -msingle-float (unknown vendor)
// RUN: %clang -target mips-linux-gnu -### -c %s -msingle-float 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-SINGLEFLOAT %s
// CHECK-SINGLEFLOAT: "-target-feature" "+single-float"
// CHECK-SINGLEFLOAT-NOT: "-target-feature" "+fpxx"
//
// -msingle-float -mfpxx (unknown vendor)
// RUN: %clang -target mips-linux-gnu -### -c %s -msingle-float -mfpxx 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-SINGLEFLOAT-FPXX %s
// CHECK-SINGLEFLOAT-FPXX: "-target-feature" "+single-float"
// CHECK-SINGLEFLOAT-FPXX: "-target-feature" "+fpxx"
//
// -msingle-float (MTI)
// RUN: %clang -target mips-mti-linux-gnu -### -c %s -msingle-float 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MTI-SINGLEFLOAT %s
// CHECK-MTI-SINGLEFLOAT: "-target-feature" "+single-float"
// CHECK-MTI-SINGLEFLOAT-NOT: "-target-feature" "+fpxx"
//
// -msingle-float -mfpxx (MTI)
// RUN: %clang -target mips-mti-linux-gnu -### -c %s -msingle-float -mfpxx 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MTI-SINGLEFLOAT-FPXX %s
// CHECK-MTI-SINGLEFLOAT-FPXX: "-target-feature" "+single-float"
// CHECK-MTI-SINGLEFLOAT-FPXX: "-target-feature" "+fpxx"
//
// -msingle-float (IMG)
// RUN: %clang -target mips-img-linux-gnu -### -c %s -msingle-float 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-IMG-SINGLEFLOAT %s
// CHECK-IMG-SINGLEFLOAT: "-target-feature" "+single-float"
// CHECK-IMG-SINGLEFLOAT-NOT: "-target-feature" "+fpxx"
//
// -msingle-float -mfpxx (IMG)
// RUN: %clang -target mips-img-linux-gnu -### -c %s -msingle-float -mfpxx 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-IMG-SINGLEFLOAT-FPXX %s
// CHECK-IMG-SINGLEFLOAT-FPXX: "-target-feature" "+single-float"
// CHECK-IMG-SINGLEFLOAT-FPXX: "-target-feature" "+fpxx"

// -mlong-call
// RUN: %clang -target mips-img-linux-gnu -### -c %s \
// RUN:        -mno-abicalls -mlong-calls 2>&1 \
// RUN:   | FileCheck --check-prefix=LONG-CALLS-ON %s
// RUN: %clang -target mips-img-linux-gnu -### -c %s \
// RUN:        -mno-abicalls -mno-long-calls 2>&1 \
// RUN:   | FileCheck --check-prefix=LONG-CALLS-OFF %s
// RUN: %clang -target mips-img-linux-gnu -### -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=LONG-CALLS-DEF %s
// RUN: %clang -target mips-img-linux-gnu -### -c %s -mlong-calls 2>&1 \
// RUN:   | FileCheck --check-prefix=LONG-CALLS-DEF %s
// LONG-CALLS-ON: "-target-feature" "+long-calls"
// LONG-CALLS-OFF: "-target-feature" "-long-calls"
// LONG-CALLS-DEF-NOT: "long-calls"
//
// -mbranch-likely
// RUN: %clang -target -mips-mti-linux-gnu -### -c %s -mbranch-likely 2>&1 \
// RUN:   | FileCheck --check-prefix=BRANCH-LIKELY %s
// BRANCH-LIKELY: argument unused during compilation: '-mbranch-likely'
//
// -mno-branch-likely
// RUN: %clang -target -mips-mti-linux-gnu -### -c %s -mno-branch-likely 2>&1 \
// RUN:   | FileCheck --check-prefix=NO-BRANCH-LIKELY %s
// NO-BRANCH-LIKELY: argument unused during compilation: '-mno-branch-likely'

// -mindirect-jump=hazard
// RUN: %clang -target mips-unknown-linux-gnu -### -c %s \
// RUN:        -mindirect-jump=hazard 2>&1 \
// RUN:   | FileCheck --check-prefix=INDIRECT-BH %s
// INDIRECT-BH: "-target-feature" "+use-indirect-jump-hazard"
//
// -mcrc
// RUN: %clang -target mips-unknown-linux-gnu -### -c %s \
// RUN:     -mno-crc -mcrc 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CRC %s
// CHECK-CRC: "-target-feature" "+crc"
//
// -mno-crc
// RUN: %clang -target mips-unknown-linux-gnu -### -c %s \
// RUN:     -mcrc -mno-crc 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-CRC %s
// CHECK-NO-CRC: "-target-feature" "-crc"
//
// -mvirt
// RUN: %clang -target mips-unknown-linux-gnu -### -c %s \
// RUN:     -mno-virt -mvirt 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-VIRT %s
// CHECK-VIRT: "-target-feature" "+virt"
//
// -mno-virt
// RUN: %clang -target mips-unknown-linux-gnu -### -c %s \
// RUN:     -mvirt -mno-virt 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-VIRT %s
// CHECK-NO-VIRT: "-target-feature" "-virt"
//
// -mginv
// RUN: %clang -target mips-unknown-linux-gnu -### -c %s \
// RUN:     -mno-ginv -mginv 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-GINV %s
// CHECK-GINV: "-target-feature" "+ginv"
//
// -mno-ginv
// RUN: %clang -target mips-unknown-linux-gnu -### -c %s \
// RUN:     -mginv -mno-ginv 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-GINV %s
// CHECK-NO-GINV: "-target-feature" "-ginv"
//
// -mrelax-pic-calls
// RUN: %clang -target mips-unknown-linux-gnu -### -c %s \
// RUN:     -mno-relax-pic-calls -mrelax-pic-calls 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-RELAX-PIC-CALLS %s
// CHECK-RELAX-PIC-CALLS-NOT: "-mllvm" "-mips-jalr-reloc=0"
//
// -mno-relax-pic-calls
// RUN: %clang -target mips-unknown-linux-gnu -### -c %s \
// RUN:     -mrelax-pic-calls -mno-relax-pic-calls 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-RELAX-PIC-CALLS %s
// CHECK-NO-RELAX-PIC-CALLS: "-mllvm" "-mips-jalr-reloc=0"
