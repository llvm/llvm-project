// RUN: %clang --autocomplete=-fsyn | FileCheck %s -check-prefix=FSYN
// FSYN: -fsyntax-only
// RUN: %clang --autocomplete=-s | FileCheck %s -check-prefix=STD
// STD: -std={{.*}}-stdlib=
// RUN: %clang --autocomplete=foo | not FileCheck %s -check-prefix=NONE
// NONE: foo
// RUN: %clang --autocomplete=-stdlib=,l | FileCheck %s -check-prefix=STDLIB
// STDLIB: libc++ libstdc++
// RUN: %clang --autocomplete=-stdlib=, | FileCheck %s -check-prefix=STDLIBALL
// STDLIBALL: libc++ libstdc++ platform
// RUN: %clang --autocomplete=-meabi,d | FileCheck %s -check-prefix=MEABI
// MEABI: default
// RUN: %clang --autocomplete=-meabi, | FileCheck %s -check-prefix=MEABIALL
// MEABIALL: default 4 5 gnu
// RUN: %clang --autocomplete=-cl-std=,CL2 | FileCheck %s -check-prefix=CLSTD
// CLSTD: CL2.0
// RUN: %clang --autocomplete=-cl-std=, | FileCheck %s -check-prefix=CLSTDALL
// CLSTDALL: cl CL cl1.1 CL1.1 cl1.2 CL1.2 cl2.0 CL2.0
// RUN: %clang --autocomplete=-fno-sanitize-coverage=,f | FileCheck %s -check-prefix=FNOSANICOVER
// FNOSANICOVER: func
// RUN: %clang --autocomplete=-fno-sanitize-coverage=, | FileCheck %s -check-prefix=FNOSANICOVERALL
// FNOSANICOVERALL: func bb edge indirect-calls trace-bb trace-cmp trace-div trace-gep 8bit-counters trace-pc trace-pc-guard no-prune inline-8bit-counters
// RUN: %clang --autocomplete=-ffp-contract=, | FileCheck %s -check-prefix=FFPALL
// FFPALL: fast on off
// RUN: %clang --autocomplete=-flto=, | FileCheck %s -check-prefix=FLTOALL
// FLTOALL: thin full
// RUN: %clang --autocomplete=-fveclib=, | FileCheck %s -check-prefix=FVECLIBALL
// FVECLIBALL: Accelerate SVML none
// RUN: %clang --autocomplete=-fshow-overloads=, | FileCheck %s -check-prefix=FSOVERALL
// FSOVERALL: best all
// RUN: %clang --autocomplete=-fvisibility=, | FileCheck %s -check-prefix=FVISIBILITYALL
// FVISIBILITYALL: hidden default
// RUN: %clang --autocomplete=-mfloat-abi=, | FileCheck %s -check-prefix=MFLOATABIALL
// MFLOATABIALL: soft softfp hard
// RUN: %clang --autocomplete=-mthread-model, | FileCheck %s -check-prefix=MTHREADMODELALL
// MTHREADMODELALL: posix single
