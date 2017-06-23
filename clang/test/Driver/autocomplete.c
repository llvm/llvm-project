// RUN: %clang --autocomplete=-fsyn | FileCheck %s -check-prefix=FSYN
// FSYN: -fsyntax-only
// RUN: %clang --autocomplete=-s | FileCheck %s -check-prefix=STD
// STD: -std={{.*}}-stdlib=
// RUN: %clang --autocomplete=foo | FileCheck %s -check-prefix=FOO
// FOO-NOT: foo
// RUN: %clang --autocomplete=-stdlib=,l | FileCheck %s -check-prefix=STDLIB
// STDLIB: libc++ libstdc++
// RUN: %clang --autocomplete=-stdlib=, | FileCheck %s -check-prefix=STDLIBALL
// STDLIBALL: libc++ libstdc++ platform
// RUN: %clang --autocomplete=-meabi,d | FileCheck %s -check-prefix=MEABI
// MEABI: default
// RUN: %clang --autocomplete=-meabi, | FileCheck %s -check-prefix=MEABIALL
// MEABIALL: 4 5 default gnu
// RUN: %clang --autocomplete=-cl-std=,CL2 | FileCheck %s -check-prefix=CLSTD
// CLSTD: CL2.0
// RUN: %clang --autocomplete=-cl-std=, | FileCheck %s -check-prefix=CLSTDALL
// CLSTDALL: cl CL cl1.1 CL1.1 cl1.2 CL1.2 cl2.0 CL2.0
// RUN: %clang --autocomplete=-fno-sanitize-coverage=,f | FileCheck %s -check-prefix=FNOSANICOVER
// FNOSANICOVER: func
// RUN: %clang --autocomplete=-fno-sanitize-coverage=, | FileCheck %s -check-prefix=FNOSANICOVERALL
// FNOSANICOVERALL: 8bit-counters bb edge func indirect-calls inline-8bit-counters no-prune trace-bb trace-cmp trace-div trace-gep trace-pc trace-pc-guard
// RUN: %clang --autocomplete=-ffp-contract=, | FileCheck %s -check-prefix=FFPALL
// FFPALL: fast off on
// RUN: %clang --autocomplete=-flto=, | FileCheck %s -check-prefix=FLTOALL
// FLTOALL: full thin
// RUN: %clang --autocomplete=-fveclib=, | FileCheck %s -check-prefix=FVECLIBALL
// FVECLIBALL: Accelerate none SVML
// RUN: %clang --autocomplete=-fshow-overloads=, | FileCheck %s -check-prefix=FSOVERALL
// FSOVERALL: all best
// RUN: %clang --autocomplete=-fvisibility=, | FileCheck %s -check-prefix=FVISIBILITYALL
// FVISIBILITYALL: default hidden
// RUN: %clang --autocomplete=-mfloat-abi=, | FileCheck %s -check-prefix=MFLOATABIALL
// MFLOATABIALL: hard soft softfp
// RUN: %clang --autocomplete=-mthread-model, | FileCheck %s -check-prefix=MTHREADMODELALL
// MTHREADMODELALL: posix single
// RUN: %clang --autocomplete=-mrelocation-model, | FileCheck %s -check-prefix=MRELOCMODELALL
// MRELOCMODELALL: dynamic-no-pic pic ropi ropi-rwpi rwpi static
