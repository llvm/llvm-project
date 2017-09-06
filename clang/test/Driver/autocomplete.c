// Test for the --autocompletion flag, which is an API used for shell
// autocompletion. You may have to update tests in this file when you
// add/modify flags, change HelpTexts or the values of some flags.

// Some corner cases.
// RUN: %clang --autocomplete= | FileCheck %s -check-prefix=ALL_FLAGS
// RUN: %clang --autocomplete=# | FileCheck %s -check-prefix=ALL_FLAGS
// Let's pick some example flags that are hopefully unlikely to change.
// ALL_FLAGS: -fast
// ALL_FLAGS: -fastcp
// ALL_FLAGS: -fastf
// Just test that this doesn't crash:
// RUN: %clang --autocomplete=,
// RUN: %clang --autocomplete==
// RUN: %clang --autocomplete=,,
// RUN: %clang --autocomplete=-

// RUN: %clang --autocomplete=-fsyn | FileCheck %s -check-prefix=FSYN
// FSYN: -fsyntax-only
// RUN: %clang --autocomplete=-std= | FileCheck %s -check-prefix=STD
// STD: -std= Language standard to compile for
// RUN: %clang --autocomplete=foo | FileCheck %s -check-prefix=FOO
// FOO-NOT: foo
// RUN: %clang --autocomplete=-stdlib=,l | FileCheck %s -check-prefix=STDLIB
// STDLIB: libc++
// STDLIB-NEXT: libstdc++
// RUN: %clang --autocomplete=-stdlib=, | FileCheck %s -check-prefix=STDLIBALL
// STDLIBALL: libc++
// STDLIBALL-NEXT: libstdc++
// STDLIBALL-NEXT: platform
// RUN: %clang --autocomplete=-meabi,d | FileCheck %s -check-prefix=MEABI
// MEABI: default
// RUN: %clang --autocomplete=-meabi, | FileCheck %s -check-prefix=MEABIALL
// MEABIALL: 4
// MEABIALL-NEXT: 5
// MEABIALL-NEXT: default
// MEABIALL-NEXT: gnu
// RUN: %clang --autocomplete=-cl-std=,CL2 | FileCheck %s -check-prefix=CLSTD
// CLSTD: CL2.0
// RUN: %clang --autocomplete=-cl-std=, | FileCheck %s -check-prefix=CLSTDALL
// CLSTDALL: cl
// CLSTDALL-NEXT: CL
// CLSTDALL-NEXT: cl1.1
// CLSTDALL-NEXT: CL1.1
// CLSTDALL-NEXT: cl1.2
// CLSTDALL-NEXT: CL1.2
// CLSTDALL-NEXT: cl2.0
// CLSTDALL-NEXT: CL2.0
// RUN: %clang --autocomplete=-fno-sanitize-coverage=,f | FileCheck %s -check-prefix=FNOSANICOVER
// FNOSANICOVER: func
// RUN: %clang --autocomplete=-fno-sanitize-coverage=, | FileCheck %s -check-prefix=FNOSANICOVERALL
// FNOSANICOVERALL: 8bit-counters
// FNOSANICOVERALL-NEXT: bb
// FNOSANICOVERALL-NEXT: edge
// FNOSANICOVERALL-NEXT: func
// FNOSANICOVERALL-NEXT: indirect-calls
// FNOSANICOVERALL-NEXT: inline-8bit-counters
// FNOSANICOVERALL-NEXT: no-prune
// FNOSANICOVERALL-NEXT: trace-bb
// FNOSANICOVERALL-NEXT: trace-cmp
// FNOSANICOVERALL-NEXT: trace-div
// FNOSANICOVERALL-NEXT: trace-gep
// FNOSANICOVERALL-NEXT: trace-pc
// FNOSANICOVERALL-NEXT: trace-pc-guard
// RUN: %clang --autocomplete=-ffp-contract=, | FileCheck %s -check-prefix=FFPALL
// FFPALL: fast
// FFPALL-NEXT: off
// FFPALL-NEXT: on
// RUN: %clang --autocomplete=-flto=, | FileCheck %s -check-prefix=FLTOALL
// FLTOALL: full
// FLTOALL-NEXT: thin
// RUN: %clang --autocomplete=-fveclib=, | FileCheck %s -check-prefix=FVECLIBALL
// FVECLIBALL: Accelerate
// FVECLIBALL-NEXT: none
// FVECLIBALL-NEXT: SVML
// RUN: %clang --autocomplete=-fshow-overloads=, | FileCheck %s -check-prefix=FSOVERALL
// FSOVERALL: all
// FSOVERALL-NEXT: best
// RUN: %clang --autocomplete=-fvisibility=, | FileCheck %s -check-prefix=FVISIBILITYALL
// FVISIBILITYALL: default
// FVISIBILITYALL-NEXT: hidden
// RUN: %clang --autocomplete=-mfloat-abi=, | FileCheck %s -check-prefix=MFLOATABIALL
// MFLOATABIALL: hard
// MFLOATABIALL-NEXT: soft
// MFLOATABIALL-NEXT: softfp
// RUN: %clang --autocomplete=-mthread-model, | FileCheck %s -check-prefix=MTHREADMODELALL
// MTHREADMODELALL: posix
// MTHREADMODELALL-NEXT: single
// RUN: %clang --autocomplete=-mrelocation-model, | FileCheck %s -check-prefix=MRELOCMODELALL
// MRELOCMODELALL: dynamic-no-pic
// MRELOCMODELALL-NEXT: pic
// MRELOCMODELALL-NEXT: ropi
// MRELOCMODELALL-NEXT: ropi-rwpi
// MRELOCMODELALL-NEXT: rwpi
// MRELOCMODELALL-NEXT: static
// RUN: %clang --autocomplete=-mrelocation-mode | FileCheck %s -check-prefix=MRELOCMODEL_CLANG
// MRELOCMODEL_CLANG-NOT: -mrelocation-model
// RUN: %clang --autocomplete=#-mrelocation-mode | FileCheck %s -check-prefix=MRELOCMODEL_CC1
// MRELOCMODEL_CC1: -mrelocation-model
// RUN: %clang --autocomplete=-Wma | FileCheck %s -check-prefix=WARNING
// WARNING: -Wmacro-redefined
// WARNING-NEXT: -Wmain
// WARNING-NEXT: -Wmain-return-type
// WARNING-NEXT: -Wmalformed-warning-check
// WARNING-NEXT: -Wmany-braces-around-scalar-init
// WARNING-NEXT: -Wmax-unsigned-zero
// RUN: %clang --autocomplete=-Wno-invalid-pp- | FileCheck %s -check-prefix=NOWARNING
// NOWARNING: -Wno-invalid-pp-token
// RUN: %clang --autocomplete=-analyzer-checker, | FileCheck %s -check-prefix=ANALYZER
// ANALYZER: unix.Malloc
// RUN: %clang --autocomplete=-std=, | FileCheck %s -check-prefix=STDVAL
// STDVAL: c99
