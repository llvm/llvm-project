!RUN: %flang -dM -E -o - -fopenmp -fopenmp-version=31 %s | FileCheck --check-prefix=V31 %s
!RUN: %flang -dM -E -o - -fopenmp -fopenmp-version=40 %s | FileCheck --check-prefix=V40 %s
!RUN: %flang -dM -E -o - -fopenmp -fopenmp-version=45 %s | FileCheck --check-prefix=V45 %s
!RUN: %flang -dM -E -o - -fopenmp -fopenmp-version=50 %s | FileCheck --check-prefix=V50 %s
!RUN: %flang -dM -E -o - -fopenmp -fopenmp-version=51 %s | FileCheck --check-prefix=V51 %s
!RUN: %flang -dM -E -o - -fopenmp -fopenmp-version=52 %s | FileCheck --check-prefix=V52 %s
!RUN: %flang -dM -E -o - -fopenmp -fopenmp-version=60 %s | FileCheck --check-prefix=V60 %s

!V31: #define _OPENMP 201107
!V40: #define _OPENMP 201307
!V45: #define _OPENMP 201511
!V50: #define _OPENMP 201811
!V51: #define _OPENMP 202011
!V52: #define _OPENMP 202111
!V60: #define _OPENMP 202411


!RUN: %flang -c -fopenmp -fopenmp-version=25 %s 2>&1 | FileCheck --check-prefix=WARN-ASSUMED %s

!WARN-ASSUMED: warning: OpenMP version 25 is no longer supported, assuming version 31


!RUN: not %flang -c -fopenmp -fopenmp-version=29 %s 2>&1 | FileCheck --check-prefix=ERR-BAD %s

!ERR-BAD: error: '29' is not a valid OpenMP version in '-fopenmp-version=29', valid versions are 31, 40, 45, 50, 51, 52, 60, 61

!RUN: %flang -c -fopenmp -fopenmp-version=61 %s 2>&1 | FileCheck --check-prefix=FUTURE %s

!FUTURE: The specification for OpenMP version 61 is still under development; the syntax and semantics of new features may be subject to change
