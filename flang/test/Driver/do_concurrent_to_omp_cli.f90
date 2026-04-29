! UNSUPPORTED: system-windows

! RUN: %flang --help | FileCheck %s --check-prefix=FLANG

! FLANG:      -fdo-concurrent=<value>
! FLANG-SAME:   Try to map `do concurrent` loops to OpenMP [none|host|device]

! RUN: bbc --help | FileCheck %s --check-prefix=BBC

! BBC:      -fdo-concurrent=<string>
! BBC-SAME:   Try to map `do concurrent` loops to OpenMP [none|host|device]

! RUN: %flang -c -fdo-concurrent=host %s 2>&1 \
! RUN: | FileCheck %s --check-prefix=OPT

! OPT: warning: OpenMP is required for lowering `do concurrent` loops to OpenMP.
! OPT-SAME:     Enable OpenMP using `-fopenmp`.

! RUN: not %flang -c -fopenmp -fdo-concurrent=devic,e %s 2>&1 \
! RUN: | FileCheck %s --check-prefix=BADVAL

! BADVAL: error: invalid value 'devic,e' in '-fdo-concurrent{{.*}}'

! RUN: %flang -c -fdo-concurrent-to-openmp=host %s 2>&1 \
! RUN: | FileCheck %s --check-prefix=OPT-ALIAS

! OPT-ALIAS: warning: OpenMP is required for lowering `do concurrent` loops to OpenMP.
! OPT-ALIAS-SAME:     Enable OpenMP using `-fopenmp`.

! RUN: not %flang -c -fopenmp -fdo-concurrent-to-openmp=devic,e %s 2>&1 \
! RUN: | FileCheck %s --check-prefix=BADVAL-ALIAS

! BADVAL-ALIAS: error: invalid value 'devic,e' in '-fdo-concurrent-to-openmp{{.*}}'

program test_cli
end program
