! UNSUPPORTED: system-windows

! RUN: %flang --help | FileCheck %s --check-prefix=FLANG

! FLANG:      -fdo-concurrent-device-pure-procedures
! FLANG-NEXT:   Emit a device version of every pure procedure defined in the file when `do concurrent` loops are mapped to the device
! FLANG:      -fdo-concurrent-to-openmp=<value>
! FLANG-NEXT:   Try to map `do concurrent` loops to OpenMP [none|host|device]

! RUN: bbc --help | FileCheck %s --check-prefix=BBC

! BBC:      -fdo-concurrent-device-pure-procedures
! BBC-SAME:   Emit a device version of every pure procedure defined in the file when `do concurrent` loops are mapped to the device
! BBC:      -fdo-concurrent-to-openmp=<string>
! BBC-SAME:   Try to map `do concurrent` loops to OpenMP [none|host|device]

! RUN: %flang -### -c -fopenmp -fdo-concurrent-to-openmp=device \
! RUN:   -fdo-concurrent-device-pure-procedures %s 2>&1 \
! RUN: | FileCheck %s --check-prefix=FORWARD

! FORWARD: "-fc1"{{.*}} "-fdo-concurrent-device-pure-procedures"

! RUN: %flang -c -fdo-concurrent-to-openmp=host %s 2>&1 \
! RUN: | FileCheck %s --check-prefix=OPT

! RUN: %flang -c -fno-openmp -fdo-concurrent-to-openmp=host %s 2>&1 \
! RUN: | FileCheck %s --check-prefix=OPT

! RUN: %flang -c -fopenmp-simd -fdo-concurrent-to-openmp=host %s 2>&1 \
! RUN: | FileCheck %s --check-prefix=OPT

! RUN: %flang -c -fno-openmp -fopenmp-simd -fdo-concurrent-to-openmp=host %s 2>&1 \
! RUN: | FileCheck %s --check-prefix=OPT

! OPT: warning: OpenMP is required for lowering `do concurrent` loops to OpenMP.
! OPT-SAME:     Enable OpenMP using `-fopenmp`.

! RUN: not %flang -c -fopenmp -fdo-concurrent-to-openmp=devic,e %s 2>&1 \
! RUN: | FileCheck %s --check-prefix=BADVAL

! BADVAL: error: invalid value 'devic,e' in '-fdo-concurrent-to-openmp{{.*}}'

program test_cli
end program
