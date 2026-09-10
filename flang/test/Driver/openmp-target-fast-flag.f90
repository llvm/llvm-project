! REQUIRES: amdgpu-registered-target

! -fopenmp-target-fast is a meta-flag: on the OpenMP offload device compilation
! it implies -fopenmp-assume-no-thread-state and
! -fopenmp-assume-no-nested-parallelism, and forces -O3 unless an explicit
! optimization level is requested. It is only enabled when explicitly
! requested; -Ofast does not enable it. -fno-openmp-target-fast disables it.

! RUN: %flang -### -fopenmp --offload-arch=gfx90a -nogpulib -O0 %s 2>&1 \
! RUN:   | FileCheck -check-prefixes=DefaultTState,DefaultNoNestParallel %s

! RUN: %flang -### -fopenmp --offload-arch=gfx90a -nogpulib -O0 -fopenmp-target-fast %s 2>&1 \
! RUN:   | FileCheck -check-prefixes=TState,NestParallel %s

! RUN: %flang -### -fopenmp --offload-arch=gfx90a -nogpulib -fopenmp-target-fast %s 2>&1 \
! RUN:   | FileCheck -check-prefixes=O3,TState,NestParallel %s

! RUN: %flang -### -fopenmp --offload-arch=gfx90a -nogpulib -O3 -fno-openmp-target-fast %s 2>&1 \
! RUN:   | FileCheck -check-prefixes=DefaultTState,DefaultNoNestParallel %s

! -Ofast must NOT enable target-fast.
! RUN: %flang -### -fopenmp --offload-arch=gfx90a -nogpulib -Ofast %s 2>&1 \
! RUN:   | FileCheck -check-prefixes=DefaultTState,DefaultNoNestParallel %s

! O3: "-O3"

! TState: "-fopenmp-assume-no-thread-state"
! TState-NOT: "-fno-openmp-assume-no-thread-state"
! DefaultTState-NOT: "-f{{(no-)?}}openmp-assume-no-thread-state"

! NestParallel: "-fopenmp-assume-no-nested-parallelism"
! NestParallel-NOT: "-fno-openmp-assume-no-nested-parallelism"
! DefaultNoNestParallel-NOT: "-f{{(no-)?}}openmp-assume-no-nested-parallelism"

end program