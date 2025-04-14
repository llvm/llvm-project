! RUN: %flang_fc1 -E %s 2>&1 | FileCheck %s --strict-whitespace --check-prefix=CHECK-E
! RUN: %flang_fc1 -fopenmp -fdebug-unparse %s 2>&1 | FileCheck %s --check-prefix=CHECK-OMP
! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s --check-prefix=CHECK-NO-OMP

!$    thread = OMP_GET_MAX_THREADS()

!$omp parallel private(ia)
!$    continue
!$omp end parallel
      end

!CHECK-E:{{^}}!$     thread = OMP_GET_MAX_THREADS()
!CHECK-E:{{^}}!$omp parallel private(ia)
!CHECK-E:{{^}}!$     continue
!CHECK-E:{{^}}!$omp end parallel

!CHECK-OMP:thread=omp_get_max_threads()
!CHECK-OMP:!$OMP PARALLEL  PRIVATE(ia)
!CHECK-OMP: CONTINUE
!CHECK-OMP:!$OMP END PARALLEL 

!CHECK-NO-OMP-NOT:thread=
!CHECK-NO-OMP-NOT:!$OMP
!CHECK-NO-OMP:END PROGRAM
