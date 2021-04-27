! This test checks lowering of OpenMP DO Directive(Worksharing).

! RUN: bbc -fopenmp -emit-fir %s -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect

!FIRDialect: func @_QQmain()
program wsloop
        integer :: i

!FIRDialect:  omp.parallel {
        !$OMP PARALLEL
!FIRDialect:    %[[PRIVATE_INDX:.*]] = fir.alloca i32 {{{.*}}uniq_name = "i"}
!FIRDialect:    %[[FINAL_INDX:.*]] = fir.do_loop %[[INDX:.*]] = {{.*}} {
        do i=1, 9
        print*, i
        end do
!FIRDialect:    }
!FIRDialect:    %[[FINAL_INDX_CVT:.*]] = fir.convert %[[FINAL_INDX]] : (index) -> i32
!FIRDialect:    fir.store %[[FINAL_INDX_CVT]] to %[[PRIVATE_INDX]] : !fir.ref<i32>
!FIRDialect:    omp.terminator
!FIRDialect:  }
        !$OMP END PARALLEL
end
