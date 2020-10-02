! This test checks lowering of OpenMP Master Directive to FIR + OpenMP Dialect.

! RUN: bbc -fopenmp -emit-fir %s -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect

subroutine foo()

!$OMP MASTER
!FIRDialect-LABEL: func @_QPfoo() {
!FIRDialect: omp.master  {
!FIRDialect:   fir.call @_FortranAioBeginExternalListOutput
!FIRDialect:   fir.call @_FortranAioOutputAscii
!FIRDialect:   fir.call @_FortranAioEndIoStatement
!FIRDialect: omp.terminator
!FIRDialect: }
!FIRDialect:  return
!FIRDialect: }
print*, "Master region"
!$OMP END MASTER

end subroutine

program main

        integer :: alpha, beta, gama
        alpha =  4
        beta = 5
        gama = 6
!$OMP PARALLEL
print*, "Parallel region"
!FIRDialect-LABEL: func @_QQmain() {
!FIRDialect: omp.parallel {
!FIRDialect:   fir.call @_FortranAioBeginExternalListOutput
!FIRDialect:   fir.call @_FortranAioOutputAscii
!FIRDialect:   fir.call @_FortranAioEndIoStatement

!$OMP MASTER
!FIRDialect: omp.master {
!FIRDialect:   fir.call @_FortranAioBeginExternalListOutput
!FIRDialect:   fir.call @_FortranAioOutputAscii
!FIRDialect:   fir.call @_FortranAioEndIoStatement
!FIRDialect: omp.terminator
!FIRDialect: }
print*, "Master region"
!$OMP END MASTER

!FIRDialect: omp.terminator
!FIRDialect: }

!$OMP END PARALLEL

!$OMP PARALLEL
!FIRDialect: omp.parallel {
!FIRDialect:   fir.call @_QPfoo() : () -> ()
!FIRDialect: omp.terminator
!FIRDialect: }
call foo()
!$OMP END PARALLEL


!$OMP MASTER
!FIRDialect: omp.master  {
!FIRDialect:   %{{.*}} = fir.load %{{.*}}
!FIRDialect:   %{{.*}} = fir.load %{{.*}}
!FIRDialect:   %[[RESULT:.*]] = cmpi "sge", %{{.*}}, %{{.*}}
!FIRDialect:   fir.if %[[RESULT]] {
if (alpha .ge. gama) then
!$OMP PARALLEL
!FIRDialect:   omp.parallel {
!FIRDialect:     fir.call @_FortranAioBeginExternalListOutput
!FIRDialect:     fir.call @_FortranAioOutputInteger64
!FIRDialect:     fir.call @_FortranAioEndIoStatement
!FIRDialect:   omp.terminator
!FIRDialect:   }
print*, alpha
!$OMP END PARALLEL
 beta = alpha + gama
end if
!FIRDialect:   %{{.*}} = fir.load %{{.*}}
!FIRDialect:   %{{.*}} = fir.load %{{.*}}
!FIRDialect:   %{{.*}} = addi %{{.*}}, %{{.*}}
!FIRDialect:   fir.store %{{.*}} to %{{.*}}
!FIRDialect:   } else {
!FIRDialect:   }
!FIRDialect: omp.terminator
!FIRDialect: }
!$OMP END MASTER

end
