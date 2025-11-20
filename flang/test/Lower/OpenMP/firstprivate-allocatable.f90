! RUN: bbc -emit-hlfir -fopenmp -o - %s | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s | FileCheck %s

program firstprivateallocatable
  Integer, Allocatable :: a,u
  a = 137

  !$omp parallel firstprivate(a,u)
  u = a**2
  !$omp end parallel
end program


! CHECK-LABEL:   func.func @_QQmain()
! [...]
! CHECK:           omp.parallel {{.*}} {
! [...]
! CHECK:             %[[VAL_50:.*]] = arith.constant 2 : i32
! CHECK:             %[[VAL_51:.*]] = math.ipowi %{{.*}}, %[[VAL_50]] : i32
! this is what we are really checking: the hlfir.assign must have realloc so that
! u is allocated when the assignment occurs
! CHECK:             hlfir.assign %[[VAL_51]] to %{{.*}}#0 realloc : i32, !fir.ref<!fir.box<!fir.heap<i32>>>
! [...]
! CHECK:             omp.terminator
! CHECK:           }
