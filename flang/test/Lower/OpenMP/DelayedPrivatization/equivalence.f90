! Test delayed privatization for variables that are storage associated via `EQUIVALENCE`.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -mmlir --openmp-enable-delayed-privatization \
! RUN:   -o - %s 2>&1 | FileCheck %s
! RUN: bbc -emit-hlfir -fopenmp --openmp-enable-delayed-privatization -o - %s 2>&1 \
! RUN:   | FileCheck %s

subroutine private_common
  real x, y
  equivalence (x,y)
  !$omp parallel firstprivate(x)
    x = 3.14
  !$omp end parallel
end subroutine

! TODO: the copy region for pointers is incorrect. OpenMP 5.2 says
!
! > If the original list item has the POINTER attribute, the new list items
! > receive the same association status as the original list item
!
! Currently the original pointer is unconditionally loaded, which is undefined
! behavior if that pointer is not associated.

! CHECK:  omp.private {type = firstprivate} @[[X_PRIVATIZER:.*]] : ![[X_TYPE:fir.ptr<f32>]] copy {
! CHECK:  ^bb0(%[[ORIG_PTR:.*]]: ![[X_TYPE]], %[[PRIV_REF:.*]]: ![[X_TYPE]]):
! CHECK:    %[[ORIG_VAL:.*]] = fir.load %[[ORIG_PTR]] : !fir.ptr<f32>
! CHECK:    hlfir.assign %[[ORIG_VAL]] to %[[PRIV_REF]] : f32, ![[X_TYPE]]
! CHECK:    omp.yield(%[[PRIV_REF]] : ![[X_TYPE]])
! CHECK:  }

! CHECK:  func.func @_QPprivate_common() {
! CHECK:    omp.parallel private(@[[X_PRIVATIZER]] %{{.*}}#0 -> %[[PRIV_ARG:.*]] : ![[X_TYPE]]) {
! CHECK:      %[[REG_DECL:.*]]:2 = hlfir.declare %[[PRIV_ARG]] {{{.*}}} : (![[X_TYPE]]) -> ({{.*}})
! CHECK:      %[[CST:.*]] = arith.constant {{.*}}
! CHECK:      hlfir.assign %[[CST]] to %[[REG_DECL]]#0 : {{.*}}
! CHECK:      omp.terminator
! CHECK:    }
! CHECK:    return
! CHECK:  }
