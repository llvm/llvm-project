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

! CHECK:  omp.private {type = firstprivate} @[[X_PRIVATIZER:.*]] : ![[X_TYPE:fir.ptr<f32>]] alloc {
! CHECK:  ^bb0(%{{.*}}: ![[X_TYPE]]):
! CHECK:    %[[PRIV_ALLOC:.*]] = fir.alloca f32 {bindc_name = "x", {{.*}}}
! CHECK:    %[[PRIV_DECL:.*]]:2 = hlfir.declare %[[PRIV_ALLOC]] {{{.*}}} : (![[PRIV_TYPE:fir.ref<f32>]]) -> ({{.*}})
! CHECK:    %[[PRIV_CONV:.*]] = fir.convert %[[PRIV_DECL]]#0 : (![[PRIV_TYPE]]) -> ![[X_TYPE]]
! CHECK:    omp.yield(%[[PRIV_CONV]] : ![[X_TYPE]])
! CHECK:  } copy {
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
