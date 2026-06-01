! Test delayed privatization for variables that are storage associated via `EQUIVALENCE`.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -mmlir --enable-delayed-privatization \
! RUN:   -o - %s 2>&1 | FileCheck %s
! RUN: bbc -emit-hlfir -fopenmp --enable-delayed-privatization -o - %s 2>&1 \
! RUN:   | FileCheck %s

subroutine private_common
  real x, y
  equivalence (x,y)
  !$omp parallel firstprivate(x)
    x = 3.14
  !$omp end parallel
end subroutine

! Verify that the privatizer's alloc type is the underlying data type (f32),
! not fir.ptr<f32>. Variables in EQUIVALENCE are lowered with fir.ptr addresses
! (see castAliasToPointer in ConvertVariable.cpp), but they are not Fortran
! POINTERs. The privatizer must allocate storage for the actual data type.

! CHECK:  omp.private {type = firstprivate} @[[X_PRIVATIZER:.*]] : [[ALLOC_TYPE:f32]] copy {
! CHECK:  ^bb0(%[[ORIG_PTR:.*]]: ![[PTR_TYPE:fir.ptr<f32>]], %[[PRIV_REF:.*]]: ![[PTR_TYPE]]):
! CHECK:    %[[ORIG_VAL:.*]] = fir.load %[[ORIG_PTR]] : !fir.ptr<f32>
! CHECK:    hlfir.assign %[[ORIG_VAL]] to %[[PRIV_REF]] : f32, ![[PTR_TYPE]]
! CHECK:    omp.yield(%[[PRIV_REF]] : ![[PTR_TYPE]])
! CHECK:  }

! CHECK:  func.func @_QPprivate_common() {
! CHECK:    omp.parallel private(@[[X_PRIVATIZER]] %{{.*}}#0 -> %[[PRIV_ARG:.*]] : ![[PTR_TYPE]]) {
! CHECK:      %[[REG_DECL:.*]]:2 = hlfir.declare %[[PRIV_ARG]] {{{.*}}} : (![[PTR_TYPE]]) -> ({{.*}})
! CHECK:      %[[CST:.*]] = arith.constant {{.*}}
! CHECK:      hlfir.assign %[[CST]] to %[[REG_DECL]]#0 : {{.*}}
! CHECK:      omp.terminator
! CHECK:    }
! CHECK:    return
! CHECK:  }
