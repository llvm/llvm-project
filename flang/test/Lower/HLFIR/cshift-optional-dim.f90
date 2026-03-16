! Test lowering of CSHIFT/EOSHIFT with a dynamically optional DIM argument.
! When DIM is an optional dummy argument that is absent at runtime, it must
! default to 1 (Fortran standard 16.9.68, 16.9.77). Prior to the fix, the
! absent case caused a segmentation fault because the optional reference was
! unconditionally dereferenced without checking for presence.
! RUN: bbc -emit-hlfir -o - -I nowhere %s 2>&1 | FileCheck %s

! CSHIFT 2D with optional DIM - DIM may be absent at runtime (defaults to 1).
subroutine cshift_optional_dim(a, sh, dim)
  integer :: a(:,:), sh(:)
  integer, optional :: dim
  a = CSHIFT(a, sh, dim)
end subroutine
! CHECK-LABEL: func.func @_QPcshift_optional_dim(
! CHECK-SAME:    {{.*}}: !fir.ref<i32> {fir.bindc_name = "dim", fir.optional})
! CHECK:         %[[DECL_DIM:.*]]:2 = hlfir.declare {{.*}} {fortran_attrs = #fir.var_attrs<optional>
! CHECK:         %[[IS_PRESENT:.*]] = fir.is_present %[[DECL_DIM]]#0 : (!fir.ref<i32>) -> i1
! CHECK:         %[[DIM_LOADED:.*]] = fir.if %[[IS_PRESENT]] -> (i32) {
! CHECK:           %[[LOADED:.*]] = fir.load %[[DECL_DIM]]#0 : !fir.ref<i32>
! CHECK:           fir.result %[[LOADED]] : i32
! CHECK:         } else {
! CHECK:           arith.constant 0 : i32
! CHECK:           fir.result
! CHECK:         }
! CHECK:         %[[ONE:.*]] = arith.constant 1 : i32
! CHECK:         %[[DIM_VAL:.*]] = arith.select %[[IS_PRESENT]], %[[DIM_LOADED]], %[[ONE]] : i32
! CHECK-NOT:     arith.cmpi
! CHECK:         hlfir.cshift {{.*}} dim %[[DIM_VAL]] :

! EOSHIFT 2D with optional DIM - DIM may be absent at runtime (defaults to 1).
subroutine eoshift_optional_dim(a, sh, dim)
  integer :: a(:,:), sh(:)
  integer, optional :: dim
  a = EOSHIFT(a, sh, dim=dim)
end subroutine
! CHECK-LABEL: func.func @_QPeoshift_optional_dim(
! CHECK-SAME:    {{.*}}: !fir.ref<i32> {fir.bindc_name = "dim", fir.optional})
! CHECK:         %[[DECL_DIM2:.*]]:2 = hlfir.declare {{.*}} {fortran_attrs = #fir.var_attrs<optional>
! CHECK:         %[[IS_PRESENT2:.*]] = fir.is_present %[[DECL_DIM2]]#0 : (!fir.ref<i32>) -> i1
! CHECK:         %[[DIM_LOADED2:.*]] = fir.if %[[IS_PRESENT2]] -> (i32) {
! CHECK:           %[[LOADED2:.*]] = fir.load %[[DECL_DIM2]]#0 : !fir.ref<i32>
! CHECK:           fir.result %[[LOADED2]] : i32
! CHECK:         } else {
! CHECK:           arith.constant 0 : i32
! CHECK:           fir.result
! CHECK:         }
! CHECK:         %[[ONE2:.*]] = arith.constant 1 : i32
! CHECK:         %[[DIM_VAL2:.*]] = arith.select %[[IS_PRESENT2]], %[[DIM_LOADED2]], %[[ONE2]] : i32
! CHECK-NOT:     arith.cmpi
! CHECK:         hlfir.eoshift {{.*}} dim %[[DIM_VAL2]] :
