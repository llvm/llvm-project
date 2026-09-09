! RUN: %flang_fc1 -emit-hlfir -O0 -o - %s | FileCheck %s --check-prefixes=SPLIT,GUARD
! RUN: %flang_fc1 -emit-hlfir -O3 -o - %s | FileCheck %s --check-prefixes=SPLIT,GUARD
! RUN: %flang_fc1 -emit-hlfir -ffp-sum-reassociation -o - %s | FileCheck %s --check-prefixes=SPLIT,GUARD
! RUN: %flang_fc1 -emit-hlfir -fno-fp-sum-reassociation -o - %s | FileCheck %s --check-prefixes=ORDERED,GUARD
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s --check-prefixes=SPLIT,GUARD
! RUN: bbc -emit-hlfir -ffp-sum-reassociation=false -o - %s | FileCheck %s --check-prefixes=ORDERED,GUARD

! The implicit INTEGER(4)-to-INTEGER(8) conversions in array subscripts
! must not prevent reassociation of the REAL(8) sum (as in SNbone).
subroutine array_elements(i,x,a,b,c,d,e,f)
  integer(4) :: i
  real(8) :: x(100),a(100),b(100),c(100),d(100),e(100),f(100)
  x(i) = x(i) + a(i)*b(i) + c(i)*d(i) + e(i)*f(i)
end

! SPLIT-LABEL: func.func @_QParray_elements
! SPLIT: %[[CD:.*]] = arith.mulf
! SPLIT: %[[EF:.*]] = arith.mulf
! SPLIT: %[[TAIL:.*]] = arith.addf %[[CD]], %[[EF]]
! SPLIT: %[[AB:.*]] = arith.mulf
! SPLIT: %[[HEAD:.*]] = arith.addf %{{.*}}, %[[AB]]
! SPLIT: %[[SUM:.*]] = arith.addf %[[TAIL]], %[[HEAD]]
! SPLIT: hlfir.assign %[[SUM]]

! ORDERED-LABEL: func.func @_QParray_elements
! ORDERED: %[[AB:.*]] = arith.mulf
! ORDERED: %[[XAB:.*]] = arith.addf %{{.*}}, %[[AB]]
! ORDERED: %[[CD:.*]] = arith.mulf
! ORDERED: %[[XABCD:.*]] = arith.addf %[[XAB]], %[[CD]]
! ORDERED: %[[EF:.*]] = arith.mulf
! ORDERED: %[[SUM:.*]] = arith.addf %[[XABCD]], %[[EF]]
! ORDERED: hlfir.assign %[[SUM]]

! Both levels of indexing contain integer conversions.
subroutine nested_subscript(i,j,x,a,b,c,d,e,f)
  integer(4) :: i,j(100)
  real(8) :: x(100),a,b,c,d,e,f
  x(j(i)) = x(j(i)) + a*b + c*d + e*f
end

! SPLIT-LABEL: func.func @_QPnested_subscript
! SPLIT: %[[CD:.*]] = arith.mulf
! SPLIT: %[[EF:.*]] = arith.mulf
! SPLIT: %[[TAIL:.*]] = arith.addf %[[CD]], %[[EF]]
! SPLIT: %[[AB:.*]] = arith.mulf
! SPLIT: %[[HEAD:.*]] = arith.addf %{{.*}}, %[[AB]]
! SPLIT: %[[SUM:.*]] = arith.addf %[[TAIL]], %[[HEAD]]
! SPLIT: hlfir.assign %[[SUM]]

! ORDERED-LABEL: func.func @_QPnested_subscript
! ORDERED: %[[AB:.*]] = arith.mulf
! ORDERED: %[[XAB:.*]] = arith.addf %{{.*}}, %[[AB]]
! ORDERED: %[[CD:.*]] = arith.mulf
! ORDERED: %[[XABCD:.*]] = arith.addf %[[XAB]], %[[CD]]
! ORDERED: %[[EF:.*]] = arith.mulf
! ORDERED: %[[SUM:.*]] = arith.addf %[[XABCD]], %[[EF]]
! ORDERED: hlfir.assign %[[SUM]]

! A conversion of a real value used only as a subscript is also preserved.
subroutine real_subscript_conversion(r,x,a,b,c,d,e,f)
  real(4) :: r
  real(8) :: x(100),a,b,c,d,e,f
  x(int(r,8)) = x(int(r,8)) + a*b + c*d + e*f
end

! SPLIT-LABEL: func.func @_QPreal_subscript_conversion
! SPLIT: %[[CD:.*]] = arith.mulf
! SPLIT: %[[EF:.*]] = arith.mulf
! SPLIT: %[[TAIL:.*]] = arith.addf %[[CD]], %[[EF]]
! SPLIT: fir.convert %{{.*}} : (f32) -> i64
! SPLIT: %[[AB:.*]] = arith.mulf
! SPLIT: %[[HEAD:.*]] = arith.addf %{{.*}}, %[[AB]]
! SPLIT: %[[SUM:.*]] = arith.addf %[[TAIL]], %[[HEAD]]
! SPLIT: hlfir.assign %[[SUM]]

! ORDERED-LABEL: func.func @_QPreal_subscript_conversion
! ORDERED: %[[AB:.*]] = arith.mulf
! ORDERED: %[[XAB:.*]] = arith.addf %{{.*}}, %[[AB]]
! ORDERED: %[[CD:.*]] = arith.mulf
! ORDERED: %[[XABCD:.*]] = arith.addf %[[XAB]], %[[CD]]
! ORDERED: %[[EF:.*]] = arith.mulf
! ORDERED: %[[SUM:.*]] = arith.addf %[[XABCD]], %[[EF]]
! ORDERED: hlfir.assign %[[SUM]]

! Ignoring subscript conversions must not hide a conversion in the real sum.
subroutine mixed_kind_array_elements(i,x,a,b,c,d,e,f)
  integer(4) :: i
  real(8) :: x,a(100),b(100),c(100),d(100)
  real(4) :: e(100),f(100)
  x = a(i)*b(i) + c(i)*d(i) + real(e(i)*f(i),8)
end

! GUARD-LABEL: func.func @_QPmixed_kind_array_elements
! GUARD: %[[AB:.*]] = arith.mulf
! GUARD: %[[CD:.*]] = arith.mulf
! GUARD: %[[HEAD:.*]] = arith.addf %[[AB]], %[[CD]]
! GUARD: %[[EF:.*]] = arith.mulf %{{.*}}, %{{.*}} {{.*}} : f32
! GUARD: %[[CONVERT:.*]] = fir.convert %[[EF]] : (f32) -> f64
! GUARD: %[[SUM:.*]] = arith.addf %[[HEAD]], %[[CONVERT]]
! GUARD: hlfir.assign %[[SUM]]
