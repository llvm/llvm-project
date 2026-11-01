! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s --check-prefixes=COARRAY
! RUN: not %flang_fc1 -emit-hlfir %s 2>&1 | FileCheck %s --check-prefixes=NOCOARRAY

! NOCOARRAY: Not yet implemented: Multi-image features are experimental and are disabled by default, use '-fcoarray' to enable.

program image_status_test
  use iso_fortran_env, only : team_type
  implicit none
  integer :: i, s
  integer(kind=8) :: d
  type(team_type) :: t

! COARRAY-DAG: %[[D:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFEd") : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
! COARRAY-DAG: %[[I:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFEi") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! COARRAY-DAG: %[[S:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFEs") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! COARRAY-DAG: %[[T:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFEt") : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{{.*}}>>) -> (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{{.*}}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{{.*}}>>)

 
  s = image_status(i)
! COARRAY: %[[LOAD1:.*]] = fir.load %[[I]]#0 : !fir.ref<i32>
! COARRAY-NEXT: %[[STATUS1:.*]] = mif.image_status %[[LOAD1]] : (i32) -> i32
! COARRAY-NEXT: hlfir.assign %[[STATUS1]] to %[[S]]#0 : i32, !fir.ref<i32>
 
  s = image_status(1)
! COARRAY: %[[C1:.*]] = arith.constant 1 : i32
! COARRAY-NEXT: %[[STATUS2:.*]] = mif.image_status %[[C1]] : (i32) -> i32
! COARRAY-NEXT: hlfir.assign %[[STATUS2]] to %[[S]]#0 : i32, !fir.ref<i32>
 
  d = image_status(i)
! COARRAY: %[[LOAD3:.*]] = fir.load %[[I]]#0 : !fir.ref<i32>
! COARRAY-NEXT: %[[STATUS3:.*]] = mif.image_status %[[LOAD3]] : (i32) -> i32
! COARRAY-NEXT: %[[CONV:.*]] = fir.convert %[[STATUS3]] : (i32) -> i64
! COARRAY-NEXT: hlfir.assign %[[CONV]] to %[[D]]#0 : i64, !fir.ref<i64>

  s = image_status(i, TEAM=t)
! COARRAY: %[[LOAD4:.*]] = fir.load %[[I]]#0 : !fir.ref<i32>
! COARRAY-NEXT: %[[STATUS4:.*]] = mif.image_status %[[LOAD4]] team %[[T]]#0 : (i32, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{{.*}}>>) -> i32
! COARRAY-NEXT: hlfir.assign %[[STATUS4]] to %[[S]]#0 : i32, !fir.ref<i32>
 
! COARRAY: return
end program
