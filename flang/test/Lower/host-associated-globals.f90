! Test lowering of internal procedure host association for global variables
! A tuple function argument should not be created for associated globals, and
! instead globals should be instantiated with a fir.address_of inside the
! contained procedures.
! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s

module test_mod_used_in_host
  integer :: i, j_in_equiv
  integer :: not_in_equiv
  equivalence (i,j_in_equiv)
end module

subroutine module_var()
  use test_mod_used_in_host
  call bar()
contains
 subroutine bar()
    print *, j_in_equiv, not_in_equiv
 end subroutine
end subroutine
! CHECK-LABEL: func.func private @_QFmodule_varPbar()
! CHECK:  %[[VAL_0:.*]] = fir.address_of(@_QMtest_mod_used_in_hostEi) : !fir.ref<!fir.array<4xi8>>
! CHECK:  %[[VAL_1:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_2:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_1]] : (!fir.ref<!fir.array<4xi8>>, index) -> !fir.ref<i8>
! CHECK:  %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<i8>) -> !fir.ptr<i32>
! CHECK:  %[[VAL_4:.*]] = fir.address_of(@_QMtest_mod_used_in_hostEnot_in_equiv) : !fir.ref<i32>

subroutine test_common()
  integer :: i(2)
  integer :: j_in_equiv
  integer :: not_in_equiv
  equivalence (i(2),j_in_equiv)
  common /x/ i, not_in_equiv
  call bar()
contains
 subroutine bar()
    print *, j_in_equiv, not_in_equiv
 end subroutine
end subroutine
! CHECK-LABEL: func.func private @_QFtest_commonPbar() attributes {fir.internal_proc, llvm.linkage = #llvm.linkage<internal>} {
! CHECK:  %[[VAL_0:.*]] = fir.address_of(@x_) : !fir.ref<!fir.array<12xi8>>
! CHECK:  %[[VAL_1:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.array<12xi8>>) -> !fir.ref<!fir.array<?xi8>>
! CHECK:  %[[VAL_2:.*]] = arith.constant 4 : index
! CHECK:  %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_2]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
! CHECK:  %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<i8>) -> !fir.ptr<i32>
! CHECK:  %[[VAL_5:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.array<12xi8>>) -> !fir.ref<!fir.array<?xi8>>
! CHECK:  %[[VAL_6:.*]] = arith.constant 8 : index
! CHECK:  %[[VAL_7:.*]] = fir.coordinate_of %[[VAL_5]], %[[VAL_6]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
! CHECK:  %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<i8>) -> !fir.ref<i32>

subroutine saved_equiv()
  integer, save :: i(2)
  integer, save :: j_in_equiv
  integer, save :: not_in_equiv
  equivalence (i(2),j_in_equiv)
  call bar()
contains
 subroutine bar()
    print *, j_in_equiv, not_in_equiv
 end subroutine
end subroutine
! CHECK-LABEL: func.func private @_QFsaved_equivPbar() attributes {fir.internal_proc, llvm.linkage = #llvm.linkage<internal>} {
! CHECK:  %[[VAL_0:.*]] = fir.address_of(@_QFsaved_equivEi) : !fir.ref<!fir.array<8xi8>>
! CHECK:  %[[VAL_1:.*]] = arith.constant 4 : index
! CHECK:  %[[VAL_2:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_1]] : (!fir.ref<!fir.array<8xi8>>, index) -> !fir.ref<i8>
! CHECK:  %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<i8>) -> !fir.ptr<i32>
! CHECK:  %[[VAL_4:.*]] = fir.address_of(@_QFsaved_equivEnot_in_equiv) : !fir.ref<i32>

subroutine mixed_capture()
  integer, save :: saved_i
  integer, save :: saved_j
  equivalence (saved_i, saved_j)
  integer :: i
  integer :: j
  equivalence (i,j)
  call bar()
contains
 subroutine bar()
    call test(saved_j, j)
 end subroutine
end subroutine
! CHECK-LABEL: func.func private @_QFmixed_capturePbar(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<tuple<!fir.ref<i32>>> {fir.host_assoc}) attributes {fir.internal_proc, llvm.linkage = #llvm.linkage<internal>} {
! CHECK:  %[[VAL_1:.*]] = fir.address_of(@_QFmixed_captureEsaved_i) : !fir.ref<!fir.array<4xi8>>
! CHECK:  %[[VAL_2:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_2]] : (!fir.ref<!fir.array<4xi8>>, index) -> !fir.ref<i8>
! CHECK:  %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<i8>) -> !fir.ptr<i32>
! CHECK:  %[[VAL_5:.*]] = arith.constant 0 : i32
! CHECK:  %[[VAL_6:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_5]] : (!fir.ref<tuple<!fir.ref<i32>>>, i32) -> !fir.llvm_ptr<!fir.ref<i32>>
! CHECK:  %[[VAL_7:.*]] = fir.load %[[VAL_6]] : !fir.llvm_ptr<!fir.ref<i32>>
! CHECK:  %[[VAL_9:.*]] = fir.convert %[[VAL_4]] : (!fir.ptr<i32>) -> !fir.ref<i32>
! CHECK:  fir.call @_QPtest(%[[VAL_9]], %[[VAL_7]]) {{.*}} : (!fir.ref<i32>, !fir.ref<i32>) -> ()
