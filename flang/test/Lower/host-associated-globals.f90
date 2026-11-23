! Test lowering of internal procedure host association for global variables
! A tuple function argument should not be created for associated globals, and
! instead globals should be instantiated with a fir.address_of inside the
! contained procedures.
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s --check-prefixes=CHECK,CAPTURE

module test_mod_used_in_host
  integer :: i, j_in_equiv
  integer :: not_in_equiv
  equivalence (i,j_in_equiv)
end module

! The host and module EQUIVALENCE groups both start at offset zero.
subroutine module_var()
  use test_mod_used_in_host
  integer, save :: saved_a, saved_b
  equivalence (saved_a, saved_b)
  call bar()
contains
 subroutine bar()
    print *, j_in_equiv, not_in_equiv, saved_b
 end subroutine
end subroutine
! CHECK-LABEL: func.func private @_QFmodule_varPbar()
! CHECK-DAG: %[[HOST_EQ:.*]] = fir.address_of(@_QFmodule_varEsaved_a)
! CHECK-DAG: %[[VAL_0:.*]] = fir.address_of(@_QMtest_mod_used_in_hostEi) : !fir.ref<!fir.array<4xi8>>
! CHECK:  %[[VAL_1:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_2:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_1]] : (!fir.ref<!fir.array<4xi8>>, index) -> !fir.ref<i8>
! CHECK:  %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<i8>) -> !fir.ptr<i32>
! CHECK:  %{{.*}}:2 = hlfir.declare %[[VAL_3]] storage(%[[VAL_0]][0]) uniq_name("_QMtest_mod_used_in_hostEj_in_equiv") : (!fir.ptr<i32>, !fir.ref<!fir.array<4xi8>>) -> (!fir.ptr<i32>, !fir.ptr<i32>)
! CHECK:  %[[VAL_4:.*]] = fir.address_of(@_QMtest_mod_used_in_hostEnot_in_equiv) : !fir.ref<i32>
! CHECK:  %{{.*}}:2 = hlfir.declare %[[VAL_4]] uniq_name("_QMtest_mod_used_in_hostEnot_in_equiv") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[HOST_MEMBER:.*]]:2 = hlfir.declare {{.*}} storage(%[[HOST_EQ]][0]) uniq_name("_QFmodule_varEsaved_b")
! CHECK: fir.load %[[HOST_MEMBER]]#0

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
! CHECK-LABEL: func.func private @_QFtest_commonPbar() attributes {fir.host_symbol = {{.*}}, llvm.linkage = #llvm.linkage<internal>} {
! CHECK:  %[[VAL_0:.*]] = fir.address_of(@x_) : !fir.ref<!fir.array<12xi8>>
! CHECK:  %[[VAL_2:.*]] = arith.constant 4 : index
! CHECK:  %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_2]] : (!fir.ref<!fir.array<12xi8>>, index) -> !fir.ref<i8>
! CHECK:  %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<i8>) -> !fir.ptr<i32>
! CHECK:  %{{.*}}:2 = hlfir.declare %[[VAL_4]] storage(%[[VAL_0]][4]) uniq_name("_QFtest_commonEj_in_equiv") fortran_attrs<internal_assoc> : (!fir.ptr<i32>, !fir.ref<!fir.array<12xi8>>) -> (!fir.ptr<i32>, !fir.ptr<i32>)
! CHECK:  %[[VAL_6:.*]] = arith.constant 8 : index
! CHECK:  %[[VAL_7:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_6]] : (!fir.ref<!fir.array<12xi8>>, index) -> !fir.ref<i8>
! CHECK:  %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<i8>) -> !fir.ref<i32>
! CHECK:  %{{.*}}:2 = hlfir.declare %[[VAL_8]] storage(%[[VAL_0]][8]) uniq_name("_QFtest_commonEnot_in_equiv") fortran_attrs<internal_assoc> : (!fir.ref<i32>, !fir.ref<!fir.array<12xi8>>) -> (!fir.ref<i32>, !fir.ref<i32>)

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
! CHECK-LABEL: func.func private @_QFsaved_equivPbar() attributes {fir.host_symbol = {{.*}}, llvm.linkage = #llvm.linkage<internal>} {
! CHECK:  %[[VAL_0:.*]] = fir.address_of(@_QFsaved_equivEi) : !fir.ref<!fir.array<8xi8>>
! CHECK:  %[[VAL_1:.*]] = arith.constant 4 : index
! CHECK:  %[[VAL_2:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_1]] : (!fir.ref<!fir.array<8xi8>>, index) -> !fir.ref<i8>
! CHECK:  %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<i8>) -> !fir.ptr<i32>
! CHECK:  %{{.*}}:2 = hlfir.declare %[[VAL_3]] storage(%[[VAL_0]][4]) uniq_name("_QFsaved_equivEj_in_equiv") fortran_attrs<internal_assoc> : (!fir.ptr<i32>, !fir.ref<!fir.array<8xi8>>) -> (!fir.ptr<i32>, !fir.ptr<i32>)
! CHECK:  %[[VAL_4:.*]] = fir.address_of(@_QFsaved_equivEnot_in_equiv) : !fir.ref<i32>
! CHECK:  %{{.*}}:2 = hlfir.declare %[[VAL_4]] uniq_name("_QFsaved_equivEnot_in_equiv") fortran_attrs<internal_assoc> : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

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
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<tuple<!fir.ref<i32>>> {fir.host_assoc}) attributes {fir.host_symbol = {{.*}}, llvm.linkage = #llvm.linkage<internal>} {
! CHECK:  %[[VAL_1:.*]] = fir.address_of(@_QFmixed_captureEsaved_i) : !fir.ref<!fir.array<4xi8>>
! CHECK:  %[[VAL_2:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_2]] : (!fir.ref<!fir.array<4xi8>>, index) -> !fir.ref<i8>
! CHECK:  %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<i8>) -> !fir.ptr<i32>
! CHECK:  %[[VAL_declare_saved_j:.*]]:2 = hlfir.declare %[[VAL_4]] storage(%[[VAL_1]][0]) uniq_name("_QFmixed_captureEsaved_j") fortran_attrs<internal_assoc> : (!fir.ptr<i32>, !fir.ref<!fir.array<4xi8>>) -> (!fir.ptr<i32>, !fir.ptr<i32>)
! CHECK:  %[[VAL_5:.*]] = arith.constant 0 : i32
! CHECK:  %[[VAL_6:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_5]] : (!fir.ref<tuple<!fir.ref<i32>>>, i32) -> !fir.llvm_ptr<!fir.ref<i32>>
! CHECK:  %[[VAL_7:.*]] = fir.load %[[VAL_6]] : !fir.llvm_ptr<!fir.ref<i32>>
! CHECK:  %[[VAL_declare_j:.*]]:2 = hlfir.declare %[[VAL_7]] uniq_name("_QFmixed_captureEj") fortran_attrs<host_assoc> : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:  %[[VAL_9:.*]] = fir.convert %[[VAL_declare_saved_j]]#0 : (!fir.ptr<i32>) -> !fir.ref<i32>
! CHECK:  fir.call @_QPtest(%[[VAL_9]], %[[VAL_declare_j]]#0) {{.*}} : (!fir.ref<i32>, !fir.ref<i32>) -> ()

! The sibling call reads the assignment destination through host association.
! Mark a as captured so alias analysis accounts for these indirect reads.
! CAPTURE-LABEL: func.func private @_QFarray_resultPacc()
! CAPTURE: hlfir.declare {{.*}}uniq_name("_QFarray_resultEa")
! CAPTURE-SAME: fortran_attrs<internal_assoc>
! CAPTURE-LABEL: func.func private @_QFarray_resultPg(
subroutine array_result()
  integer, save :: a(3) = [1, 2, 3]
  call acc()
  print *, a
contains
  subroutine acc()
    a = g()
  end subroutine
  function g()
    integer :: g(3)
    integer :: i
    do i = 1, 3
      g(i) = a(4-i)
    end do
  end function
end subroutine

! Both entry points bind the EQUIVALENCE members used by the siblings.
subroutine selected_stores()
  integer, save :: a(2), b, u(2), v
  equivalence (a(2), b), (u(2), v)
entry selected_stores_alt()
  a = [1, 2]
  u = [3, 4]
  call first()
  call second()
contains
  subroutine first()
    b = b + 1
  end subroutine
  subroutine second()
    v = v + 2
  end subroutine
end subroutine
! CHECK-LABEL: func.func @_QPselected_stores()
! CHECK-DAG: %[[HOST_A:.*]] = fir.address_of(@_QFselected_storesEa)
! CHECK-DAG: %[[HOST_U:.*]] = fir.address_of(@_QFselected_storesEu)
! CHECK-DAG: hlfir.declare {{.*}} storage(%[[HOST_A]][4]) uniq_name("_QFselected_storesEb")
! CHECK-DAG: hlfir.declare {{.*}} storage(%[[HOST_U]][4]) uniq_name("_QFselected_storesEv")
! CHECK: fir.call @_QFselected_storesPfirst()
! CHECK: fir.call @_QFselected_storesPsecond()
! CHECK-LABEL: func.func @_QPselected_stores_alt()
! CHECK-DAG: %[[ENTRY_A:.*]] = fir.address_of(@_QFselected_storesEa)
! CHECK-DAG: %[[ENTRY_U:.*]] = fir.address_of(@_QFselected_storesEu)
! CHECK-DAG: hlfir.declare {{.*}} storage(%[[ENTRY_A]][4]) uniq_name("_QFselected_storesEb")
! CHECK-DAG: hlfir.declare {{.*}} storage(%[[ENTRY_U]][4]) uniq_name("_QFselected_storesEv")
! CHECK: fir.call @_QFselected_storesPfirst()
! CHECK: fir.call @_QFselected_storesPsecond()
! CHECK-LABEL: func.func private @_QFselected_storesPfirst()
! CHECK: %[[FIRST_STORE:.*]] = fir.address_of(@_QFselected_storesEa)
! CHECK: %[[FIRST_ALIAS:.*]]:2 = hlfir.declare {{.*}} storage(%[[FIRST_STORE]][4]) uniq_name("_QFselected_storesEb")
! CHECK: hlfir.assign {{.*}} to %[[FIRST_ALIAS]]#0
! CHECK: return
! CHECK-LABEL: func.func private @_QFselected_storesPsecond()
! CHECK: %[[SECOND_STORE:.*]] = fir.address_of(@_QFselected_storesEu)
! CHECK: %[[SECOND_ALIAS:.*]]:2 = hlfir.declare {{.*}} storage(%[[SECOND_STORE]][4]) uniq_name("_QFselected_storesEv")
! CHECK: hlfir.assign {{.*}} to %[[SECOND_ALIAS]]#0
! CHECK: return

! MAIN's internal procedure accesses the saved alias through global storage.
program main_stores
  integer, save :: a(2), b, u(2), v
  equivalence (a(2), b), (u(2), v)
  a = [1, 2]
  call child()
contains
  subroutine child()
    b = b + 1
  end subroutine
end program
! CHECK-LABEL: func.func private @_QFPchild()
! CHECK: %[[MAIN_STORE:.*]] = fir.address_of(@_QFEa)
! CHECK: %[[MAIN_ALIAS:.*]]:2 = hlfir.declare {{.*}} storage(%[[MAIN_STORE]][4]) uniq_name("_QFEb")
! CHECK: hlfir.assign {{.*}} to %[[MAIN_ALIAS]]#0
! CHECK: return

! A zero-length local block must not replace a captured global's storage.
subroutine zero_stores()
  character(0), save :: a, b
  character(0) :: u, v
  equivalence (a,b), (u,v)
  call child()
contains
  subroutine child()
    call sink(a)
  end subroutine
end subroutine
! CHECK-LABEL: func.func private @_QFzero_storesPchild()
! CHECK: %[[ZERO_STORE:.*]] = fir.address_of(@_QFzero_storesE{{a|b}})
! CHECK: %[[ZERO_ALIAS:.*]]:2 = hlfir.declare {{.*}} storage(%[[ZERO_STORE]][0]) uniq_name("_QFzero_storesEa")
! CHECK: %[[ZERO_REF:.*]] = fir.convert %[[ZERO_ALIAS]]#0
! CHECK: %[[ZERO_ARG:.*]] = fir.emboxchar %[[ZERO_REF]],
! CHECK: fir.call @_QPsink(%[[ZERO_ARG]])

! Bind the captured scalar even when no EQUIVALENCE member is captured.
subroutine scalar_capture()
  integer, save :: x, a, z
  equivalence (a,z)
  call child()
contains
  subroutine child()
    x = 1
  end subroutine
end subroutine
! CHECK-LABEL: func.func private @_QFscalar_capturePchild()
! CHECK: %[[SCALAR_ADDR:.*]] = fir.address_of(@_QFscalar_captureEx)
! CHECK: %[[SCALAR:.*]]:2 = hlfir.declare %[[SCALAR_ADDR]] uniq_name("_QFscalar_captureEx")
! CHECK: hlfir.assign {{.*}} to %[[SCALAR]]#0
! CHECK: return
