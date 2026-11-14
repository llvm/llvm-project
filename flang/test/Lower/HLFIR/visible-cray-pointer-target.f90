! RUN: bbc -emit-hlfir -o - -I nowhere %s | FileCheck %s

subroutine visible_association()
  real :: pointee, associated, unrelated
  integer(8) :: ptr
  pointer(ptr, pointee)
  ptr = loc(associated)
  pointee = 1.0
  print *, associated, unrelated
end

! CHECK-LABEL: func.func @_QPvisible_association()
! CHECK-DAG: %[[ASSOCIATED_ALLOC:.*]] = fir.alloca f32 <{bindc_name = "associated", uniq_name = "_QFvisible_associationEassociated"}> {fir.target}
! CHECK-DAG: %[[ASSOCIATED:.*]]:2 = hlfir.declare %[[ASSOCIATED_ALLOC]] uniq_name("_QFvisible_associationEassociated") fortran_attrs<target>
! CHECK-DAG: fir.alloca f32 <{bindc_name = "unrelated", uniq_name = "_QFvisible_associationEunrelated"}>{{$}}

subroutine nested_association(flag)
  logical :: flag
  real :: pointee, associated
  integer(8) :: ptr
  pointer(ptr, pointee)
  if (flag) then
    ptr = loc(associated)
  end if
end

! CHECK-LABEL: func.func @_QPnested_association(
! CHECK-DAG: %[[NESTED_ALLOC:.*]] = fir.alloca f32 <{bindc_name = "associated", uniq_name = "_QFnested_associationEassociated"}> {fir.target}
! CHECK-DAG: hlfir.declare %[[NESTED_ALLOC]] uniq_name("_QFnested_associationEassociated") fortran_attrs<target>

subroutine ordinary_loc()
  real :: object
  integer(8) :: address
  address = loc(object)
end

! CHECK-LABEL: func.func @_QPordinary_loc()
! CHECK: fir.alloca f32 <{bindc_name = "object", uniq_name = "_QFordinary_locEobject"}>{{$}}

subroutine hidden_association(ptr, associated)
  real :: pointee, associated
  integer(8) :: ptr
  pointer(ptr, pointee)
  call associate_target_to_cray_ptr(ptr, associated)
end

! CHECK-LABEL: func.func @_QPhidden_association(
! CHECK: hlfir.declare %{{.*}} dummy_scope %{{.*}} uniq_name("_QFhidden_associationEassociated") :

subroutine loc_plus_offset()
  real :: pointee, associated(2), unrelated
  integer(8) :: ptr
  pointer(ptr, pointee)
  ptr = loc(associated) + 4
  print *, associated, unrelated
end

! CHECK-LABEL: func.func @_QPloc_plus_offset()
! CHECK-DAG: hlfir.declare %{{.*}} uniq_name("_QFloc_plus_offsetEassociated") fortran_attrs<target>
! CHECK-DAG: hlfir.declare %{{.*}} uniq_name("_QFloc_plus_offsetEunrelated") :

subroutine loc_actual_argument()
  real :: pointee, associated
  integer(8) :: ptr
  pointer(ptr, pointee)
  call set_cray_ptr(ptr, loc(associated))
end

! CHECK-LABEL: func.func @_QPloc_actual_argument()
! CHECK: hlfir.declare %{{.*}} uniq_name("_QFloc_actual_argumentEassociated") fortran_attrs<target>

subroutine loc_through_integer_variable()
  real :: pointee, associated
  integer(8) :: ptr, address
  pointer(ptr, pointee)
  address = loc(associated)
  ptr = address
end

! CHECK-LABEL: func.func @_QPloc_through_integer_variable()
! CHECK: hlfir.declare %{{.*}} uniq_name("_QFloc_through_integer_variableEassociated") fortran_attrs<target>

subroutine equivalence_partner()
  real :: pointee, associated(10), partner(10), unrelated(10)
  equivalence (associated, partner)
  integer(8) :: ptr
  pointer(ptr, pointee)
  ptr = loc(associated(5))
  print *, partner, unrelated
end

! CHECK-LABEL: func.func @_QPequivalence_partner()
! CHECK-DAG: hlfir.declare %{{.*}} storage(%{{.*}}[0]) uniq_name("_QFequivalence_partnerEassociated") fortran_attrs<target>
! CHECK-DAG: hlfir.declare %{{.*}} storage(%{{.*}}[0]) uniq_name("_QFequivalence_partnerEpartner") fortran_attrs<target>
! CHECK-DAG: hlfir.declare %{{.*}} uniq_name("_QFequivalence_partnerEunrelated") :

subroutine several_equivalence_sets()
  real :: pointee, associated(2), left, right
  equivalence (associated(1), left), (associated(2), right)
  integer(8) :: ptr
  pointer(ptr, pointee)
  ptr = loc(associated(2))
  print *, left, right
end

! CHECK-LABEL: func.func @_QPseveral_equivalence_sets()
! CHECK-DAG: hlfir.declare %{{.*}} storage(%{{.*}}[0]) uniq_name("_QFseveral_equivalence_setsEassociated") fortran_attrs<target>
! CHECK-DAG: hlfir.declare %{{.*}} storage(%{{.*}}[0]) uniq_name("_QFseveral_equivalence_setsEleft") fortran_attrs<target>
! CHECK-DAG: hlfir.declare %{{.*}} storage(%{{.*}}[4]) uniq_name("_QFseveral_equivalence_setsEright") fortran_attrs<target>

subroutine common_block_partner()
  real :: pointee, associated, partner, unrelated
  common /partners/ associated, partner
  common /others/ unrelated
  integer(8) :: ptr
  pointer(ptr, pointee)
  ptr = loc(associated)
  print *, partner, unrelated
end

! CHECK-LABEL: func.func @_QPcommon_block_partner()
! CHECK-DAG: hlfir.declare %{{.*}} storage(%{{.*}}[0]) uniq_name("_QFcommon_block_partnerEassociated") fortran_attrs<target>
! CHECK-DAG: hlfir.declare %{{.*}} storage(%{{.*}}[4]) uniq_name("_QFcommon_block_partnerEpartner") fortran_attrs<target>
! CHECK-DAG: hlfir.declare %{{.*}} storage(%{{.*}}[0]) uniq_name("_QFcommon_block_partnerEunrelated") :

subroutine equivalence_into_common_block()
  real :: pointee, member(10), partner(10), associated(5)
  common /extended/ member, partner
  equivalence (associated(1), member(6))
  integer(8) :: ptr
  pointer(ptr, pointee)
  ptr = loc(associated)
  print *, partner
end

! CHECK-LABEL: func.func @_QPequivalence_into_common_block()
! CHECK-DAG: hlfir.declare %{{.*}} storage(%{{.*}}[20]) uniq_name("_QFequivalence_into_common_blockEassociated") fortran_attrs<target>
! CHECK-DAG: hlfir.declare %{{.*}} storage(%{{.*}}[0]) uniq_name("_QFequivalence_into_common_blockEmember") fortran_attrs<target>
! CHECK-DAG: hlfir.declare %{{.*}} storage(%{{.*}}[40]) uniq_name("_QFequivalence_into_common_blockEpartner") fortran_attrs<target>

subroutine associate_name()
  real :: pointee, associated(10)
  integer(8) :: ptr
  pointer(ptr, pointee)
  associate (alias => associated(5))
    ptr = loc(alias)
  end associate
  print *, associated
end

! CHECK-LABEL: func.func @_QPassociate_name()
! CHECK: hlfir.declare %{{.*}} uniq_name("_QFassociate_nameEassociated") fortran_attrs<target>

subroutine assumed_shape_dummy(associated)
  real :: pointee, associated(:)
  integer(8) :: ptr
  pointer(ptr, pointee)
  ptr = loc(associated(5))
end

! CHECK-LABEL: func.func @_QPassumed_shape_dummy(
! CHECK: hlfir.declare %{{.*}} dummy_scope %{{.*}} arg 1 uniq_name("_QFassumed_shape_dummyEassociated") fortran_attrs<target>

subroutine assumed_type_dummy(associated)
  type(*) :: associated
  real :: pointee
  integer(8) :: ptr
  pointer(ptr, pointee)
  ptr = loc(associated)
end

! CHECK-LABEL: func.func @_QPassumed_type_dummy(
! CHECK: hlfir.declare %{{.*}} dummy_scope %{{.*}} arg 1 uniq_name("_QFassumed_type_dummyEassociated") fortran_attrs<target>

subroutine host_association()
  real :: pointee, associated
  integer(8) :: ptr
  pointer(ptr, pointee)
  call associate_in_internal_procedure()
  associated = pointee
contains
  subroutine associate_in_internal_procedure()
    ptr = loc(associated)
  end subroutine
end

! CHECK-LABEL: func.func @_QPhost_association()
! CHECK: hlfir.declare %{{.*}} uniq_name("_QFhost_associationEassociated") fortran_attrs<target, internal_assoc>
! CHECK-LABEL: func.func private @_QFhost_associationPassociate_in_internal_procedure(
! CHECK: hlfir.declare %{{.*}} uniq_name("_QFhost_associationEassociated") fortran_attrs<target, host_assoc>

module cray_pointer_module
  real :: module_pointee
  integer(8) :: module_ptr
  pointer(module_ptr, module_pointee)
contains
  subroutine associate_module_pointer()
    real :: associated
    module_ptr = loc(associated)
  end subroutine

  subroutine use_module_pointee_only()
    real :: associated
    call set_cray_ptr_address(loc(associated))
    associated = module_pointee
  end subroutine
end module

! CHECK-LABEL: func.func @_QMcray_pointer_modulePassociate_module_pointer()
! CHECK: hlfir.declare %{{.*}} uniq_name("_QMcray_pointer_moduleFassociate_module_pointerEassociated") fortran_attrs<target>
! CHECK-LABEL: func.func @_QMcray_pointer_modulePuse_module_pointee_only()
! CHECK: hlfir.declare %{{.*}} uniq_name("_QMcray_pointer_moduleFuse_module_pointee_onlyEassociated") fortran_attrs<target>

subroutine pointer_in_block()
  real :: associated
  block
    real :: pointee
    integer(8) :: ptr
    pointer(ptr, pointee)
    ptr = loc(associated)
  end block
end

! CHECK-LABEL: func.func @_QPpointer_in_block()
! CHECK: hlfir.declare %{{.*}} uniq_name("_QFpointer_in_blockEassociated") fortran_attrs<target>

module target_module
  real :: global_target(10)
end module

subroutine internal_procedure_uses_host_pointee()
  use target_module
  real :: pointee(10)
  integer(8) :: ptr
  pointer(ptr, pointee)
  ptr = loc(global_target)
  call copy_from_pointee()
contains
  subroutine copy_from_pointee()
    global_target(2:10) = pointee(1:9)
  end subroutine
end

! CHECK-LABEL: func.func @_QPinternal_procedure_uses_host_pointee()
! CHECK: hlfir.declare %{{.*}} uniq_name("_QMtarget_moduleEglobal_target") fortran_attrs<target>
! CHECK-LABEL: func.func private @_QFinternal_procedure_uses_host_pointeePcopy_from_pointee(
! CHECK: hlfir.declare %{{.*}} uniq_name("_QMtarget_moduleEglobal_target") fortran_attrs<target>
