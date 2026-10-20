! Test -skip-external-rtti-definition option

!RUN: rm -rf %t && mkdir -p %t
!RUN: %flang_fc1 -fsyntax-only -DSTEP=1 -J%t %s
!RUN: %flang_fc1 -emit-llvm -J%t  %s -o - | FileCheck %s -check-prefix=LINKONCE
!RUN: %flang_fc1 -emit-llvm -J%t -mllvm -skip-external-rtti-definition %s -o - | FileCheck %s -check-prefix=EXTERNAL

#if STEP == 1
module module_external_type_definition
 type t1
 end type
end module
#else

module module_same_unit_type_definition
 type t2
 end type
end module

subroutine test
  use module_external_type_definition
  use module_same_unit_type_definition
  interface
  subroutine needs_descriptor(x)
    class(*) :: x
  end subroutine
  end interface
  type(t1) :: x1
  type(t2) :: x2
  call needs_descriptor(x1)
  call needs_descriptor(x2)
end subroutine

#endif

! LINKONCE-DAG: @_QMmodule_external_type_definitionEXnXt1 = linkonce_odr constant [2 x i8] c"t1"
! LINKONCE-DAG: @_QMmodule_external_type_definitionEXdtXt1 = linkonce_odr constant {{.*}} {
! LINKONCE-DAG: @_QMmodule_same_unit_type_definitionEXnXt2 = linkonce_odr constant [2 x i8] c"t2"
! LINKONCE-DAG: @_QMmodule_same_unit_type_definitionEXdtXt2 = linkonce_odr constant {{.*}} {

! EXTERNAL-NOT: @_QMmodule_external_type_definitionEXnXt1
! EXTERNAL: @_QMmodule_same_unit_type_definitionEXnXt2 = constant [2 x i8] c"t2"
! EXTERNAL-NOT: @_QMmodule_external_type_definitionEXnXt1
! EXTERNAL: @_QMmodule_same_unit_type_definitionEXdtXt2 = constant {{.*}} {
! EXTERNAL-NOT: @_QMmodule_external_type_definitionEXnXt1
! EXTERNAL: @_QMmodule_external_type_definitionEXdtXt1 =  external constant ptr
! EXTERNAL-NOT: @_QMmodule_external_type_definitionEXnXt1
