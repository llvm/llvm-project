! RUN: bbc -fopenacc -emit-hlfir %s -o - | fir-opt -pass-pipeline='builtin.module(test-fir-openacc-interfaces)' --mlir-disable-threading 2>&1 | FileCheck %s

module mm
  type, public :: polyty
    real :: field
  end type
contains
  subroutine init(this)
    class(polyty), intent(inout) :: this
    !$acc enter data copyin(this, this%field)
  end subroutine
  subroutine init_assumed_type(var)
    type(*), intent(inout) :: var
    !$acc enter data copyin(var)
  end subroutine
  subroutine init_unlimited(this)
    class(*), intent(inout) :: this
    !$acc enter data copyin(this)
    select type(this)
    type is(real)
      !$acc enter data copyin(this)
    class is(polyty)
      !$acc enter data copyin(this, this%field)
    end select
  end subroutine
end module

! CHECK: Visiting: {{.*}} acc.copyin {{.*}} {name = "this", structured = false}
! CHECK: Mappable: !fir.class<!fir.type<_QMmmTpolyty{field:f32}>>
! CHECK: Type category: composite
! CHECK: Visiting: {{.*}} acc.copyin {{.*}} {name = "this%field", structured = false}
! CHECK: Pointer-like and Mappable: !fir.ref<f32>
! CHECK: Type category: composite

! For unlimited polymorphic entities and assumed types - they effectively have
! no declared type. Thus the type categorizer cannot categorize it.
! CHECK: Visiting: {{.*}} = acc.copyin {{.*}} {name = "var", structured = false}
! CHECK: Pointer-like and Mappable: !fir.ref<none>
! CHECK: Type category: uncategorized
! CHECK: Visiting: {{.*}} = acc.copyin {{.*}} {name = "this", structured = false}
! CHECK: Mappable: !fir.class<none>
! CHECK: Type category: uncategorized

! TODO: After using select type - the appropriate type category should be
! possible. Add the rest of the test once OpenACC lowering correctly handles
! unlimited polymorhic.
