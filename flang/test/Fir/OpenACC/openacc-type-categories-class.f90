module mm
  type, public :: polyty
    real :: field
  end type
contains
  subroutine init(this)
    class(polyty), intent(inout) :: this
    !$acc enter data copyin(this, this%field)
  end subroutine
end module

! RUN: bbc -fopenacc -emit-hlfir %s -o - | fir-opt -pass-pipeline='builtin.module(test-fir-openacc-interfaces)' --mlir-disable-threading 2>&1 | FileCheck %s
! CHECK: Visiting: {{.*}} acc.copyin {{.*}} {name = "this", structured = false}
! CHECK: Mappable: !fir.class<!fir.type<_QMmmTpolyty{field:f32}>>
! CHECK: Type category: composite
! CHECK: Visiting: {{.*}} acc.copyin {{.*}} {name = "this%field", structured = false}
! CHECK: Pointer-like: !fir.ref<f32>
! CHECK: Type category: composite
