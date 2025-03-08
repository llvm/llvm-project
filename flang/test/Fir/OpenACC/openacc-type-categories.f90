! RUN: bbc -fopenacc -emit-hlfir %s -o - | fir-opt -pass-pipeline='builtin.module(test-fir-openacc-interfaces)' --mlir-disable-threading 2>&1 | FileCheck %s

program main
  real :: scalar
  real, allocatable :: scalaralloc
  type tt
    real :: field
    real :: fieldarray(10)
  end type tt
  type(tt) :: ttvar
  real :: arrayconstsize(10)
  real, allocatable :: arrayalloc(:)
  complex :: complexvar
  character*1 :: charvar

  !$acc enter data copyin(scalar, scalaralloc, ttvar, arrayconstsize, arrayalloc)
  !$acc enter data copyin(complexvar, charvar, ttvar%field, ttvar%fieldarray, arrayconstsize(1))
end program

! CHECK: Visiting: {{.*}} acc.copyin {{.*}} {name = "scalar", structured = false}
! CHECK: Pointer-like: !fir.ref<f32>
! CHECK: Type category: scalar
! CHECK: Visiting: {{.*}} acc.copyin {{.*}} {name = "scalaralloc", structured = false}
! CHECK: Pointer-like: !fir.ref<!fir.box<!fir.heap<f32>>>
! CHECK: Type category: nonscalar
! CHECK: Visiting: {{.*}} acc.copyin {{.*}} {name = "ttvar", structured = false}
! CHECK: Pointer-like: !fir.ref<!fir.type<_QFTtt{field:f32,fieldarray:!fir.array<10xf32>}>>
! CHECK: Type category: composite
! CHECK: Visiting: {{.*}} acc.copyin {{.*}} {name = "arrayconstsize", structured = false}
! CHECK: Pointer-like: !fir.ref<!fir.array<10xf32>>
! CHECK: Type category: array
! CHECK: Visiting: {{.*}} acc.copyin {{.*}} {name = "arrayalloc", structured = false}
! CHECK: Pointer-like: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK: Type category: array
! CHECK: Visiting: {{.*}} acc.copyin {{.*}} {name = "complexvar", structured = false}
! CHECK: Pointer-like: !fir.ref<complex<f32>>
! CHECK: Type category: scalar
! CHECK: Visiting: {{.*}} acc.copyin {{.*}} {name = "charvar", structured = false}
! CHECK: Pointer-like: !fir.ref<!fir.char<1>>
! CHECK: Type category: nonscalar
! CHECK: Visiting: {{.*}} acc.copyin {{.*}} {name = "ttvar%field", structured = false}
! CHECK: Pointer-like: !fir.ref<f32>
! CHECK: Type category: composite
! CHECK: Visiting: {{.*}} acc.copyin {{.*}} {name = "ttvar%fieldarray", structured = false}
! CHECK: Pointer-like: !fir.ref<!fir.array<10xf32>>
! CHECK: Type category: array
! CHECK: Visiting: {{.*}} acc.copyin {{.*}} {name = "arrayconstsize(1)", structured = false}
! CHECK: Pointer-like: !fir.ref<!fir.array<10xf32>>
! CHECK: Type category: array
