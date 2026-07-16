! Separate-compilation user-defined operator declare reduction: the declare
! reduction lives in a module compiled on its own to a .mod, and a different
! translation unit uses the module and names the operator in a reduction clause.
! The module is not a unit of the consumer TU, so the primary lowering pass never
! emits its omp.declare_reduction op; lowering the consumer must materialize the
! imported reduction (mirroring imported declare mappers) for the clause to bind,
! rather than hit a clean "not yet implemented" TODO.
! https://github.com/llvm/llvm-project/issues/207255

! RUN: rm -rf %t && split-file %s %t && cd %t

! Case 1 (plain USE). The materialized op name must be byte-identical to the
! clause reference; the shared [[RED1]] capture pins that. A .mod re-parse that
! resolved to the operator generic instead of the reduction symbol would name a
! different op and fall through to the CHECK-NOT TODO.
! RUN: %flang_fc1 -emit-hlfir -fopenmp plain.mod.f90 -o - > /dev/null
! RUN: %flang_fc1 -emit-hlfir -fopenmp plain.use.f90 -o - | FileCheck plain.use.f90

! Case 2 (renamed operator, .local. => .remote.). The reduction is still named
! from the source module operator, so the clause reference and the materialized
! op agree.
! RUN: %flang_fc1 -emit-hlfir -fopenmp rename.mod.f90 -o - > /dev/null
! RUN: %flang_fc1 -emit-hlfir -fopenmp rename.use.f90 -o - | FileCheck rename.use.f90

! Case 3 (collision). Two modules define operator(.remote.) with a reduction on
! the same type. Imported under different local names, they must lower to two
! distinct ops keyed by their source module, not a single shared op that would
! run one variable through the other's combiner.
! RUN: %flang_fc1 -emit-hlfir -fopenmp collide_ty.mod.f90 -o - > /dev/null
! RUN: %flang_fc1 -emit-hlfir -fopenmp collide_add.mod.f90 -o - > /dev/null
! RUN: %flang_fc1 -emit-hlfir -fopenmp collide_mul.mod.f90 -o - > /dev/null
! RUN: %flang_fc1 -emit-hlfir -fopenmp collide.use.f90 -o - | FileCheck collide.use.f90

!--- plain.mod.f90
module red_plain
  type :: t
    integer :: val = 0
  end type
  interface operator(.plus.)
    module procedure add_t
  end interface
  !$omp declare reduction(.plus.:t:omp_out%val=omp_out%val+omp_in%val) &
  !$omp   initializer(omp_priv=t(0))
contains
  type(t) function add_t(a, b)
    type(t), intent(in) :: a, b
    add_t%val = a%val + b%val
  end function
end module

!--- plain.use.f90
! CHECK: omp.declare_reduction @[[RED1:_QQ[A-Za-z0-9_.]*op\.plus\.[A-Za-z0-9_.]*]] : !fir.ref
! CHECK: omp.wsloop
! CHECK-SAME: reduction(byref @[[RED1]]
! CHECK-NOT: not yet implemented
program main
  use red_plain
  type(t) :: x
  integer :: i
  x = t(0)
  !$omp parallel do reduction(.plus.:x)
  do i = 1, 100
    x%val = x%val + 1
  end do
  print *, x%val
end program

!--- rename.mod.f90
module red_rename
  type :: t
    integer :: val = 0
  end type
  interface operator(.remote.)
    module procedure add_t
  end interface
  !$omp declare reduction(.remote.:t:omp_out%val=omp_out%val+omp_in%val) &
  !$omp   initializer(omp_priv=t(0))
contains
  type(t) function add_t(a, b)
    type(t), intent(in) :: a, b
    add_t%val = a%val + b%val
  end function
end module

!--- rename.use.f90
! CHECK: omp.declare_reduction @[[RED2:_QQ[A-Za-z0-9_.]*op\.remote\.[A-Za-z0-9_.]*]] : !fir.ref
! CHECK: omp.wsloop
! CHECK-SAME: reduction(byref @[[RED2]]
! CHECK-NOT: not yet implemented
program main
  use red_rename, only: t, operator(.local.) => operator(.remote.)
  type(t) :: x
  integer :: i
  x = t(0)
  !$omp parallel do reduction(.local.:x)
  do i = 1, 100
    x%val = x%val + 1
  end do
  print *, x%val
end program

!--- collide_ty.mod.f90
module red_ty
  type :: t
    integer :: val = 0
  end type
end module

!--- collide_add.mod.f90
module red_addmod
  use red_ty
  interface operator(.remote.)
    module procedure add_t
  end interface
  !$omp declare reduction(.remote.:t:omp_out%val=omp_out%val+omp_in%val) &
  !$omp   initializer(omp_priv=t(0))
contains
  type(t) function add_t(a, b)
    type(t), intent(in) :: a, b
    add_t%val = a%val + b%val
  end function
end module

!--- collide_mul.mod.f90
module red_mulmod
  use red_ty
  interface operator(.remote.)
    module procedure mul_t
  end interface
  !$omp declare reduction(.remote.:t:omp_out%val=omp_out%val*omp_in%val) &
  !$omp   initializer(omp_priv=t(1))
contains
  type(t) function mul_t(a, b)
    type(t), intent(in) :: a, b
    mul_t%val = a%val * b%val
  end function
end module

!--- collide.use.f90
! The two ops are distinguished by their source module (addmod vs mulmod) in the
! mangled name, so pinning one op to each module proves they are distinct.
! CHECK-DAG: omp.declare_reduction @{{_QQ[A-Za-z0-9_.]*addmod[A-Za-z0-9_.]*op\.remote\.[A-Za-z0-9_.]*}} : !fir.ref
! CHECK-DAG: omp.declare_reduction @{{_QQ[A-Za-z0-9_.]*mulmod[A-Za-z0-9_.]*op\.remote\.[A-Za-z0-9_.]*}} : !fir.ref
! CHECK-NOT: not yet implemented
program main
  use red_ty, only: t
  use red_addmod, only: operator(.addop.) => operator(.remote.)
  use red_mulmod, only: operator(.mulop.) => operator(.remote.)
  type(t) :: x, y
  integer :: i
  x = t(0)
  !$omp parallel do reduction(.addop.:x)
  do i = 1, 5
    x%val = x%val + i
  end do
  y = t(1)
  !$omp parallel do reduction(.mulop.:y)
  do i = 1, 5
    y%val = y%val * i
  end do
  print *, x%val, y%val
end program
