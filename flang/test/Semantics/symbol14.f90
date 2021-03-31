! RUN: %S/test_symbols.sh %s %t %f18
! "Bare" uses of type parameters and components

 !DEF: /MainProgram1/t1 DerivedType
 !DEF: /MainProgram1/t1/k TypeParam INTEGER(4)
 type :: t1(k)
  !REF: /MainProgram1/t1/k
  integer, kind :: k=666
  !DEF: /MainProgram1/t1/a ObjectEntity REAL(4)
  !REF: /MainProgram1/t1/k
  real :: a(k)
 end type t1
 !REF: /MainProgram1/t1
 !DEF: /MainProgram1/t2 DerivedType
 type, extends(t1) :: t2
  !DEF: /MainProgram1/t2/b ObjectEntity REAL(4)
  !REF: /MainProgram1/t1/k
  real :: b(k)
  !DEF: /MainProgram1/t2/c ObjectEntity REAL(4)
  !DEF: /MainProgram1/size INTRINSIC, PURE (Function) ProcEntity
  !REF: /MainProgram1/t1/a
  real :: c(size(a))
  !REF: /MainProgram1/t1
  !DEF: /MainProgram1/t2/x ObjectEntity TYPE(t1(k=666_4))
  type(t1) :: x
 end type t2
end program
