! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

subroutine concat_cleanup(a, b, c)
  character(*) :: a, b, c
  c = (a // b)
end subroutine
! CHECK-LABEL: func.func @_QPconcat_cleanup
! CHECK: %[[EXPR:.*]] = hlfir.concat
! CHECK: %[[ALIAS:.*]] = hlfir.no_reassoc %[[EXPR]]
! CHECK: hlfir.assign %[[ALIAS]]
! CHECK: hlfir.destroy %[[EXPR]]
! CHECK-NOT: hlfir.destroy
! CHECK: return

subroutine statement_function_cleanup(a, b)
  character(*) :: a, b
  character(len(a)) :: f, x
  f(x) = x
  b = f(a)
end subroutine
! CHECK-LABEL: func.func @_QPstatement_function_cleanup
! CHECK: %[[EXPR:.*]] = hlfir.set_length
! CHECK: hlfir.assign %[[EXPR]]
! CHECK: hlfir.destroy %[[EXPR]]
! CHECK-NOT: hlfir.destroy
! CHECK: return

subroutine parens_cleanup(a, b)
  character(*) :: a, b
  b = (a)
end subroutine
! CHECK-LABEL: func.func @_QPparens_cleanup
! CHECK: %[[EXPR:.*]] = hlfir.as_expr
! CHECK: hlfir.assign %[[EXPR]]
! CHECK: hlfir.destroy %[[EXPR]]
! CHECK-NOT: hlfir.destroy
! CHECK: return

subroutine kind_cleanup(a, b)
  character(*) :: a
  character(kind=4, len=*) :: b
  b = a
end subroutine
! CHECK-LABEL: func.func @_QPkind_cleanup
! CHECK: fir.allocmem !fir.char<4,?>
! CHECK: fir.char_convert
! CHECK: %[[EXPR:.*]] = hlfir.as_expr {{.*}} move
! CHECK: hlfir.assign %[[EXPR]]
! CHECK: hlfir.destroy %[[EXPR]]
! CHECK: return

! Cleanups of the enclosing statement do not create cleanup regions in the
! array constructor elementals.
subroutine array_ctor_outer_cleanup(s, n, y)
  character(*) :: s
  integer :: n, i, y(n)
  y = ichar(trim(s)) + [(i, i=1,n)]
end subroutine
! CHECK-LABEL: func.func @_QParray_ctor_outer_cleanup
! CHECK: hlfir.yield_element %{{.*}} : i32
! CHECK-NEXT: }
! CHECK: hlfir.yield_element %{{.*}} : i32
! CHECK-NEXT: }
! CHECK: hlfir.destroy
! CHECK: return

! The SET_LENGTH of a component initializer is bypassed, so it is only
! destroyed, which canonicalization cleans up.
subroutine structure_constructor_set_length(name, x)
  type t
    character(8) :: c
  end type
  character(*) :: name
  type(t) :: x
  x = t(name)
end subroutine
! CHECK-LABEL: func.func @_QPstructure_constructor_set_length
! CHECK: %[[EXPR:.*]] = hlfir.set_length
! CHECK-NOT: %[[EXPR]]
! CHECK: hlfir.destroy %[[EXPR]]
! CHECK: return
