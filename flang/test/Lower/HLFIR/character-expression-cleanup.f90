! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

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
