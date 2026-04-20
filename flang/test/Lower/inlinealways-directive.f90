! Check the appropriate flags are added to inline functions when inlinealways is used, or ignored when the name is incorrect.

! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

subroutine test_function()
  !DIR$ INLINEALWAYS test_function
end subroutine
! CHECK: func.func @_QPtest_function() attributes {llvm.always_inline} {

subroutine test_function2()
end subroutine

subroutine test_function3()
  !DIR$ INLINEALWAYS wrong_func
end subroutine
! CHECK: func.func @_QPtest_function2() {

subroutine test()
  !DIR$ INLINEALWAYS
  call test_function2()
end subroutine
! CHECK: fir.call @_QPtest_function2() fastmath<contract> {inline_attr = #fir.inline_attrs<always_inline>} : () -> ()
