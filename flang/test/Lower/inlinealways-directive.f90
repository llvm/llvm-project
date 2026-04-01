! Check the appropriate flags are added to inline functions when inlinealways is used

! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir %s 2>&1 | FileCheck %s --check-prefix=CHECK-WARN

subroutine test_function()
  !DIR$ INLINEALWAYS test_function
end subroutine

subroutine test_function2()
end subroutine

subroutine test_function3()
  !DIR$ INLINEALWAYS wrong_func
end subroutine

subroutine test()
  !DIR$ INLINEALWAYS
  call test_function2()
end subroutine

! CHECK: func.func @_QPtest_function() attributes {llvm.always_inline} {
! CHECK: fir.call @_QPtest_function2() fastmath<contract> {inline_attr = #fir.inline_attrs<always_inline>} : () -> ()

! CHECK-WARN:      warning: loc({{.*}}inlinealways-directive.f90{{.*}}): Directive Ignored:
! CHECK-WARN-SAME: INLINEALWAYS directive function name 'wrong_func' does not match the function 'test_function3' where this is declared.
