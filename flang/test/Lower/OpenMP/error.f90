! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 -o - %s 2>&1 | FileCheck %s

! Lowering of `!$omp error at(execution)` to the `omp.error` operation. The
! `at(compilation)` form is handled entirely in semantics and produces no IR.
! Severity is `warning` or `fatal` (default `fatal`); the message is an optional
! constant string that is omitted when absent. A non-constant message is lowered
! to a null-terminated runtime string and passed as an operand.

! CHECK-LABEL: func.func @_QQmain()
program p
  ! CHECK: omp.error severity(warning) message("a warning")
  !$omp error at(execution) severity(warning) message("a warning")

  ! CHECK: omp.error severity(fatal) message("fatal error")
  !$omp error at(execution) severity(fatal) message("fatal error")

  ! No MESSAGE clause; severity defaults to `fatal`.
  ! CHECK: omp.error severity(fatal)
  !$omp error at(execution)

  ! An explicit empty MESSAGE string is lowered to an empty constant `message`
  ! attribute (distinct from an absent clause) at the HLFIR level.
  ! CHECK: omp.error severity(warning) message("")
  !$omp error at(execution) severity(warning) message("")
end program p

! A non-constant MESSAGE is only known at run time, so it is lowered to a
! null-terminated copy in memory and passed as the `message_expr` operand
! instead of as a constant attribute.
! CHECK-LABEL: func.func @_QPf_runtime_message
subroutine f_runtime_message
  character(len=16) :: msg
  msg = "runtime warning"
  ! CHECK: omp.error severity(warning) message_expr(%{{.*}} : !fir.ref<!fir.char<1>>)
  !$omp error at(execution) severity(warning) message(msg)
end subroutine f_runtime_message

! The `at(compilation)` form (and the implicit default, which is also
! compilation) is resolved entirely in semantics and must not lower to an
! `omp.error` operation. `severity(warning)` keeps these as non-fatal
! diagnostics so compilation still succeeds.
! CHECK-LABEL: func.func @_QPf_compilation
subroutine f_compilation
  ! CHECK-NOT: omp.error
  !$omp error severity(warning) message("implicit compilation")
  !$omp error at(compilation) severity(warning) message("explicit compilation")
end subroutine f_compilation
