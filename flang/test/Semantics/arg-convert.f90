!RUN: %flang_fc1 -fdebug-unparse  %s  2>&1 | FileCheck %s
!Ensure that argument conversion does not take place when the procedure
!interface is implicit at the point of call, even when the interface
!is known due because the procedure's definition is in the same source file.

subroutine test
!CHECK: warning: If the procedure's interface were explicit, this reference would be in error
!CHECK: because: Actual argument type 'INTEGER(8)' is not compatible with dummy argument type 'INTEGER(4)'
!CHECK: CALL samesourcefile((1_8))
  call sameSourceFile((1_8))
!CHECK: CALL somewhereelse((2_8))
  call somewhereElse((2_8))
end

subroutine sameSourceFile(n)
end
