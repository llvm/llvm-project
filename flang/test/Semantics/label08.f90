
! RUN: not %flang_fc1 -fdebug-unparse-with-symbols %s 2>&1 | FileCheck %s
! CHECK: CYCLE construct-name is not in scope
! CHECK: IF construct name unexpected
! CHECK: unnamed IF statement
! CHECK: DO construct name mismatch
! CHECK: should be

subroutine sub00(a,b,n,m)
  real a(n,m)
  real b(n,m)
  labelone: do i = 1, m
     labeltwo: do j = 1, n
50      a(i,j) = b(i,j) + 2.0
        if (n .eq. m) then
           cycle label3
        end if label3
60   end do labeltwo
  end do label1
end subroutine sub00
