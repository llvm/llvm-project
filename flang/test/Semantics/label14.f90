! Tests implemented for this standard
! 11.1.4 - 4 It is permissible to branch to and end-block-stmt only withinh its
!            Block Construct

! RUN: %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s
! CHECK: Label '20' is not in scope

subroutine s1
  block
    goto (10) 1
10  end block

  block
20  end block
end

subroutine s2
  block
    goto (20) 1
10  end block

  block
20  end block
end

subroutine s3
  block
    block
      goto (10) 1
10  end block
20  end block
end

subroutine s4
  block
    block
      goto (20) 1
10  end block
20  end block
end
