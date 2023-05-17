! RUN: %python %S/test_errors.py %s %flang_fc1
! test global name conflicts

subroutine ext1
end

subroutine ext2
  !ERROR: Two entities have the same global name 'ext1'
  common /ext1/ x
end

module ext4
 contains
  !ERROR: Two entities have the same global name 'ext2'
  subroutine foo() bind(c,name="ext2")
  end
  !ERROR: Two entities have the same global name 'ext3'
  subroutine bar() bind(c,name="ext3")
  end
end

block data ext3
  !PORTABILITY: Global name 'ext4' conflicts with a module
  common /ext4/ x
end
