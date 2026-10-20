! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s
! CHECK: error: misplaced declaration in the execution part
! CHECK:  real, pointer :: p2(:,:)
! CHECK: in the context: execution part construct
real, allocatable, target :: a2(:,:)
allocate(a2(2:11,0:9))
real, pointer :: p2(:,:)
p2 => a2(2:3,1:2)
end
