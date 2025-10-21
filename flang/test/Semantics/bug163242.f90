!RUN: %flang -fc1 -fsyntax-only %s | FileCheck --allow-empty %s
!CHECK-NOT: error:
character(0), allocatable :: ch
allocate(character(-1) :: ch)
end
