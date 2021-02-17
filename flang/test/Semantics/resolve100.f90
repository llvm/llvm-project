!RUN: %f18 -fdebug-dump-symbols %s | FileCheck %s

program p
  ! CHECK: a size=4 offset=0: ObjectEntity type: LOGICAL(4)
  ! CHECK: b size=4 offset=4: ObjectEntity type: REAL(4)
  logical :: a = .false.
  real :: b = 9.73
  ! CHECK: a: AssocEntity type: REAL(4) expr:b
  ! CHECK: b: AssocEntity type: LOGICAL(4) expr:a
  associate (b => a, a => b)
    print*, a, b
  end associate
  print*, a, b
end
