! Out-of-range constant subscripts are warnings by default, so use
! -fno-out-of-bounds-subscripts here to exercise the downgrade of errors to
! warnings in code that is known at compilation time to be dead.
! RUN: not %flang_fc1 -fsyntax-only -fno-out-of-bounds-subscripts %s 2>&1 | FileCheck --check-prefix=CHECK-WARNING %s
! RUN: not %flang_fc1 -fsyntax-only -fno-out-of-bounds-subscripts -Wno-bad-value-in-dead-code %s 2>&1 | FileCheck %s

real a(2)

if (.false.) then
  !CHECK-WARNING: warning: subscript 3 is greater than upper bound 2 for dimension 1 of array [-Wbad-value-in-dead-code]
  print *, a(3)
end if

if (.true.) then
  !CHECK: error: subscript 0 is less than lower bound 1 for dimension 1 of array
  print *, a(0)
else
  !CHECK-WARNING: warning: subscript 0 is less than lower bound 1 for dimension 1 of array [-Wbad-value-in-dead-code]
  print *, a(0)
end if

if (.false.) then
else if (.true.) then
  !CHECK: error: subscript 0 is less than lower bound 1 for dimension 1 of array
  print *, a(0)
else
  !CHECK-WARNING: warning: subscript 0 is less than lower bound 1 for dimension 1 of array [-Wbad-value-in-dead-code]
  print *, a(0)
end if

if (.true.) then
else if (.true.) then
  !CHECK-WARNING: warning: subscript -1 is less than lower bound 1 for dimension 1 of array [-Wbad-value-in-dead-code]
  print *, a(-1)
else
  !CHECK-WARNING: warning: subscript 3 is greater than upper bound 2 for dimension 1 of array [-Wbad-value-in-dead-code]
  print *, a(3)
end if

end
