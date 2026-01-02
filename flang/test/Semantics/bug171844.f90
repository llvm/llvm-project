! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck --check-prefix=CHECK-WARNING %s
! RUN: not %flang_fc1 -fsyntax-only -Wno-bad-value-in-dead-code %s 2>&1 | FileCheck %s

real a(2)

if (.false.) then
  !CHECK-WARNING::8:12: warning: subscript 3 is greater than upper bound 2 for dimension 1 of array [-Wbad-value-in-dead-code]
  print *, a(3)
end if

if (.true.) then
  !CHECK::13:12: error: subscript 0 is less than lower bound 1 for dimension 1 of array
  print *, a(0)
else
  !CHECK-WARNING::16:12: warning: subscript 0 is less than lower bound 1 for dimension 1 of array [-Wbad-value-in-dead-code]
  print *, a(0)
end if

if (.false.) then
else if (.true.) then
  !CHECK::22:12: error: subscript 0 is less than lower bound 1 for dimension 1 of array
  print *, a(0)
else
  !CHECK-WARNING::25:12: warning: subscript 0 is less than lower bound 1 for dimension 1 of array [-Wbad-value-in-dead-code]
  print *, a(0)
end if

if (.true.) then
else if (.true.) then
  !CHECK-WARNING::31:12: warning: subscript -1 is less than lower bound 1 for dimension 1 of array [-Wbad-value-in-dead-code]
  print *, a(-1)
else
  !CHECK-WARNING::34:12: warning: subscript 3 is greater than upper bound 2 for dimension 1 of array [-Wbad-value-in-dead-code]
  print *, a(3)
end if

end
