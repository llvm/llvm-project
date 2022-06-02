! RUN: bbc %s -o - | tco | FileCheck %s

  COMPLEX c
  c%RE = 3.14
  CALL sub(c)
END

! Verify that the offset in the struct does not regress from i32.
! CHECK-LABEL: define void @_QQmain()
! CHECK: getelementptr { float, float }, ptr %{{[0-9]+}}, i64 0, i32 0

