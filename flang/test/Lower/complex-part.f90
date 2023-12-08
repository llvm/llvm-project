! RUN: bbc %s -o - | tco | FileCheck %s
! RUN: %flang -emit-llvm -S -mmlir -disable-external-name-interop %s -o - | FileCheck %s

  COMPLEX c
  c%RE = 3.14
  CALL sub(c)
END

! Verify that the offset in the struct does not regress from i32.
! CHECK-LABEL: define void @_QQmain()
! CHECK: getelementptr { float, float }, ptr %{{[0-9]+}}, i32 0, i32 0

