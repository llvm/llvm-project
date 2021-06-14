! RUN: %S/test_symbols.sh %s %t %flang_fc1
! REQUIRES: shell

! Test that a procedure is only implicitly resolved as an intrinsic function
! (resp. subroutine) if this is a function (resp. subroutine)

!DEF: /expect_external (Subroutine) Subprogram
subroutine expect_external
 !DEF: /acos EXTERNAL (Subroutine) ProcEntity
 !DEF: /expect_external/x (Implicit) ObjectEntity REAL(4)
 call acos(x)
 !DEF: /expect_external/i (Implicit) ObjectEntity INTEGER(4)
 !DEF: /system_clock EXTERNAL (Function, Implicit) ProcEntity REAL(4)
 !DEF: /expect_external/icount (Implicit) ObjectEntity INTEGER(4)
 i = system_clock(icount)
end subroutine

!DEF: /expect_intrinsic (Subroutine) Subprogram
subroutine expect_intrinsic
 !DEF: /expect_intrinsic/y (Implicit) ObjectEntity REAL(4)
 !DEF: /expect_intrinsic/acos ELEMENTAL, INTRINSIC, PURE (Function) ProcEntity
 !DEF: /expect_intrinsic/x (Implicit) ObjectEntity REAL(4)
 y = acos(x)
 !DEF: /expect_intrinsic/system_clock INTRINSIC (Subroutine) ProcEntity
 !DEF: /expect_intrinsic/icount (Implicit) ObjectEntity INTEGER(4)
 call system_clock(icount)
end subroutine

! Sanity check that the EXTERNAL attribute is not bypassed by
! implicit intrinsic resolution, even if it otherwise perfectly
! matches an intrinsic call.

!DEF: /expect_external_2 (Subroutine) Subprogram
subroutine expect_external_2
 !DEF: /expect_external_2/matmul EXTERNAL (Function, Implicit) ProcEntity INTEGER(4)
 external :: matmul
 !DEF: /expect_external_2/cpu_time EXTERNAL (Subroutine) ProcEntity
 external :: cpu_time
 !DEF: /expect_external_2/x ObjectEntity REAL(4)
 !DEF: /expect_external_2/y ObjectEntity REAL(4)
 !DEF: /expect_external_2/z ObjectEntity REAL(4)
 !DEF: /expect_external_2/t ObjectEntity REAL(4)
 real x(2,2), y(2), z(2), t
 !REF: /expect_external_2/z
 !REF: /expect_external_2/matmul
 !REF: /expect_external_2/x
 !REF: /expect_external_2/y
 z = matmul(x, y)
 !REF: /expect_external_2/cpu_time
 !REF: /expect_external_2/t
 call cpu_time(t)
end subroutine
