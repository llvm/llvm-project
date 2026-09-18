! REQUIRES: powerpc-registered-target
! RUN: %flang_fc1 -triple powerpc64-ibm-aix -emit-hlfir -o - %s \
! RUN:   | FileCheck %s --check-prefix=CHECK-AIX
! RUN: %flang_fc1 -triple powerpc64le-unknown-linux-gnu -emit-hlfir -o - %s \
! RUN:   | FileCheck %s --check-prefix=CHECK-LNX

! On Linux PPC the sticky bits are written directly via mffs/mtfsf
! (llvm.ppc.readflm/llvm.ppc.setflm) instead of feraiseexcept/
! feclearexcept, to avoid spurious SIGFPE when trapping is armed.
! On AIX feraiseexcept/feclearexcept path is called.

program test
  use ieee_arithmetic
  logical :: flag_val

! CHECK-AIX-LABEL: func.func @_QQmain()
! CHECK-LNX-LABEL: func.func @_QQmain()

  ! ------------------------------------------------------------------
  ! ieee_set_flag(ieee_invalid, .false.)
  ! AIX:       feraiseexcept/feclearexcept call
  ! Linux PPC: inline arith mask + readflm/setflm
  ! ------------------------------------------------------------------

  ! CHECK-AIX:      fir.convert %false
  ! CHECK-AIX:      fir.if
  ! CHECK-AIX:        fir.call @_FortranAferaiseexcept({{.*}}) {{.*}} : (i32) -> ()
  ! CHECK-AIX:      } else {
  ! CHECK-AIX:        fir.call @_FortranAfeclearexcept({{.*}}) {{.*}} : (i32) -> ()
  ! CHECK-AIX:      }

  ! CHECK-LNX:      %false = arith.constant false
  ! CHECK-LNX:      fir.if
  ! CHECK-LNX:        fir.call @llvm.ppc.readflm() {{.*}} : () -> f64
  ! CHECK-LNX:        fir.call @llvm.ppc.setflm({{.*}}) {{.*}} : (f64) -> f64
  ! CHECK-LNX:      } else {
  ! CHECK-LNX:        fir.call @llvm.ppc.readflm() {{.*}} : () -> f64
  ! CHECK-LNX:        fir.call @llvm.ppc.setflm({{.*}}) {{.*}} : (f64) -> f64
  ! CHECK-LNX:      }
  call ieee_set_flag(ieee_invalid, .false.)

  ! ------------------------------------------------------------------
  ! ieee_set_flag(ieee_overflow, .true.)
  ! ------------------------------------------------------------------

  ! CHECK-AIX:      fir.convert %true
  ! CHECK-AIX:      fir.if
  ! CHECK-AIX:        fir.call @_FortranAferaiseexcept({{.*}}) {{.*}} : (i32) -> ()
  ! CHECK-AIX:      } else {
  ! CHECK-AIX:        fir.call @_FortranAfeclearexcept({{.*}}) {{.*}} : (i32) -> ()
  ! CHECK-AIX:      }

  ! CHECK-LNX:      %true = arith.constant true
  ! CHECK-LNX:      fir.if
  ! CHECK-LNX:        fir.call @llvm.ppc.readflm() {{.*}} : () -> f64
  ! CHECK-LNX:        fir.call @llvm.ppc.setflm({{.*}}) {{.*}} : (f64) -> f64
  ! CHECK-LNX:      } else {
  ! CHECK-LNX:        fir.call @llvm.ppc.readflm() {{.*}} : () -> f64
  ! CHECK-LNX:        fir.call @llvm.ppc.setflm({{.*}}) {{.*}} : (f64) -> f64
  ! CHECK-LNX:      }
  call ieee_set_flag(ieee_overflow, .true.)

  ! ------------------------------------------------------------------
  ! ieee_set_flag(ieee_divide_by_zero, flag_val) -- runtime boolean
  ! ------------------------------------------------------------------

  ! CHECK-AIX:      fir.if
  ! CHECK-AIX:        fir.call @_FortranAferaiseexcept({{.*}}) {{.*}} : (i32) -> ()
  ! CHECK-AIX:      } else {
  ! CHECK-AIX:        fir.call @_FortranAfeclearexcept({{.*}}) {{.*}} : (i32) -> ()
  ! CHECK-AIX:      }

  ! CHECK-LNX:      fir.load {{.*}} : !fir.ref<!fir.logical<4>>
  ! CHECK-LNX:      fir.if
  ! CHECK-LNX:        fir.call @llvm.ppc.readflm() {{.*}} : () -> f64
  ! CHECK-LNX:        fir.call @llvm.ppc.setflm({{.*}}) {{.*}} : (f64) -> f64
  ! CHECK-LNX:      } else {
  ! CHECK-LNX:        fir.call @llvm.ppc.readflm() {{.*}} : () -> f64
  ! CHECK-LNX:        fir.call @llvm.ppc.setflm({{.*}}) {{.*}} : (f64) -> f64
  ! CHECK-LNX:      }
  call ieee_set_flag(ieee_divide_by_zero, flag_val)

  ! ------------------------------------------------------------------
  ! On Linux PPC, feraiseexcept/feclearexcept must NOT appear for
  ! ieee_set_flag.
  ! ------------------------------------------------------------------
  ! CHECK-LNX-NOT: fir.call {{.*}}feraiseexcept
  ! CHECK-LNX-NOT: fir.call {{.*}}feclearexcept

end program
