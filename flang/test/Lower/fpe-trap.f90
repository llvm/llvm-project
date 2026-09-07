! Test that -ffpe-trap= generates a call to _FortranAEnableFPETraps in main().

! RUN: %flang_fc1 -emit-fir -ffpe-trap=invalid %s -o - | FileCheck --check-prefix=CHECK-INVALID %s
! RUN: %flang_fc1 -emit-fir -ffpe-trap=invalid,zero,overflow %s -o - | FileCheck --check-prefix=CHECK-MULTI %s
! RUN: %flang_fc1 -emit-fir -ffpe-trap=invalid,underflow,inexact,denormal %s -o - | FileCheck --check-prefix=CHECK-EXT %s
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck --check-prefix=CHECK-NONE %s
! RUN: %flang_fc1 -emit-fir -ffpe-trap=none %s -o - | FileCheck --check-prefix=CHECK-NONE %s
! RUN: %flang_fc1 -emit-fir -ffpe-trap=invalid,none %s -o - | FileCheck --check-prefix=CHECK-NONE %s

program test
  continue
end

! CHECK-INVALID: %[[INVALID:.*]] = arith.constant 1 : i32
! CHECK-INVALID: fir.call @_FortranAProgramStart(
! CHECK-INVALID: fir.call @_FortranAEnableFPETraps(%[[INVALID]])
! CHECK-INVALID: fir.call @_QQmain

! invalid=1, zero=4, overflow=8 => 13
! CHECK-MULTI: %[[MULTI:.*]] = arith.constant 13 : i32
! CHECK-MULTI: fir.call @_FortranAProgramStart(
! CHECK-MULTI: fir.call @_FortranAEnableFPETraps(%[[MULTI]])
! CHECK-MULTI: fir.call @_QQmain

! invalid=1, denormal=2, underflow=16, inexact=32 => 51
! CHECK-EXT: %[[EXT:.*]] = arith.constant 51 : i32
! CHECK-EXT: fir.call @_FortranAProgramStart(
! CHECK-EXT: fir.call @_FortranAEnableFPETraps(%[[EXT]])
! CHECK-EXT: fir.call @_QQmain

! CHECK-NONE-NOT: @_FortranAEnableFPETraps
