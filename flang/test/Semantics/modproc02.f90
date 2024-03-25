!RUN: %flang_fc1 -fdebug-dump-symbols %s | FileCheck %s
module m
  interface
    module subroutine s(x) ! implicitly typed
    end
  end interface
end

submodule (m) sm
  implicit none
 contains
  !Ensure no error here due to IMPLICIT NONE
  module procedure s
  end
end

!CHECK:  Module scope: m size=0 alignment=1 sourceRange=63 bytes
!CHECK:    s, MODULE, PUBLIC (Subroutine): Subprogram isInterface (REAL(4) x)
!CHECK:    Subprogram scope: s size=4 alignment=4 sourceRange=26 bytes
!CHECK:      s (Subroutine): HostAssoc
!CHECK:      x (Implicit) size=4 offset=0: ObjectEntity dummy type: REAL(4)
!CHECK:    Module scope: sm size=0 alignment=1 sourceRange=65 bytes
!CHECK:      s, MODULE, PUBLIC (Subroutine): Subprogram (REAL(4) x) moduleInterface: s, MODULE, PUBLIC (Subroutine): Subprogram isInterface (REAL(4) x)
!CHECK:      Subprogram scope: s size=4 alignment=4 sourceRange=22 bytes
!CHECK:        s: HostAssoc
!CHECK:        x size=4 offset=0: ObjectEntity dummy type: REAL(4)
