! This checks that -grecord-command-line is forwarded by the flang driver to
! an FC1 -dwarf-debug-flags argument and that -gno-record-command-line
! disables it, matching clang behavior.
!
! RUN: %flang -### -grecord-command-line %s 2>&1 | FileCheck --check-prefix=GRECORD %s
! RUN: %flang -### -gno-record-command-line %s 2>&1 | FileCheck --check-prefix=GNO_RECORD %s
! RUN: %flang -### -grecord-command-line -gno-record-command-line %s 2>&1 | FileCheck --check-prefix=GNO_RECORD %s
!
! GRECORD: "-dwarf-debug-flags"
!
! GNO_RECORD-NOT: "-dwarf-debug-flags"

program p
end program p
