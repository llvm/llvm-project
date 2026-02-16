! This checks that -grecord-command-line is forwarded by the flang driver to
! an FC1 -dwarf-debug-flags argument and that -gno-record-command-line
! disables it, matching clang behavior.
!
! RUN: %flang -### -target x86_64-unknown-linux-gnu -c -grecord-command-line %s 2>&1 | FileCheck --check-prefix=GRECORD %s
! RUN: %flang -### -target x86_64-unknown-linux-gnu -c -gno-record-command-line %s 2>&1 | FileCheck --check-prefix=GNO_RECORD %s
! RUN: %flang -### -target x86_64-unknown-linux-gnu -c -grecord-command-line -gno-record-command-line %s 2>&1 | FileCheck --check-prefix=GNO_RECORD %s
! RUN: %flang -### -target x86_64-unknown-linux-gnu -c -grecord-command-line -o - %s 2>&1 | FileCheck --check-prefix=GRECORD_O %s
! RUN: %flang -target x86_64-unknown-linux-gnu -c -g -grecord-command-line %s -o %t.o
! RUN: llvm-dwarfdump --debug-info %t.o | FileCheck --check-prefix=WITH_GRECORD %s
! RUN: %flang -target x86_64-unknown-linux-gnu -c -g -gno-record-command-line %s -o %t.norecord.o
! RUN: llvm-dwarfdump --debug-info %t.norecord.o | FileCheck --check-prefix=NO_GRECORD %s
!
! GRECORD: "-dwarf-debug-flags"
! GRECORD: -grecord-command-line
!
! GNO_RECORD-NOT: "-dwarf-debug-flags"
! GNO_RECORD-NOT: -grecord-command-line
!
! GRECORD_O: -grecord-command-line
! GRECORD_O: -o -
!
! WITH_GRECORD: DW_AT_producer
! WITH_GRECORD-SAME: -grecord-command-line
!
! NO_GRECORD: DW_AT_producer
! NO_GRECORD-NOT: -grecord-command-line

program p
end program p
