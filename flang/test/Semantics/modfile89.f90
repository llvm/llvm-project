! RUN: split-file %s %t
! RUN: %flang_fc1 -fsyntax-only -module-dir %t %t/defs.f90
! RUN: %flang_fc1 -fsyntax-only -module-dir %t %t/use.f90
! RUN: FileCheck --check-prefix=PK --input-file=%t/pk.mod %s
! RUN: FileCheck --check-prefix=VM --input-file=%t/vm.mod %s
! RUN: FileCheck --check-prefix=MIDDLE --input-file=%t/middle.mod %s
! RUN: FileCheck --check-prefix=DOWN --input-file=%t/down.mod %s
! RUN: FileCheck --check-prefix=ONLYMOD --input-file=%t/down_only.mod %s
! RUN: FileCheck --check-prefix=RENMOD --input-file=%t/down_rename.mod %s
! RUN: FileCheck --check-prefix=DTMID --input-file=%t/dtmiddle.mod %s
! RUN: FileCheck --check-prefix=DTDOWN --input-file=%t/dtdown.mod %s

! Regression test: a module that re-exports another module's ambiguous
! (never-referenced, and therefore legal per F2023 14.2.2 p8) USE-associated
! name must not name that entity in its own module file, because the
! originating module's module file does not provide it.
!
! MIDDLE combines two distinct USEs of the same name (JPRB) that is never
! itself referenced within MIDDLE, so MIDDLE's module file correctly omits
! the ambiguous name.  DOWN re-exports MIDDLE via a whole-module USE; its
! module file must likewise omit JPRB.  Before the fix, compiling use.f90
! failed with "'jprb' not found in module 'middle'".
!
! DOWN_ONLY and DOWN_RENAME cover the same ambiguity reached through
! use middle,only:jprb and use middle,only:myjprb=>jprb, which resolve
! through ModuleVisitor::AddUse rather than the whole-module
! AddUseForPublicSymbols path that DOWN exercises.
!
! DTMIDDLE/DTDOWN cover a second, independent way of creating the
! ambiguous poison-pill symbol: two distinct derived types of the same
! name, handled by the "many possible combinations" tail of DoAddUse
! rather than its early-return whole-module path.  DTMIDDLE only pins
! the precondition (its module file omits the ambiguous name either way);
! DTDOWN is the one that catches a regression.

!--- defs.f90
module pk
  integer jprb
end module
module vm
  integer jprb
end module
module middle
  use pk
  use vm
end module
module down
  use middle
end module
module down_only
  use middle, only: jprb
end module
module down_rename
  use middle, only: myjprb => jprb
end module
module dtpk
  type :: dt
    integer a
  end type
end module
module dtvm
  type :: dt
    real b
  end type
end module
module dtmiddle
  use dtpk
  use dtvm
end module
module dtdown
  use dtmiddle
end module

!--- use.f90
use down
use down_only
use down_rename
use dtdown
end

! PK: module pk
! PK: jprb
! PK: end

! VM: module vm
! VM: jprb
! VM: end

! MIDDLE: module middle
! MIDDLE-NOT: jprb
! MIDDLE: end

! DOWN: module down
! DOWN-NOT: jprb
! DOWN: end

! ONLYMOD: module down_only
! ONLYMOD-NOT: jprb
! ONLYMOD: end

! RENMOD: module down_rename
! RENMOD-NOT: jprb
! RENMOD: end

! DTMID: module dtmiddle
! DTMID-NOT: dt
! DTMID: end

! DTDOWN: module dtdown
! DTDOWN-NOT: dt
! DTDOWN: end
