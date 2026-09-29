  use modfile70
end

! RUN: not %flang_fc1 -fsyntax-only -J%S/Inputs -w %s 2>&1 | FileCheck --check-prefix=ERROR %s
! ERROR: Cannot use module file for module 'modfile70': File has invalid checksum:
