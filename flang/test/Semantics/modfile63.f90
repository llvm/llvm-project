! RUN: %flang_fc1 -fsyntax-only -I%S/Inputs/dir1 %s
! RUN: not %flang_fc1 -fsyntax-only -I%S/Inputs/dir2 %s 2>&1 | FileCheck --check-prefix=ERROR %s
! RUN: %flang_fc1 -Werror -fsyntax-only -I%S/Inputs/dir1 -I%S/Inputs/dir2 %s
! RUN: not %flang_fc1 -Werror -fsyntax-only -I%S/Inputs/dir2 -I%S/Inputs/dir1 %s 2>&1 | FileCheck  --check-prefix=WARNING %s

! Inputs/dir1 and Inputs/dir2 each have identical copies of modfile63b.mod.
! modfile63b.mod depends on Inputs/dir1/modfile63a.mod - the version in
! Inputs/dir2/modfile63a.mod has a distinct checksum and should be
! ignored with a warning.

! If it becomes necessary to recompile those modules, just use the
! module files as Fortran source.

use modfile63b
call s2
end

! ERROR: Could not find a module file for 'modfile63a' in the module search path with the expected checksum
! WARNING: Module file for 'modfile63a' appears later in the module search path than conflicting modules with different checksums
