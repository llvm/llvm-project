! Test how the driver lowers -M, -MM, -MD, -MMD, -MF, -MT and -MQ to `flang
! -fc1` and the generated dependency file. Fortran has no system/user header
! split, so -M behaves like -MM and -MD like -MMD. Unlike clang, -M/-MM use
! -fsyntax-only (not -Eonly) so that USE module dependencies are resolved.

!--------------------------------------------------------------------------
! -M / -MM: emit dependencies via -fsyntax-only (runs through semantics to
! resolve USE statements), default to stdout, imply -w, and use the object
! file as the default target.
!--------------------------------------------------------------------------
! RUN: %flang -### -M  %s 2>&1 | FileCheck %s --check-prefix=M
! RUN: %flang -### -MM %s 2>&1 | FileCheck %s --check-prefix=M
! M: "-fc1"
! M-SAME: "-fsyntax-only"
! M-SAME: "-w"
! M-SAME: "-dependency-file" "-"
! M-SAME: "-MT" "dependency-file.o"

!--------------------------------------------------------------------------
! -M with -o: -o names the dependency file, NOT the dependency target. The target
! still defaults to the object file derived from the input.
!--------------------------------------------------------------------------
! RUN: %flang -### -M  -o named.d %s 2>&1 | FileCheck %s --check-prefix=M-O
! RUN: %flang -### -MM -o named.d %s 2>&1 | FileCheck %s --check-prefix=M-O
! M-O: "-fsyntax-only"
! M-O-SAME: "-dependency-file" "named.d"
! M-O-SAME: "-MT" "dependency-file.o"

!--------------------------------------------------------------------------
! -M with -o -: dependencies go to stdout, the target is still the object.
!--------------------------------------------------------------------------
! RUN: %flang -### -M -o - %s 2>&1 | FileCheck %s --check-prefix=M-O-STDOUT
! M-O-STDOUT: "-dependency-file" "-"
! M-O-STDOUT-SAME: "-MT" "dependency-file.o"

!--------------------------------------------------------------------------
! -MF redirects the dependency file; the target is unaffected.
!--------------------------------------------------------------------------
! RUN: %flang -### -M -MF custom.d %s 2>&1 | FileCheck %s --check-prefix=M-MF
! M-MF: "-fsyntax-only"
! M-MF-SAME: "-dependency-file" "custom.d"
! M-MF-SAME: "-MT" "dependency-file.o"

!--------------------------------------------------------------------------
! -MT sets the target verbatim and replaces the default object target.
!--------------------------------------------------------------------------
! RUN: %flang -### -M -MT my_target %s 2>&1 | FileCheck %s --check-prefix=M-MT
! M-MT: "-dependency-file" "-"
! M-MT-SAME: "-MT" "my_target"
! M-MT-NOT: "-MT" "dependency-file.o"

!--------------------------------------------------------------------------
! -MQ sets the target but quotes characters special to Make ($ -> $$), whereas
! -MT writes the target verbatim. Checked on the written output.
!--------------------------------------------------------------------------
! RUN: %flang -M -MQ 'a$b' %s 2>&1 | FileCheck %s --check-prefix=MQ
! MQ: a$$b:
! RUN: %flang -M -MT 'a$b' %s 2>&1 | FileCheck %s --check-prefix=MT
! MT: a$b:

!--------------------------------------------------------------------------
! -MD / -MMD: compile normally (no -Eonly) and also write the dependency
! file, defaulting its name and the target to those derived from the input.
!--------------------------------------------------------------------------
! RUN: %flang -### -MD  -c %s 2>&1 | FileCheck %s --check-prefix=MD
! RUN: %flang -### -MMD -c %s 2>&1 | FileCheck %s --check-prefix=MD
! MD: "-fc1"
! MD-NOT: "-fsyntax-only"
! MD-SAME: "-dependency-file" "dependency-file.d"
! MD-SAME: "-MT" "dependency-file.o"

!--------------------------------------------------------------------------
! -MD with -o: the dependency file name and target both follow -o.
!--------------------------------------------------------------------------
! RUN: %flang -### -MD -c -o out.o %s 2>&1 | FileCheck %s --check-prefix=MD-O
! MD-O: "-dependency-file" "out.d"
! MD-O-SAME: "-MT" "out.o"

!--------------------------------------------------------------------------
! -MD combined with -MF and -MQ.
!--------------------------------------------------------------------------
! RUN: %flang -### -MD -MF dep.d -MQ obj.o -c -o out.o %s 2>&1 | FileCheck %s --check-prefix=MD-MIX
! MD-MIX: "-dependency-file" "dep.d"
! MD-MIX-SAME: "-MT" "obj.o"

!--------------------------------------------------------------------------
! End-to-end: the dependencies are written and no preprocessed source leaks.
!--------------------------------------------------------------------------
! RUN: %flang -M %s -MT custom.o 2>&1 | FileCheck %s --check-prefix=DEPS
! DEPS: custom.o:
! DEPS: dependency-file.f90

! End-to-end with -o: dependencies are written to the named file with the
! object as the dependency target.
! RUN: rm -rf %t && mkdir %t
! RUN: %flang -M -o %t/deps.d %s
! RUN: FileCheck %s --check-prefix=DEPS-O --input-file=%t/deps.d
! DEPS-O: dependency-file.o:
! DEPS-O: dependency-file.f90

! End-to-end with -o -: dependencies go to stdout, object is the target.
! RUN: %flang -M -o - %s 2>&1 | FileCheck %s --check-prefix=DEPS-STDOUT
! DEPS-STDOUT: dependency-file.o:
! DEPS-STDOUT: dependency-file.f90

program test
  x = 1
end program test
