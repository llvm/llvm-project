! Test forwarding of -frepack-arrays-contiguity options:
! RUN: %flang -frepack-arrays-contiguity=whole %s -### -fsyntax-only 2>&1 | FileCheck --check-prefix=WHOLECMD %s
! RUN: %flang -frepack-arrays-contiguity=innermost %s -### -fsyntax-only 2>&1 | FileCheck --check-prefix=INNERMOSTCMD %s
! RUN: %flang -frepack-arrays-contiguity=innermost -frepack-arrays-contiguity=whole %s -### -fsyntax-only 2>&1 | FileCheck --check-prefix=WHOLECMD %s
! RUN: %flang -frepack-arrays-contiguity=whole -frepack-arrays-contiguity=innermost %s -### -fsyntax-only 2>&1 | FileCheck --check-prefix=INNERMOSTCMD %s
! RUN: not %flang -frepack-arrays-contiguity= -frepack-arrays-contiguity=innermost %s -### -fsyntax-only 2>&1 | FileCheck --check-prefix=ERROR %s
! RUN: not %flang -frepack-arrays-contiguity=whole3 -frepack-arrays-contiguity=innermost %s -### -fsyntax-only 2>&1 | FileCheck --check-prefix=ERROR %s
! RUN: not %flang -frepack-arrays-contiguity=innermostg -frepack-arrays-contiguity=innermost %s -### -fsyntax-only 2>&1 | FileCheck --check-prefix=ERROR %s

! Test proper setting of the lowering options:
! RUN: %flang_fc1 -frepack-arrays -frepack-arrays-contiguity=whole %s -emit-hlfir -o - | FileCheck --check-prefix=WHOLE %s
! RUN: %flang_fc1 -frepack-arrays-contiguity=whole %s -emit-hlfir -o - | FileCheck --check-prefix=NOREPACK %s
! RUN: %flang_fc1 -frepack-arrays -frepack-arrays-contiguity=innermost %s -emit-hlfir -o - | FileCheck --check-prefix=INNERMOST %s
! RUN: %flang_fc1 -frepack-arrays-contiguity=innermost %s -emit-hlfir -o - | FileCheck --check-prefix=NOREPACK %s

! Default setting is 'innermost':
! RUN: %flang_fc1 -frepack-arrays %s -emit-hlfir -o - | FileCheck --check-prefix=INNERMOST %s

! ERROR: error: unsupported argument '{{.*}}' to option '-frepack-arrays-contiguity='

! WHOLECMD: "-fc1"{{.*}}"-frepack-arrays-contiguity=whole"
! INNERMOSTCMD: "-fc1"{{.*}}"-frepack-arrays-contiguity=innermost"

subroutine test(x)
  real :: x(:,:)
  ! WHOLE: fir.pack_array{{.*}}whole
  ! WHOLE: fir.unpack_array
  ! INERMOST: fir.pack_array{{.*}}innermost
  ! INNERMOST: fir.unpack_array
  ! NOREPACK-NOT: fir.pack_array
  ! NOREPACK-NOT: fir.unpack_array
end subroutine
