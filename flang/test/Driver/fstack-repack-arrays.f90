! Test forwarding of -f[no-]stack-repack-arrays options:
! RUN: %flang -fstack-repack-arrays %s -### -fsyntax-only 2>&1 | FileCheck --check-prefix=STACKCMD %s
! RUN: %flang -fno-stack-repack-arrays %s -### -fsyntax-only 2>&1 | FileCheck --check-prefix=HEAPCMD %s
! RUN: %flang -fstack-repack-arrays -fno-stack-repack-arrays %s -### -fsyntax-only 2>&1 | FileCheck --check-prefix=HEAPCMD %s
! RUN: %flang -fno-stack-repack-arrays -fstack-arrays %s -### -fsyntax-only 2>&1 | FileCheck --check-prefix=HEAPCMD %s
! RUN: %flang -fno-stack-repack-arrays -fstack-repack-arrays %s -### -fsyntax-only 2>&1 | FileCheck --check-prefix=STACKCMD %s

! Test proper setting of the lowering options:
! RUN: %flang_fc1 -frepack-arrays -fstack-repack-arrays %s -emit-hlfir -o - | FileCheck --check-prefix=STACK %s
! RUN: %flang_fc1 -frepack-arrays -fno-stack-repack-arrays %s -emit-hlfir -o - | FileCheck --check-prefix=HEAP %s

! Default setting is 'heap':
! RUN: %flang_fc1 -frepack-arrays %s -emit-hlfir -o - | FileCheck --check-prefix=HEAP %s

! STACKCMD: "-fc1"{{.*}}"-fstack-repack-arrays"
! HEAPCMD: "-fc1"{{.*}}"-fno-stack-repack-arrays"

subroutine test(x)
  real :: x(:,:)
  ! STACK: fir.pack_array{{.*}}stack
  ! STACK: fir.unpack_array{{.*}}stack
  ! HEAP: fir.pack_array{{.*}}heap
  ! HEAP: fir.unpack_array{{.*}}heap
end subroutine
