! Test forwarding of -f[no-]repack-arrays options:
! RUN: %flang -frepack-arrays %s -### -fsyntax-only 2>&1 | FileCheck --check-prefix=REPACKCMD %s
! RUN: %flang -fno-repack-arrays %s -### -fsyntax-only 2>&1 | FileCheck --check-prefix=NOREPACKCMD %s
! RUN: %flang -frepack-arrays -fno-repack-arrays %s -### -fsyntax-only 2>&1 | FileCheck --check-prefix=NOREPACKCMD %s
! RUN: %flang -fno-repack-arrays -frepack-arrays %s -### -fsyntax-only 2>&1 | FileCheck --check-prefix=REPACKCMD %s

! Test proper setting of the lowering options:
! RUN: %flang_fc1 -frepack-arrays %s -emit-hlfir -o - | FileCheck --check-prefix=REPACK %s
! RUN: %flang_fc1 -fno-repack-arrays %s -emit-hlfir -o - | FileCheck --check-prefix=NOREPACK %s
! RUN: %flang_fc1 -frepack-arrays -fno-repack-arrays %s -emit-hlfir -o - | FileCheck --check-prefix=NOREPACK %s
! RUN: %flang_fc1 -fno-repack-arrays -frepack-arrays %s -emit-hlfir -o - | FileCheck --check-prefix=REPACK %s

! REPACKCMD: "-fc1"{{.*}}"-frepack-arrays"
! REPACKCMD-NOT: -fno-repack-arrays
! NOREPACKCMD: "-fc1"{{.*}}"-fno-repack-arrays"
! NOREPACKCMD-NOT: -frepack-arrays

subroutine test(x)
  real :: x(:)
  ! REPACK: fir.pack_array
  ! REPACK: fir.unpack_array
  ! NOREPACK-NOT: fir.pack_array
  ! NOREPACK-NOT: fir.unpack_array
end subroutine
