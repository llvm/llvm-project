! Tests that `--save-temps` works properly when a module from a non standard dir
! is included with `-I/...`.

! RUN: rm -rf %t && split-file %s %t
! RUN: mkdir %t/mod_inc_dir
! RUN: mv %t/somemodule.mod %t/mod_inc_dir
! RUN: %flang -S -emit-llvm --save-temps=obj -I%t/mod_inc_dir -fno-integrated-as \
! RUN:   %t/ModuleUser.f90 -o %t/ModuleUser
! RUN: ls %t | FileCheck %s

! Verify that the temp file(s) were written to disk.
! CHECK: ModuleUser.i

!--- somemodule.mod
!mod$ v1 sum:e9e8fd2bd49e8daa
module SomeModule

end module SomeModule
!--- ModuleUser.f90

module User
  use SomeModule
end module User

program dummy
end program
