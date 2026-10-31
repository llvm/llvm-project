!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

! RUN: %flang_fc1 -emit-llvm -O2 -mllvm --enable-constant-argument-globalisation %s -o - | FileCheck %s

! CHECK: @_global_const_{{.*}} = internal constant i32 2
! CHECK: call void @take_int_(ptr nonnull @_global_const_{{.*}})

subroutine test()
  interface
  subroutine take_int(n)
    integer :: n
  end subroutine
  end interface
  call take_int(2)
end
