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
