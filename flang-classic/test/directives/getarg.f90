! RUN: %flang -S -emit-llvm %s -o - | FileCheck %s

subroutine test()
  integer(kind=1) :: pos
  character(len=10) :: arg

  pos = 1_1

  call getarg(pos, arg)
end subroutine
! CHECK: call void {{.*}}@f90_getarga(ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, i64{{.*}}),
! CHECK-NOT: call void {{.*}}@getarg_(i8 {{.*}}, i8 {{.*}}, i64{{.*}}),
