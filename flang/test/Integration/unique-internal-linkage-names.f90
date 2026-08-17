! Test that -funique-internal-linkage-names appends a hash suffix to internal
! procedures and sets the "sample-profile-suffix-elision-policy" attribute.

! RUN: %flang_fc1 -emit-llvm -funique-internal-linkage-names -o - %s | FileCheck %s

! CHECK-LABEL: define void @test_(
! CHECK      : call void @_QFtestPfooX__uniqX{{[0-9]+}}(ptr {{.*}})

! CHECK-LABEL: define internal void @_QFtestPfooX__uniqX{{[0-9]+}}(ptr {{.*}}) #0
! CHECK-NOT  : define internal void @_QFtestPfoo(

! CHECK: attributes #0 = { "sample-profile-suffix-elision-policy"="selected" }

subroutine test(x)
  integer, intent(inout) :: x
  call foo(x)
contains
  subroutine foo(y)
    integer, intent(inout) :: y
    y = y + 1
  end subroutine
end subroutine
