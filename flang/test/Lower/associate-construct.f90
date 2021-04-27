! RUN: bbc -emit-fir -o - %s | FileCheck %s

! CHECK-LABEL: func @_QQmain
program p
  ! CHECK: [[N:%[0-9]+]] = fir.alloca i32 {{{.*}}uniq_name = "_QEn"}
  ! CHECK: [[T:%[0-9]+]] = fir.address_of(@_QEt) : !fir.ref<!fir.array<3xi32>>
  integer :: n, foo, t(3)
  ! CHECK: [[N]]
  ! CHECK-COUNT-3: fir.coordinate_of [[T]]
  n = 100; t(1) = 111; t(2) = 222; t(3) = 333
  ! CHECK: fir.load [[N]]
  ! CHECK: addi {{.*}} %c5
  ! CHECK: fir.store %{{[0-9]*}} to [[B:%[0-9]+]]
  ! CHECK: [[C:%[0-9]+]] = fir.coordinate_of [[T]]
  ! CHECK: fir.call @_QPfoo
  ! CHECK: fir.store %{{[0-9]*}} to [[D:%[0-9]+]]
  associate (a => n, b => n+5, c => t(2), d => foo(7))
    ! CHECK: fir.load [[N]]
    ! CHECK: addi %{{[0-9]*}}, %c1
    ! CHECK: fir.store %{{[0-9]*}} to [[N]]
    a = a + 1
    ! CHECK: fir.load [[C]]
    ! CHECK: addi %{{[0-9]*}}, %c1
    ! CHECK: fir.store %{{[0-9]*}} to [[C]]
    c = c + 1
    ! CHECK: fir.load [[N]]
    ! CHECK: addi %{{[0-9]*}}, %c1
    ! CHECK: fir.store %{{[0-9]*}} to [[N]]
    n = n + 1
    ! CHECK: fir.load [[N]]
    ! CHECK: fir.embox [[T]]
    ! CHECK: fir.load [[N]]
    ! CHECK: fir.load [[B]]
    ! CHECK: fir.load [[C]]
    ! CHECK: fir.load [[D]]
    print*, n, t, a, b, c, d ! expected output: 102 111 223 333 102 105 223 7
  end associate
end

! CHECK-LABEL: func @_QPfoo
integer function foo(x)
  integer x
  integer, save :: i = 0
  i = i + x
  foo = i
end function foo
