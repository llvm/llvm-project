! RUN: bbc -emit-fir -o - %s | FileCheck %s

! CHECK-LABEL: func @_QQmain
program p
  ! CHECK-DAG: [[I:%[0-9]+]] = fir.alloca i32 {{{.*}}uniq_name = "_QFEi"}
  ! CHECK-DAG: [[N:%[0-9]+]] = fir.alloca i32 {{{.*}}uniq_name = "_QFEn"}
  ! CHECK: [[T:%[0-9]+]] = fir.address_of(@_QFEt) : !fir.ref<!fir.array<3xi32>>
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
    print*, n, t, a, b, c, d ! expect: 102 111 223 333 102 105 223 7
  end associate

  call nest

  associate (x=>i)
    ! CHECK: [[IVAL:%[0-9]+]] = fir.load [[I]] : !fir.ref<i32>
    ! CHECK: [[TWO:%.*]] = arith.constant 2 : i32
    ! CHECK: arith.cmpi eq, [[IVAL]], [[TWO]] : i32
    ! CHECK: ^bb
    if (x==2) goto 9
    ! CHECK: [[IVAL:%[0-9]+]] = fir.load [[I]] : !fir.ref<i32>
    ! CHECK: [[THREE:%.*]] = arith.constant 3 : i32
    ! CHECK: arith.cmpi eq, [[IVAL]], [[THREE]] : i32
    ! CHECK: ^bb
    ! CHECK: fir.call @_FortranAStopStatementText
    ! CHECK: fir.unreachable
    ! CHECK: ^bb
    if (x==3) stop 'Halt'
    ! CHECK: fir.call @_FortranAioOutputAscii
    print*, "ok"
  9 end associate
end

! CHECK-LABEL: func @_QPfoo
integer function foo(x)
  integer x
  integer, save :: i = 0
  i = i + x
  foo = i
end function foo

! CHECK-LABEL: func @_QPnest(
subroutine nest
  integer, parameter :: n = 10
  integer :: a(5), b(n)
  associate (s => sequence(size(a)))
    a = s
    associate(t => sequence(n))
      b = t
      ! CHECK:   cond_br %{{.*}}, [[BB1:\^bb[0-9]]], [[BB2:\^bb[0-9]]]
      ! CHECK: [[BB1]]:
      ! CHECK:   br [[BB3:\^bb[0-9]]]
      ! CHECK: [[BB2]]:
      if (a(1) > b(1)) goto 9
    end associate
    a = a * a
  end associate
  ! CHECK:   br [[BB3]]
  ! CHECK: [[BB3]]:
9 print *, sum(a), sum(b) ! expect: 55 55
contains
  function sequence(n)
    integer sequence(n)
    sequence = [(i,i=1,n)]
  end function
end subroutine nest
