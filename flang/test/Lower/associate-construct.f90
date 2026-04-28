! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s

! CHECK-LABEL: func @_QQmain
program p
  ! CHECK-DAG: %[[I_ALLOC:.*]] = fir.alloca i32 {{{.*}}uniq_name = "_QFEi"}
  ! CHECK-DAG: %[[I:.*]]:2 = hlfir.declare %[[I_ALLOC]] {uniq_name = "_QFEi"}
  ! CHECK-DAG: %[[N_ALLOC:.*]] = fir.alloca i32 {{{.*}}uniq_name = "_QFEn"}
  ! CHECK-DAG: %[[N:.*]]:2 = hlfir.declare %[[N_ALLOC]] {uniq_name = "_QFEn"}
  ! CHECK: %[[T_ALLOC:.*]] = fir.alloca !fir.array<3xi32> {bindc_name = "t", uniq_name = "_QFEt"}
  ! CHECK: %[[T:.*]]:2 = hlfir.declare %[[T_ALLOC]](%{{.*}}) {uniq_name = "_QFEt"}
  integer :: n, foo, t(3)
  ! CHECK: hlfir.assign %c100{{.*}} to %[[N]]#0
  ! CHECK-COUNT-3: hlfir.designate %[[T]]#0
  n = 100; t(1) = 111; t(2) = 222; t(3) = 333
  ! 'a' is associated to 'n' directly via hlfir.declare wrapping %[[N]]
  ! CHECK: %[[A:.*]]:2 = hlfir.declare %[[N]]#0 {uniq_name = "_QFEa"}
  ! 'b' = n+5: load n, addi 5, store to alloca, hlfir.declare it
  ! CHECK: %[[NLOAD_B:.*]] = fir.load %[[N]]#0
  ! CHECK: %[[NPLUS5:.*]] = arith.addi %[[NLOAD_B]], %c5{{.*}} : i32
  ! CHECK: fir.store %[[NPLUS5]] to %[[B_ALLOC:.*]] : !fir.ref<i32>
  ! CHECK: %[[B:.*]]:2 = hlfir.declare %[[B_ALLOC]] {uniq_name = "_QFEb"}
  ! 'c' = t(2): designate, hlfir.declare it
  ! CHECK: %[[C_DESIG:.*]] = hlfir.designate %[[T]]#0 (%c2{{.*}})
  ! CHECK: %[[C:.*]]:2 = hlfir.declare %[[C_DESIG]] {uniq_name = "_QFEc"}
  ! 'd' = foo(7): call foo, store result, hlfir.declare it
  ! CHECK: %[[FOO_CALL:.*]] = fir.call @_QPfoo(%{{.*}})
  ! CHECK: fir.store %[[FOO_CALL]] to %[[D_ALLOC:.*]] : !fir.ref<i32>
  ! CHECK: %[[D:.*]]:2 = hlfir.declare %[[D_ALLOC]] {uniq_name = "_QFEd"}
  associate (a => n, b => n+5, c => t(2), d => foo(7))
    ! CHECK: fir.load %[[A]]#0
    ! CHECK: arith.addi %{{.*}}, %c1
    ! CHECK: hlfir.assign %{{.*}} to %[[A]]#0
    a = a + 1
    ! CHECK: fir.load %[[C]]#0
    ! CHECK: arith.addi %{{.*}}, %c1
    ! CHECK: hlfir.assign %{{.*}} to %[[C]]#0
    c = c + 1
    ! CHECK: fir.load %[[N]]#0
    ! CHECK: arith.addi %{{.*}}, %c1
    ! CHECK: hlfir.assign %{{.*}} to %[[N]]#0
    n = n + 1
    ! CHECK: fir.load %[[N]]#0
    ! CHECK: fir.embox %[[T]]#0
    ! CHECK: fir.load %[[A]]#0
    ! CHECK: fir.load %[[B]]#0
    ! CHECK: fir.load %[[C]]#0
    ! CHECK: fir.load %[[D]]#0
    print*, n, t, a, b, c, d ! expect: 102 111 223 333 102 105 223 7
  end associate

  call nest

  do i=1,4
    associate (x=>i)
      ! 'x' is associated to 'i' directly via hlfir.declare
      ! CHECK: %[[X:.*]]:2 = hlfir.declare %[[I]]#0 {uniq_name = "_QFEx"}
      ! CHECK: %[[XVAL:.*]] = fir.load %[[X]]#0 : !fir.ref<i32>
      ! CHECK: %[[TWO:.*]] = arith.constant 2 : i32
      ! CHECK: arith.cmpi eq, %[[XVAL]], %[[TWO]] : i32
      ! CHECK: ^bb
      if (x==2) goto 9
      ! CHECK: %[[XVAL2:.*]] = fir.load %[[X]]#0 : !fir.ref<i32>
      ! CHECK: %[[THREE:.*]] = arith.constant 3 : i32
      ! CHECK: arith.cmpi eq, %[[XVAL2]], %[[THREE]] : i32
      ! CHECK: ^bb
      ! CHECK: fir.call @_FortranAStopStatementText
      ! CHECK: fir.unreachable
      ! CHECK: ^bb
      if (x==3) stop 'Halt'
      ! CHECK: fir.call @_FortranAioOutputAscii
      print*, "ok"
  9 end associate
  enddo
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
