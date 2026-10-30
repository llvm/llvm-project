! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -O2 -emit-llvm %s -o - | FileCheck --check-prefix=LLVM %s

! The sequence view created for VALUE sequence-association copies must keep
! the source VOLATILE qualification so the copy reads the element sequence
! with volatile accesses. The optimized one-element control pins the actual
! volatile load; at -O0 the copy is a descriptor-based runtime assignment,
! so the guarantee at that level is the qualification of the IR types.

! One-element control (correctly sized even before the sequence view
! existed): the qualifier must survive the view and the source load must
! stay volatile.
! CHECK-LABEL: func.func @_QPvolatile_one
! CHECK: %[[ELT1:.*]] = hlfir.designate %{{.*}} (%{{.*}})  : (!fir.ref<!fir.array<3xi32>, volatile>, index) -> !fir.ref<i32, volatile>
! CHECK: %[[SEQ1:.*]] = fir.convert %[[ELT1]] : (!fir.ref<i32, volatile>) -> !fir.ref<!fir.array<1xi32>, volatile>
! CHECK: %[[VIEW1:.*]]:2 = hlfir.declare %[[SEQ1]](%{{.*}}) {uniq_name = ".sequence.assoc"} : (!fir.ref<!fir.array<1xi32>, volatile>, !fir.shape<1>) -> (!fir.ref<!fir.array<1xi32>, volatile>, !fir.ref<!fir.array<1xi32>, volatile>)
! CHECK: hlfir.as_expr %[[VIEW1]]#0 : (!fir.ref<!fir.array<1xi32>, volatile>) -> !hlfir.expr<1xi32>
! LLVM-LABEL: define {{.*}}@volatile_one_
! LLVM: load volatile i32
subroutine volatile_one(a)
  integer, volatile :: a(3)
  interface
    subroutine sub1(x)
      integer, value :: x(1)
    end subroutine
  end interface
  call sub1(a(2))
end subroutine

! Static multi-element view keeps the qualifier: all three element reads
! stay volatile in the optimized copy.
! LLVM-LABEL: define {{.*}}@volatile_static_
! LLVM: load volatile i32
! LLVM: load volatile i32
! LLVM: load volatile i32
! CHECK-LABEL: func.func @_QPvolatile_static
! CHECK: %[[ELT3:.*]] = hlfir.designate %{{.*}} (%{{.*}})  : (!fir.ref<!fir.array<4xi32>, volatile>, index) -> !fir.ref<i32, volatile>
! CHECK: %[[SEQ3:.*]] = fir.convert %[[ELT3]] : (!fir.ref<i32, volatile>) -> !fir.ref<!fir.array<3xi32>, volatile>
! CHECK: %[[VIEW3:.*]]:2 = hlfir.declare %[[SEQ3]](%{{.*}}) {uniq_name = ".sequence.assoc"} : (!fir.ref<!fir.array<3xi32>, volatile>, !fir.shape<1>) -> (!fir.ref<!fir.array<3xi32>, volatile>, !fir.ref<!fir.array<3xi32>, volatile>)
! CHECK: hlfir.as_expr %[[VIEW3]]#0 : (!fir.ref<!fir.array<3xi32>, volatile>) -> !hlfir.expr<3xi32>
subroutine volatile_static(a)
  integer, volatile :: a(4)
  interface
    subroutine sub3(x)
      integer, value :: x(3)
    end subroutine
  end interface
  call sub3(a(2))
end subroutine

! Runtime-shaped remaining-sequence view keeps the qualifier, including on
! the boxed view; the optimized copy loop reads volatile.
! LLVM-LABEL: define {{.*}}@volatile_dynamic_
! LLVM: load volatile i32
! CHECK-LABEL: func.func @_QPvolatile_dynamic
! CHECK: %[[ELTN:.*]] = hlfir.designate %{{.*}} (%{{.*}})  : (!fir.ref<!fir.array<4xi32>, volatile>, index) -> !fir.ref<i32, volatile>
! CHECK: %[[SEQN:.*]] = fir.convert %[[ELTN]] : (!fir.ref<i32, volatile>) -> !fir.ref<!fir.array<?xi32>, volatile>
! CHECK: %[[VIEWN:.*]]:2 = hlfir.declare %[[SEQN]](%{{.*}}) {uniq_name = ".sequence.assoc"} : (!fir.ref<!fir.array<?xi32>, volatile>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>, volatile>, !fir.ref<!fir.array<?xi32>, volatile>)
! CHECK: hlfir.as_expr %[[VIEWN]]#0 : (!fir.box<!fir.array<?xi32>, volatile>) -> !hlfir.expr<?xi32>
subroutine volatile_dynamic(a, n)
  integer, volatile :: a(4)
  integer :: n
  interface
    subroutine subn(n, x)
      integer, intent(in) :: n
      integer, value :: x(n)
    end subroutine
  end interface
  call subn(n, a(2))
end subroutine
