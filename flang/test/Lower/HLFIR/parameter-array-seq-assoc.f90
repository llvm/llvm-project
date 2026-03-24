! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! Test sequence association of PARAMETER array elements (F'2023 15.5.2.5p14).
! When a PARAMETER array element a(i) is passed to an explicit-shape dummy
! array, the element address must be passed directly so the callee can access
! subsequent elements via sequence association.  The element must NOT be copied
! into a scalar temporary first.

! CHECK-LABEL: func.func @_QQmain()

integer, parameter :: n = 5

integer, parameter :: a(n) = [1, 2, 3, 4, 5]
! Integer PARAMETER array declared with parameter attribute.
! CHECK: %[[A:.*]]:2 = hlfir.declare {{.*}} {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QFECa"} : (!fir.ref<!fir.array<5xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<5xi32>>, !fir.ref<!fir.array<5xi32>>)

character, parameter :: ch(n) = ['a', 'b', 'c', 'd', 'e']
! Character PARAMETER array declared with parameter attribute.
! CHECK: %[[CH:.*]]:2 = hlfir.declare {{.*}} typeparams {{.*}} {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QFECch"} : (!fir.ref<!fir.array<5x!fir.char<1>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<5x!fir.char<1>>>, !fir.ref<!fir.array<5x!fir.char<1>>>)

type :: mytype
  integer :: arr(n)
end type

type(mytype), parameter :: pt = mytype([1, 2, 3, 4, 5])
! PARAMETER of derived type with an integer array component.
! CHECK: %[[PT:.*]]:2 = hlfir.declare {{.*}} {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QFECpt"} : (!fir.ref<!fir.type<_QFTmytype{{.*}}>>) -> (!fir.ref<!fir.type<_QFTmytype{{.*}}>>, !fir.ref<!fir.type<_QFTmytype{{.*}}>>)

call passNint(a(1))
! Element address is obtained via hlfir.designate (subscript 1) and converted
! to an array ref for sequence association.  No scalar copy (hlfir.as_expr).
! CHECK-NOT: hlfir.as_expr
! CHECK: %[[ELEM1:.*]] = hlfir.designate %[[A]]#0 (%{{.*}})  : (!fir.ref<!fir.array<5xi32>>, i64) -> !fir.ref<i32>
! CHECK: %[[ARR1:.*]] = fir.convert %[[ELEM1]] : (!fir.ref<i32>) -> !fir.ref<!fir.array<5xi32>>
! CHECK: fir.call @_QFPpassnint(%[[ARR1]])

call passN2int(a(1 + 2))
! Element 3 address passed for sequence association with a 3-element dummy array.
! CHECK-NOT: hlfir.as_expr
! CHECK: %[[ELEM3:.*]] = hlfir.designate %[[A]]#0 (%{{.*}})  : (!fir.ref<!fir.array<5xi32>>, i64) -> !fir.ref<i32>
! CHECK: %[[ARR3:.*]] = fir.convert %[[ELEM3]] : (!fir.ref<i32>) -> !fir.ref<!fir.array<3xi32>>
! CHECK: fir.call @_QFPpassn2int(%[[ARR3]])

call passNchar(ch(1))
! Character element address is obtained via hlfir.designate and wrapped in a
! boxchar for the character dummy argument.  No scalar copy.
! CHECK-NOT: hlfir.as_expr
! CHECK: %[[CELEM1:.*]] = hlfir.designate %[[CH]]#0 (%{{.*}})  typeparams %{{.*}} : (!fir.ref<!fir.array<5x!fir.char<1>>>, i64, index) -> !fir.ref<!fir.char<1>>
! CHECK: %[[BOX1:.*]] = fir.emboxchar %[[CELEM1]], %{{.*}} : (!fir.ref<!fir.char<1>>, index) -> !fir.boxchar<1>
! CHECK: fir.call @_QFPpassnchar(%[[BOX1]])

call passN2char(ch(1 + 2))
! Character element 3 address passed for sequence association.
! CHECK-NOT: hlfir.as_expr
! CHECK: %[[CELEM3:.*]] = hlfir.designate %[[CH]]#0 (%{{.*}})  typeparams %{{.*}} : (!fir.ref<!fir.array<5x!fir.char<1>>>, i64, index) -> !fir.ref<!fir.char<1>>
! CHECK: %[[BOX3:.*]] = fir.emboxchar %[[CELEM3]], %{{.*}} : (!fir.ref<!fir.char<1>>, index) -> !fir.boxchar<1>
! CHECK: fir.call @_QFPpassn2char(%[[BOX3]])

call passNint(pt%arr(1))
! Component arr(1) of PARAMETER derived type: element address reused for sequence
! association.  No scalar copy.
! CHECK-NOT: hlfir.as_expr
! CHECK: %[[PELEM1:.*]] = hlfir.designate %[[PT]]#0{"arr"} <%{{.*}}> (%{{.*}})  : (!fir.ref<!fir.type<_QFTmytype{{.*}}>>, !fir.shape<1>, index) -> !fir.ref<i32>
! CHECK: %[[PARR1:.*]] = fir.convert %[[PELEM1]] : (!fir.ref<i32>) -> !fir.ref<!fir.array<5xi32>>
! CHECK: fir.call @_QFPpassnint(%[[PARR1]])

call passN2int(pt%arr(1 + 2))
! Component arr(3) element address passed for sequence association with a 3-element dummy.
! CHECK-NOT: hlfir.as_expr
! CHECK: %[[PELEM3:.*]] = hlfir.designate %[[PT]]#0{"arr"} <%{{.*}}> (%{{.*}})  : (!fir.ref<!fir.type<_QFTmytype{{.*}}>>, !fir.shape<1>, index) -> !fir.ref<i32>
! CHECK: %[[PARR3:.*]] = fir.convert %[[PELEM3]] : (!fir.ref<i32>) -> !fir.ref<!fir.array<3xi32>>
! CHECK: fir.call @_QFPpassn2int(%[[PARR3]])

contains

subroutine passNint(b)
  integer, intent(in) :: b(n)
end subroutine

subroutine passN2int(b)
  integer, intent(in) :: b(n - 2)
end subroutine

subroutine passNchar(b)
  character, intent(in) :: b(n)
end subroutine

subroutine passN2char(b)
  character, intent(in) :: b(n - 2)
end subroutine

end
