! RUN: bbc -emit-hlfir -o - %s | FileCheck %s
  
! Test that no temporary copy is made for parameter/constant actual arguments
! when passed to INTENT(IN) dummy arguments. Since the callee guarantees not
! to modify the argument (F2018 8.5.10), there is no risk of copy-out writing
! into read-only memory, so the global address can be passed directly.

! Case 1: named PARAMETER array passed to INTENT(IN) assumed-shape dummy
subroutine test_param_array_intent_in()
  interface
    subroutine bar(x)
      integer, intent(in) :: x(:)
    end subroutine
  end interface
  integer, parameter :: p(5) = [1, 2, 3, 4, 5]
  call bar(p)
end subroutine

! CHECK-LABEL:   func.func @_QPtest_param_array_intent_in() {
! CHECK:            %0 = fir.dummy_scope : !fir.dscope
! CHECK:            %[[ADDR:.*]] = fir.address_of(@_QFtest_param_array_intent_inECp) : !fir.ref<!fir.array<5xi32>>
! CHECK:            %[[INDEX:.*]] = arith.constant 5 : index
! CHECK:            %[[SHAPE:.*]] = fir.shape %[[INDEX]] : (index) -> !fir.shape<1>
! CHECK:            %[[DECL:.*]]:2 = hlfir.declare %[[ADDR]](%[[SHAPE]]) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QFtest_param_array_intent_inECp"} : (!fir.ref<!fir.array<5xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<5xi32>>, !fir.ref<!fir.array<5xi32>>)
! CHECK:            %[[ADDR2:.*]] = fir.address_of(@_QQro.5xi4.0) : !fir.ref<!fir.array<5xi32>>
! CHECK:            %[[INDEX2:.*]] = arith.constant 5 : index
! CHECK:            %[[SHAPE2:.*]] = fir.shape %[[INDEX2]] : (index) -> !fir.shape<1>
! CHECK:            %[[DECL2:.*]]:2 = hlfir.declare %[[ADDR2]](%[[SHAPE2]]) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.5xi4.0"} : (!fir.ref<!fir.array<5xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<5xi32>>, !fir.ref<!fir.array<5xi32>>)
! CHECK:            %[[EMBOX:.*]] = fir.embox %[[DECL2]]#0(%[[SHAPE2]]) : (!fir.ref<!fir.array<5xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<5xi32>>
! CHECK:            %[[CONVERT:.*]] = fir.convert %[[EMBOX]] : (!fir.box<!fir.array<5xi32>>) -> !fir.box<!fir.array<?xi32>>
! CHECK:            fir.call @_QPbar(%[[CONVERT]]) fastmath<contract> : (!fir.box<!fir.array<?xi32>>) -> ()
! CHECK:            return
! CHECK:         }

! Case 2: string literal passed to INTENT(IN) character dummy
subroutine test_string_literal_intent_in()
  interface
    subroutine baz(msg)
      character(*), intent(in) :: msg
    end subroutine
  end interface
  call baz("hello world")
end subroutine
! CHECK-LABEL:   func.func @_QPtest_string_literal_intent_in() {
! CHECK:           %[[ADDR:.*]] = fir.address_of(@_QQ{{.*}}) : !fir.ref<!fir.char<1,11>>
! CHECK:           %[[DECL:.*]]:2 = hlfir.declare %[[ADDR]] {{.*}}{fortran_attrs = #fir.var_attrs<parameter>
! CHECK-NOT:       hlfir.as_expr
! CHECK-NOT:       hlfir.associate
! CHECK:           %[[BOXCHAR:.*]] = fir.emboxchar %[[DECL]]#0, %{{.*}} : (!fir.ref<!fir.char<1,11>>, index) -> !fir.boxchar<1>
! CHECK:           fir.call @_QPbaz(%[[BOXCHAR]])
! CHECK-NOT:       hlfir.end_associate
! CHECK:           return
! CHECK:         }

! Case 3: scalar PARAMETER passed to INTENT(IN) scalar dummy
subroutine test_param_scalar_intent_in()
  interface
    subroutine scalar_in(n)
      integer, intent(in) :: n
    end subroutine
  end interface
  integer, parameter :: k = 42
  call scalar_in(k)
end subroutine
! CHECK-LABEL:   func.func @_QPtest_param_scalar_intent_in() {
! CHECK:           %0 = fir.dummy_scope : !fir.dscope
! CHECK:           %[[ADDR:.*]]= fir.address_of(@_QFtest_param_scalar_intent_inECk) : !fir.ref<i32>
! CHECK:           %[[DECL:.*]]:2 = hlfir.declare %1 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QFtest_param_scalar_intent_inECk"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[CONST:.*]] = arith.constant 42 : i32
! CHECK:           %[[ASSO:.*]]:3 = hlfir.associate %[[CONST]] {adapt.valuebyref} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
! CHECK:           fir.call @_QPscalar_in(%[[ASSO]]#0) fastmath<contract> : (!fir.ref<i32>) -> ()
! CHECK:           hlfir.end_associate %[[ASSO]]#1, %[[ASSO]]#2 : !fir.ref<i32>, i1
! CHECK:           return
! CHECK:         }

! Case 4: constant array expression passed to dummy WITHOUT intent,copy still be made
subroutine test_param_array_no_intent()
  interface
    subroutine qux(x)
      integer :: x(:)
    end subroutine
  end interface
  integer, parameter :: p(3) = [10, 20, 30]
  call qux(p)
end subroutine
! CHECK-LABEL:   func.func @_QPtest_param_array_no_intent() {
! CHECK:            %0 = fir.dummy_scope : !fir.dscope
! CHECK:            %[[ADDR:.*]] = fir.address_of(@_QFtest_param_array_no_intentECp) : !fir.ref<!fir.array<3xi32>>
! CHECK:            %[[CONST:.*]] = arith.constant 3 : index
! CHECK:            %[[SHAPE:.*]] = fir.shape %[[CONST]] : (index) -> !fir.shape<1>
! CHECK:            %[[DECL:.*]]:2 = hlfir.declare %1(%2) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QFtest_param_array_no_intentECp"} : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3xi32>>, !fir.ref<!fir.array<3xi32>>)
! CHECK:            %[[ADDR2:.*]] = fir.address_of(@_QQro.3xi4.1) : !fir.ref<!fir.array<3xi32>>
! CHECK:            %[[CONST2:.*]] = arith.constant 3 : index
! CHECK:            %[[SHAPE2:.*]] = fir.shape %[[CONST2]] : (index) -> !fir.shape<1>
! CHECK:            %[[DECL2:.*]]:2 = hlfir.declare %[[ADDR2]](%[[SHAPE2]]) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.3xi4.1"} : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3xi32>>, !fir.ref<!fir.array<3xi32>>)
! CHECK:            %[[EXPR:.*]] = hlfir.as_expr %[[DECL2]]#0 : (!fir.ref<!fir.array<3xi32>>) -> !hlfir.expr<3xi32>
! CHECK:            %[[ASSO:.*]]:3 = hlfir.associate %[[EXPR]](%[[SHAPE2]]) {adapt.valuebyref} : (!hlfir.expr<3xi32>, !fir.shape<1>) -> (!fir.ref<!fir.array<3xi32>>, !fir.ref<!fir.array<3xi32>>, i1)
! CHECK:            %[[EMBOX:.*]] = fir.embox %[[ASSO]]#0(%[[SHAPE2]]) : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<3xi32>>
! CHECK:            %[[CONVERT:.*]] = fir.convert %[[EMBOX]] : (!fir.box<!fir.array<3xi32>>) -> !fir.box<!fir.array<?xi32>>
! CHECK:            fir.call @_QPqux(%[[CONVERT]]) fastmath<contract> : (!fir.box<!fir.array<?xi32>>) -> ()
! CHECK:            hlfir.end_associate %[[ASSO]]#1, %[[ASSO]]#2 : !fir.ref<!fir.array<3xi32>>, i1
! CHECK:            return
! CHECK:          }