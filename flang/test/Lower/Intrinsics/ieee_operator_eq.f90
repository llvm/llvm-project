! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s

! CHECK-LABEL: func.func @_QPs
subroutine s(r1,r2)
  use ieee_arithmetic, only: ieee_round_type, operator(==)
  type(ieee_round_type) :: r1, r2
  ! CHECK:   %[[R1DECL:.*]]:2 = hlfir.declare %arg0 {{.*}} {uniq_name = "_QFsEr1"}
  ! CHECK:   %[[R2DECL:.*]]:2 = hlfir.declare %arg1 {{.*}} {uniq_name = "_QFsEr2"}
  ! CHECK:   %[[V_6:[0-9]+]] = fir.coordinate_of %[[R1DECL]]#0, _QM__fortran_builtinsT__builtin_ieee_round_type.mode
  ! CHECK:   %[[V_7:[0-9]+]] = fir.coordinate_of %[[R2DECL]]#0, _QM__fortran_builtinsT__builtin_ieee_round_type.mode
  ! CHECK:   %[[V_8:[0-9]+]] = fir.load %[[V_6]] : !fir.ref<i8>
  ! CHECK:   %[[V_9:[0-9]+]] = fir.load %[[V_7]] : !fir.ref<i8>
  ! CHECK:   %[[V_10:[0-9]+]] = arith.cmpi eq, %[[V_8]], %[[V_9]] : i8
  ! CHECK:   fir.call @_FortranAioOutputLogical(%{{.*}}, %[[V_10]]) {{.*}} : (!fir.ref<i8>, i1) -> i1
  ! CHECK:   return
  ! CHECK: }
  print*, r1 == r2
end

! CHECK-LABEL: func.func @_QQmain
  use ieee_arithmetic, only: ieee_round_type, ieee_nearest, ieee_to_zero
  interface
    subroutine s(r1,r2)
      import ieee_round_type
      type(ieee_round_type) :: r1, r2
    end
  end interface

  ! CHECK:   hlfir.as_expr
  ! CHECK:   %[[ASSOC1:.*]]:3 = hlfir.associate {{.*}} {adapt.valuebyref}
  ! CHECK:   hlfir.as_expr
  ! CHECK:   %[[ASSOC2:.*]]:3 = hlfir.associate {{.*}} {adapt.valuebyref}
  ! CHECK:   fir.call @_QPs(%[[ASSOC1]]#0, %[[ASSOC2]]#0) {{.*}} : (!fir.ref<!fir.type<{{.*}}>>, !fir.ref<!fir.type<{{.*}}>>) -> ()
  call s(ieee_to_zero, ieee_nearest)

  ! CHECK:   hlfir.as_expr
  ! CHECK:   %[[ASSOC3:.*]]:3 = hlfir.associate {{.*}} {adapt.valuebyref}
  ! CHECK:   hlfir.as_expr
  ! CHECK:   %[[ASSOC4:.*]]:3 = hlfir.associate {{.*}} {adapt.valuebyref}
  ! CHECK:   fir.call @_QPs(%[[ASSOC3]]#0, %[[ASSOC4]]#0) {{.*}} : (!fir.ref<!fir.type<{{.*}}>>, !fir.ref<!fir.type<{{.*}}>>) -> ()
  call s(ieee_nearest, ieee_nearest)
end
