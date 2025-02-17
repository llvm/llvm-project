! Test questionable but existing abuses of implicit interfaces.
! Lowering must close the eyes and do as if it did not know
! about the function definition since semantic lets these
! programs through with a warning.
! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

subroutine takes_char(c)
  character(8) :: c
end subroutine

subroutine pass_real_to_char(r)
  real(8) :: r
  call takes_char(r)
end subroutine
! CHECK-LABEL: func.func @_QPpass_real_to_char(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare {{.*}}Er
! CHECK:  %[[VAL_2:.*]] = fir.address_of(@_QPtakes_char) : (!fir.boxchar<1>) -> ()
! CHECK:  %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : ((!fir.boxchar<1>) -> ()) -> ((!fir.ref<f64>) -> ())
! CHECK:  fir.call %[[VAL_3]](%[[VAL_1]]#1) {{.*}}: (!fir.ref<f64>) -> ()

subroutine pass_char_proc_to_char()
  character(8), external :: char_proc
  call takes_char(char_proc)
end subroutine
! CHECK:  %[[VAL_0:.*]] = fir.address_of(@_QPchar_proc) : (!fir.ref<!fir.char<1,8>>, index) -> !fir.boxchar<1>
! CHECK:  %[[VAL_1:.*]] = arith.constant 8 : i64
! CHECK:  %[[VAL_2:.*]] = fir.emboxproc %[[VAL_0]] : ((!fir.ref<!fir.char<1,8>>, index) -> !fir.boxchar<1>) -> !fir.boxproc<() -> ()>
! CHECK:  %[[VAL_3:.*]] = fir.undefined tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_4:.*]] = fir.insert_value %[[VAL_3]], %[[VAL_2]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxproc<() -> ()>) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_5:.*]] = fir.insert_value %[[VAL_4]], %[[VAL_1]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, i64) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_6:.*]] = fir.address_of(@_QPtakes_char) : (!fir.boxchar<1>) -> ()
! CHECK:  %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : ((!fir.boxchar<1>) -> ()) -> ((tuple<!fir.boxproc<() -> ()>, i64>) -> ())
! CHECK:  fir.call %[[VAL_7]](%[[VAL_5]]) {{.*}}: (tuple<!fir.boxproc<() -> ()>, i64>) -> ()

subroutine pass_kind2_char_to_char(c)
  character(4, kind=2) :: c
  call takes_char(c)
end subroutine
! CHECK-LABEL: func.func @_QPpass_kind2_char_to_char(
! CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare {{.*}}Ec
! CHECK:  %[[VAL_3:.*]] = fir.emboxchar %[[VAL_2]]#1, %{{.*}} : (!fir.ref<!fir.char<2,4>>, index) -> !fir.boxchar<2>
! CHECK:  %[[VAL_4:.*]] = fir.address_of(@_QPtakes_char) : (!fir.boxchar<1>) -> ()
! CHECK:  %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : ((!fir.boxchar<1>) -> ()) -> ((!fir.boxchar<2>) -> ())
! CHECK:  fir.call %[[VAL_5]](%[[VAL_3]]) {{.*}}: (!fir.boxchar<2>) -> ()

subroutine takes_real(r)
  real(8) :: r
end subroutine

subroutine pass_int_to_real(i)
  integer(8) :: i
  call takes_real(i)
end subroutine
! CHECK-LABEL: func.func @_QPpass_int_to_real(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare {{.*}}Ei
! CHECK:  %[[VAL_2:.*]] = fir.convert %[[VAL_1]]#1 : (!fir.ref<i64>) -> !fir.ref<f64>
! CHECK:  fir.call @_QPtakes_real(%[[VAL_2]]) {{.*}}: (!fir.ref<f64>) -> ()

subroutine pass_char_to_real(c)
  character(8) :: c
  call takes_real(c)
end subroutine
! CHECK-LABEL: func.func @_QPpass_char_to_real(
! CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare {{.*}}Ec
! CHECK:  %[[VAL_3:.*]] = fir.emboxchar %[[VAL_2]]#1, %{{.*}} : (!fir.ref<!fir.char<1,8>>, index) -> !fir.boxchar<1>
! CHECK:  %[[VAL_4:.*]] = fir.address_of(@_QPtakes_real) : (!fir.ref<f64>) -> ()
! CHECK:  %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : ((!fir.ref<f64>) -> ()) -> ((!fir.boxchar<1>) -> ())
! CHECK:  fir.call %[[VAL_5]](%[[VAL_3]]) {{.*}}: (!fir.boxchar<1>) -> ()

subroutine pass_proc_to_real()
  real(8), external :: proc
  call takes_real(proc)
end subroutine
! CHECK-LABEL: func.func @_QPpass_proc_to_real() {
! CHECK:  %[[VAL_0:.*]] = fir.address_of(@_QPproc) : () -> f64
! CHECK:  %[[VAL_1:.*]] = fir.emboxproc %[[VAL_0]] : (() -> f64) -> !fir.boxproc<() -> ()>
! CHECK:  %[[VAL_2:.*]] = fir.address_of(@_QPtakes_real) : (!fir.ref<f64>) -> ()
! CHECK:  %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : ((!fir.ref<f64>) -> ()) -> ((!fir.boxproc<() -> ()>) -> ())
! CHECK:  fir.call %[[VAL_3]](%[[VAL_1]]) {{.*}}: (!fir.boxproc<() -> ()>) -> ()

subroutine pass_complex_to_real(cmplx)
  complex(4) :: cmplx
  call takes_real(cmplx)
end subroutine
! CHECK-LABEL: func.func @_QPpass_complex_to_real(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare {{.*}}Ecmplx
! CHECK:  %[[VAL_2:.*]] = fir.convert %[[VAL_1]]#1 : (!fir.ref<complex<f32>>) -> !fir.ref<f64>
! CHECK:  fir.call @_QPtakes_real(%[[VAL_2]]) {{.*}}: (!fir.ref<f64>) -> ()

subroutine takes_char_proc(c)
  character(8), external :: c
end subroutine

subroutine pass_proc_to_char_proc()
  external :: proc
  call takes_char_proc(proc)
end subroutine
! CHECK-LABEL: func.func @_QPpass_proc_to_char_proc() {
! CHECK:  %[[VAL_0:.*]] = fir.address_of(@_QPproc) : () -> f64
! CHECK:  %[[VAL_1:.*]] = fir.emboxproc %[[VAL_0]] : (() -> f64) -> !fir.boxproc<() -> ()>
! CHECK:  %[[VAL_2:.*]] = fir.address_of(@_QPtakes_char_proc) : (tuple<!fir.boxproc<() -> ()>, i64>) -> ()
! CHECK:  %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : ((tuple<!fir.boxproc<() -> ()>, i64>) -> ()) -> ((!fir.boxproc<() -> ()>) -> ())
! CHECK:  fir.call %[[VAL_3]](%[[VAL_1]]) {{.*}}: (!fir.boxproc<() -> ()>) -> ()

subroutine pass_char_to_char_proc(c)
  character(8) :: c
  call takes_char_proc(c)
end subroutine
! CHECK-LABEL: func.func @_QPpass_char_to_char_proc(
! CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare {{.*}}Ec
! CHECK:  %[[VAL_3:.*]] = fir.emboxchar %[[VAL_2]]#1, %{{.*}} : (!fir.ref<!fir.char<1,8>>, index) -> !fir.boxchar<1>
! CHECK:  %[[VAL_4:.*]] = fir.address_of(@_QPtakes_char_proc) : (tuple<!fir.boxproc<() -> ()>, i64>) -> ()
! CHECK:  %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : ((tuple<!fir.boxproc<() -> ()>, i64>) -> ()) -> ((!fir.boxchar<1>) -> ())
! CHECK:  fir.call %[[VAL_5]](%[[VAL_3]]) {{.*}}: (!fir.boxchar<1>) -> ()

subroutine takes_proc(proc)
  real(8), external :: proc
end subroutine

subroutine pass_char_proc_to_proc()
  character(8), external :: char_proc
  call takes_proc(char_proc)
end subroutine
! CHECK-LABEL: func.func @_QPpass_char_proc_to_proc() {
! CHECK:  %[[VAL_0:.*]] = fir.address_of(@_QPchar_proc) : (!fir.ref<!fir.char<1,8>>, index) -> !fir.boxchar<1>
! CHECK:  %[[VAL_1:.*]] = arith.constant 8 : i64
! CHECK:  %[[VAL_2:.*]] = fir.emboxproc %[[VAL_0]] : ((!fir.ref<!fir.char<1,8>>, index) -> !fir.boxchar<1>) -> !fir.boxproc<() -> ()>
! CHECK:  %[[VAL_3:.*]] = fir.undefined tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_4:.*]] = fir.insert_value %[[VAL_3]], %[[VAL_2]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxproc<() -> ()>) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_5:.*]] = fir.insert_value %[[VAL_4]], %[[VAL_1]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, i64) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_6:.*]] = fir.address_of(@_QPtakes_proc) : (!fir.boxproc<() -> ()>) -> ()
! CHECK:  %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : ((!fir.boxproc<() -> ()>) -> ()) -> ((tuple<!fir.boxproc<() -> ()>, i64>) -> ())
! CHECK:  fir.call %[[VAL_7]](%[[VAL_5]]) {{.*}}: (tuple<!fir.boxproc<() -> ()>, i64>) -> ()

subroutine pass_real_to_proc(r)
  real(8) :: r
  call takes_proc(r)
end subroutine
! CHECK-LABEL: func.func @_QPpass_real_to_proc(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare {{.*}}Er
! CHECK:  %[[VAL_2:.*]] = fir.address_of(@_QPtakes_proc) : (!fir.boxproc<() -> ()>) -> ()
! CHECK:  %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : ((!fir.boxproc<() -> ()>) -> ()) -> ((!fir.ref<f64>) -> ())
! CHECK:  fir.call %[[VAL_3]](%[[VAL_1]]#1) {{.*}}: (!fir.ref<f64>) -> ()

subroutine pass_too_many_args()
  call takes_real(I, Kown, what, I, am, doing)
end subroutine
! CHECK-LABEL: func.func @_QPpass_too_many_args() {
! CHECK:  %[[VAL_10:.*]] = fir.address_of(@_QPtakes_real) : (!fir.ref<f64>) -> ()
! CHECK:  %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : ((!fir.ref<f64>) -> ()) -> ((!fir.ref<i32>, !fir.ref<i32>, !fir.ref<f32>, !fir.ref<i32>, !fir.ref<f32>, !fir.ref<f32>) -> ())
! CHECK:  fir.call %[[VAL_11]](%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<i32>, !fir.ref<i32>, !fir.ref<f32>, !fir.ref<i32>, !fir.ref<f32>, !fir.ref<f32>) -> ()

subroutine pass_too_few_args()
  call takes_real()
end subroutine
! CHECK-LABEL: func.func @_QPpass_too_few_args() {
! CHECK:  %[[VAL_0:.*]] = fir.address_of(@_QPtakes_real) : (!fir.ref<f64>) -> ()
! CHECK:  %[[VAL_1:.*]] = fir.convert %[[VAL_0]] : ((!fir.ref<f64>) -> ()) -> (() -> ())
! CHECK:  fir.call %[[VAL_1]]() {{.*}}: () -> ()
