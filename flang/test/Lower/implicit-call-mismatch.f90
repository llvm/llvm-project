! Test questionable but existing abuses of implicit interfaces.
! Lowering must close the eyes and do as if it did not know
! about the function definition since semantic lets these
! programs through with a warning.
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! Test reference to non-char procedure conversion.

subroutine takes_proc(proc)
  real(8), external :: proc
end subroutine

subroutine pass_int_to_proc(a)
  integer(4) :: a
  call takes_proc(a)
end subroutine
! CHECK-LABEL: func.func @_QPpass_int_to_proc(
! CHECK: %[[VAL_addr:.*]] = fir.address_of(@_QPtakes_proc) : (!fir.boxproc<() -> ()>) -> ()
! CHECK: %[[VAL_conv:.*]] = fir.convert %[[VAL_addr]] : ((!fir.boxproc<() -> ()>) -> ()) -> ((!fir.ref<i32>) -> ())
! CHECK: fir.call %[[VAL_conv]]({{.*}}) {{.*}}: (!fir.ref<i32>) -> ()

subroutine pass_logical_to_proc(a)
  logical(4) :: a
  call takes_proc(a)
end subroutine
! CHECK-LABEL: func.func @_QPpass_logical_to_proc(
! CHECK: %[[VAL_addr:.*]] = fir.address_of(@_QPtakes_proc) : (!fir.boxproc<() -> ()>) -> ()
! CHECK: %[[VAL_conv:.*]] = fir.convert %[[VAL_addr]] : ((!fir.boxproc<() -> ()>) -> ()) -> ((!fir.ref<!fir.logical<4>>) -> ())
! CHECK: fir.call %[[VAL_conv]]({{.*}}) {{.*}}: (!fir.ref<!fir.logical<4>>) -> ()

subroutine pass_real_to_proc(a)
  real(8) :: a
  call takes_proc(a)
end subroutine
! CHECK-LABEL: func.func @_QPpass_real_to_proc(
! CHECK: %[[VAL_addr:.*]] = fir.address_of(@_QPtakes_proc) : (!fir.boxproc<() -> ()>) -> ()
! CHECK: %[[VAL_conv:.*]] = fir.convert %[[VAL_addr]] : ((!fir.boxproc<() -> ()>) -> ()) -> ((!fir.ref<f64>) -> ())
! CHECK: fir.call %[[VAL_conv]]({{.*}}) {{.*}}: (!fir.ref<f64>) -> ()

subroutine pass_complex_to_proc(a)
  complex(4) :: a
  call takes_proc(a)
end subroutine
! CHECK-LABEL: func.func @_QPpass_complex_to_proc(
! CHECK: %[[VAL_addr:.*]] = fir.address_of(@_QPtakes_proc) : (!fir.boxproc<() -> ()>) -> ()
! CHECK: %[[VAL_conv:.*]] = fir.convert %[[VAL_addr]] : ((!fir.boxproc<() -> ()>) -> ()) -> ((!fir.ref<complex<f32>>) -> ())
! CHECK: fir.call %[[VAL_conv]]({{.*}}) {{.*}}: (!fir.ref<complex<f32>>) -> ()

subroutine pass_char_to_proc(a)
  character(8) :: a
  call takes_proc(a)
end subroutine
! CHECK-LABEL: func.func @_QPpass_char_to_proc(
! CHECK: %[[VAL_addr:.*]] = fir.address_of(@_QPtakes_proc) : (!fir.boxproc<() -> ()>) -> ()
! CHECK: %[[VAL_conv:.*]] = fir.convert %[[VAL_addr]] : ((!fir.boxproc<() -> ()>) -> ()) -> ((!fir.boxchar<1>) -> ())
! CHECK: fir.call %[[VAL_conv]]({{.*}}) {{.*}}: (!fir.boxchar<1>) -> ()

subroutine pass_dt_to_proc(a)
  type :: dt
    integer(4) :: i, j
  end type
  type(dt) :: a

  call takes_proc(a)
end subroutine
! CHECK-LABEL: func.func @_QPpass_dt_to_proc(
! CHECK: %[[VAL_addr:.*]] = fir.address_of(@_QPtakes_proc) : (!fir.boxproc<() -> ()>) -> ()
! CHECK: %[[VAL_conv:.*]] = fir.convert %[[VAL_addr]] : ((!fir.boxproc<() -> ()>) -> ()) -> ((!fir.ref<!fir.type<{{.*}}>>) -> ())
! CHECK: fir.call %[[VAL_conv]]({{.*}}) {{.*}}: (!fir.ref<!fir.type<{{.*}}>>) -> ()

subroutine pass_array_to_proc(a)
  integer(4) :: a(10,10)
  call takes_proc(a)
end subroutine
! CHECK-LABEL: func.func @_QPpass_array_to_proc(
! CHECK: %[[VAL_addr:.*]] = fir.address_of(@_QPtakes_proc) : (!fir.boxproc<() -> ()>) -> ()
! CHECK: %[[VAL_conv:.*]] = fir.convert %[[VAL_addr]] : ((!fir.boxproc<() -> ()>) -> ()) -> ((!fir.ref<!fir.array<10x10xi32>>) -> ())
! CHECK: fir.call %[[VAL_conv]]({{.*}}) {{.*}}: (!fir.ref<!fir.array<10x10xi32>>) -> ()

! Test procedure conversion.

subroutine pass_char_proc_to_proc(a)
  character(8), external :: a
  call takes_proc(a)
end subroutine
! CHECK-LABEL: func.func @_QPpass_char_proc_to_proc(
! CHECK: %[[VAL_addr:.*]] = fir.address_of(@_QPtakes_proc) : (!fir.boxproc<() -> ()>) -> ()
! CHECK: %[[VAL_conv:.*]] = fir.convert %[[VAL_addr]] : ((!fir.boxproc<() -> ()>) -> ()) -> ((tuple<!fir.boxproc<() -> ()>, i64>) -> ())
! CHECK: fir.call %[[VAL_conv]]({{.*}}) {{.*}}: (tuple<!fir.boxproc<() -> ()>, i64>) -> ()

subroutine pass_proc_to_proc(a)
  external :: a
  call takes_proc(a)
end subroutine
! CHECK-LABEL: func.func @_QPpass_proc_to_proc(
! CHECK: fir.call @_QPtakes_proc({{.*}}) {{.*}}: (!fir.boxproc<() -> ()>) -> ()

! Test conversion from procedure to other data types.

! CHECK-LABEL: func.func @_QPtest_conversion_from_proc(
subroutine test_conversion_from_proc
  external :: proc

  ! CHECK: %[[proc:.*]] = fir.address_of(@_QPproc) : () -> ()
  ! CHECK: %[[boxproc:.*]] = fir.emboxproc %[[proc]] : (() -> ()) -> !fir.boxproc<() -> ()>
  ! CHECK: %[[callee:.*]] = fir.address_of(@_QPpass_int_to_proc)
  ! CHECK: %[[convert:.*]] = fir.convert %[[callee]]
  ! CHECK: fir.call %[[convert]](%[[boxproc]])
  call pass_int_to_proc(proc)

  ! CHECK: %[[proc:.*]] = fir.address_of(@_QPproc) : () -> ()
  ! CHECK: %[[boxproc:.*]] = fir.emboxproc %[[proc]] : (() -> ()) -> !fir.boxproc<() -> ()>
  ! CHECK: %[[callee:.*]] = fir.address_of(@_QPpass_logical_to_proc)
  ! CHECK: %[[convert:.*]] = fir.convert %[[callee]]
  ! CHECK: fir.call %[[convert]](%[[boxproc]])
  call pass_logical_to_proc(proc)

  ! CHECK: %[[proc:.*]] = fir.address_of(@_QPproc) : () -> ()
  ! CHECK: %[[boxproc:.*]] = fir.emboxproc %[[proc]] : (() -> ()) -> !fir.boxproc<() -> ()>
  ! CHECK: %[[callee:.*]] = fir.address_of(@_QPpass_real_to_proc)
  ! CHECK: %[[convert:.*]] = fir.convert %[[callee]]
  ! CHECK: fir.call %[[convert]](%[[boxproc]])
  call pass_real_to_proc(proc)

  ! CHECK: %[[proc:.*]] = fir.address_of(@_QPproc) : () -> ()
  ! CHECK: %[[boxproc:.*]] = fir.emboxproc %[[proc]] : (() -> ()) -> !fir.boxproc<() -> ()>
  ! CHECK: %[[callee:.*]] = fir.address_of(@_QPpass_complex_to_proc)
  ! CHECK: %[[convert:.*]] = fir.convert %[[callee]]
  ! CHECK: fir.call %[[convert]](%[[boxproc]])
  call pass_complex_to_proc(proc)

  ! CHECK: %[[proc:.*]] = fir.address_of(@_QPproc) : () -> ()
  ! CHECK: %[[boxproc:.*]] = fir.emboxproc %[[proc]] : (() -> ()) -> !fir.boxproc<() -> ()>
  ! CHECK: %[[callee:.*]] = fir.address_of(@_QPpass_char_to_proc)
  ! CHECK: %[[convert:.*]] = fir.convert %[[callee]]
  ! CHECK: fir.call %[[convert]](%[[boxproc]])
  call pass_char_to_proc(proc)

  ! CHECK: %[[proc:.*]] = fir.address_of(@_QPproc) : () -> ()
  ! CHECK: %[[boxproc:.*]] = fir.emboxproc %[[proc]] : (() -> ()) -> !fir.boxproc<() -> ()>
  ! CHECK: %[[callee:.*]] = fir.address_of(@_QPpass_dt_to_proc)
  ! CHECK: %[[convert:.*]] = fir.convert %[[callee]]
  ! CHECK: fir.call %[[convert]](%[[boxproc]])
  call pass_dt_to_proc(proc)

  ! CHECK: %[[proc:.*]] = fir.address_of(@_QPproc) : () -> ()
  ! CHECK: %[[boxproc:.*]] = fir.emboxproc %[[proc]] : (() -> ()) -> !fir.boxproc<() -> ()>
  ! CHECK: %[[callee:.*]] = fir.address_of(@_QPpass_array_to_proc)
  ! CHECK: %[[convert:.*]] = fir.convert %[[callee]]
  ! CHECK: fir.call %[[convert]](%[[boxproc]])
  call pass_array_to_proc(proc)

  ! CHECK: %[[proc:.*]] = fir.address_of(@_QPproc) : () -> ()
  ! CHECK: %[[boxProc:.*]] = fir.emboxproc %[[proc]] : (() -> ()) -> !fir.boxproc<() -> ()>
  ! CHECK: %[[callee:.*]] = fir.address_of(@_QPpass_char_proc_to_proc)
  ! CHECK: %[[convert:.*]] = fir.convert %[[callee]]
  ! CHECK: fir.call %[[convert]](%[[boxProc]])
  call pass_char_proc_to_proc(proc)

  ! CHECK: %[[proc:.*]] = fir.address_of(@_QPproc) : () -> ()
  ! CHECK: %[[box:.*]] = fir.emboxproc %[[proc]] : (() -> ()) -> !fir.boxproc<() -> ()>
  ! CHECK: fir.call @_QPpass_proc_to_proc(%[[box]])
  call pass_proc_to_proc(proc)
end subroutine

! Test reference to char procedure conversion.

subroutine takes_char_proc(cp)
  character(8), external :: cp
end subroutine

subroutine pass_int_to_char_proc(a)
  integer(4) :: a
  call takes_char_proc(a)
end subroutine
! CHECK-LABEL: func.func @_QPpass_int_to_char_proc(
! CHECK: %[[VAL_addr:.*]] = fir.address_of(@_QPtakes_char_proc) : (tuple<!fir.boxproc<() -> ()>, i64>) -> ()
! CHECK: %[[VAL_conv:.*]] = fir.convert %[[VAL_addr]] : ((tuple<!fir.boxproc<() -> ()>, i64>) -> ()) -> ((!fir.ref<i32>) -> ())
! CHECK: fir.call %[[VAL_conv]]({{.*}}) {{.*}}: (!fir.ref<i32>) -> ()

subroutine pass_logical_to_char_proc(a)
  logical(4) :: a
  call takes_char_proc(a)
end subroutine
! CHECK-LABEL: func.func @_QPpass_logical_to_char_proc(
! CHECK: %[[VAL_addr:.*]] = fir.address_of(@_QPtakes_char_proc) : (tuple<!fir.boxproc<() -> ()>, i64>) -> ()
! CHECK: %[[VAL_conv:.*]] = fir.convert %[[VAL_addr]] : ((tuple<!fir.boxproc<() -> ()>, i64>) -> ()) -> ((!fir.ref<!fir.logical<4>>) -> ())
! CHECK: fir.call %[[VAL_conv]]({{.*}}) {{.*}}: (!fir.ref<!fir.logical<4>>) -> ()

subroutine pass_real_to_char_proc(a)
  real(8) :: a
  call takes_char_proc(a)
end subroutine
! CHECK-LABEL: func.func @_QPpass_real_to_char_proc(
! CHECK: %[[VAL_addr:.*]] = fir.address_of(@_QPtakes_char_proc) : (tuple<!fir.boxproc<() -> ()>, i64>) -> ()
! CHECK: %[[VAL_conv:.*]] = fir.convert %[[VAL_addr]] : ((tuple<!fir.boxproc<() -> ()>, i64>) -> ()) -> ((!fir.ref<f64>) -> ())
! CHECK: fir.call %[[VAL_conv]]({{.*}}) {{.*}}: (!fir.ref<f64>) -> ()

subroutine pass_complex_to_char_proc(a)
  complex(4) :: a
  call takes_char_proc(a)
end subroutine
! CHECK-LABEL: func.func @_QPpass_complex_to_char_proc(
! CHECK: %[[VAL_addr:.*]] = fir.address_of(@_QPtakes_char_proc) : (tuple<!fir.boxproc<() -> ()>, i64>) -> ()
! CHECK: %[[VAL_conv:.*]] = fir.convert %[[VAL_addr]] : ((tuple<!fir.boxproc<() -> ()>, i64>) -> ()) -> ((!fir.ref<complex<f32>>) -> ())
! CHECK: fir.call %[[VAL_conv]]({{.*}}) {{.*}}: (!fir.ref<complex<f32>>) -> ()

subroutine pass_char_to_char_proc(a)
  character(8) :: a
  call takes_char_proc(a)
end subroutine
! CHECK-LABEL: func.func @_QPpass_char_to_char_proc(
! CHECK: %[[VAL_addr:.*]] = fir.address_of(@_QPtakes_char_proc) : (tuple<!fir.boxproc<() -> ()>, i64>) -> ()
! CHECK: %[[VAL_conv:.*]] = fir.convert %[[VAL_addr]] : ((tuple<!fir.boxproc<() -> ()>, i64>) -> ()) -> ((!fir.boxchar<1>) -> ())
! CHECK: fir.call %[[VAL_conv]]({{.*}}) {{.*}}: (!fir.boxchar<1>) -> ()

subroutine pass_dt_to_char_proc(a)
  type :: dt
    integer(4) :: i, j
  end type
  type(dt) :: a

  call takes_char_proc(a)
end subroutine
! CHECK-LABEL: func.func @_QPpass_dt_to_char_proc(
! CHECK: %[[VAL_addr:.*]] = fir.address_of(@_QPtakes_char_proc) : (tuple<!fir.boxproc<() -> ()>, i64>) -> ()
! CHECK: %[[VAL_conv:.*]] = fir.convert %[[VAL_addr]] : ((tuple<!fir.boxproc<() -> ()>, i64>) -> ()) -> ((!fir.ref<!fir.type<{{.*}}>>) -> ())
! CHECK: fir.call %[[VAL_conv]]({{.*}}) {{.*}}: (!fir.ref<!fir.type<{{.*}}>>) -> ()

subroutine pass_array_to_char_proc(a)
  integer(4) :: a(10,10)
  call takes_char_proc(a)
end subroutine
! CHECK-LABEL: func.func @_QPpass_array_to_char_proc(
! CHECK: %[[VAL_addr:.*]] = fir.address_of(@_QPtakes_char_proc) : (tuple<!fir.boxproc<() -> ()>, i64>) -> ()
! CHECK: %[[VAL_conv:.*]] = fir.convert %[[VAL_addr]] : ((tuple<!fir.boxproc<() -> ()>, i64>) -> ()) -> ((!fir.ref<!fir.array<10x10xi32>>) -> ())
! CHECK: fir.call %[[VAL_conv]]({{.*}}) {{.*}}: (!fir.ref<!fir.array<10x10xi32>>) -> ()

! Test procedure conversion.

subroutine pass_proc_to_char_proc(a)
  external :: a
  call takes_char_proc(a)
end subroutine
! CHECK-LABEL: func.func @_QPpass_proc_to_char_proc(
! CHECK: %[[VAL_addr:.*]] = fir.address_of(@_QPtakes_char_proc) : (tuple<!fir.boxproc<() -> ()>, i64>) -> ()
! CHECK: %[[VAL_conv:.*]] = fir.convert %[[VAL_addr]] : ((tuple<!fir.boxproc<() -> ()>, i64>) -> ()) -> ((!fir.boxproc<() -> ()>) -> ())
! CHECK: fir.call %[[VAL_conv]]({{.*}}) {{.*}}: (!fir.boxproc<() -> ()>) -> ()

subroutine pass_char_proc_to_char_proc(a)
  character(8), external :: a
  call takes_char_proc(a)
end subroutine
! CHECK-LABEL: func.func @_QPpass_char_proc_to_char_proc(
! CHECK: fir.call @_QPtakes_char_proc({{.*}}) {{.*}}: (tuple<!fir.boxproc<() -> ()>, i64>) -> ()

! Test conversion from character procedure to other data types.

! CHECK-LABEL: func.func @_QPtest_conversion_from_char_proc(
subroutine test_conversion_from_char_proc
  character(8), external :: char_proc

  ! CHECK: %[[proc:.*]] = fir.address_of(@_QPchar_proc) : {{.*}} -> !fir.boxchar<1>
  ! CHECK: %[[boxproc:.*]] = fir.emboxproc %[[proc]] : ({{.*}} -> !fir.boxchar<1>) -> !fir.boxproc<() -> ()>
  ! CHECK: %[[callee:.*]] = fir.address_of(@_QPpass_int_to_char_proc)
  ! CHECK: %[[convert:.*]] = fir.convert %[[callee]] : ((!fir.ref<i32>) -> ()) -> ((tuple<!fir.boxproc<() -> ()>, i64>) -> ())
  ! CHECK: fir.call %[[convert]]({{.*}})
  call pass_int_to_char_proc(char_proc)

  ! CHECK: %[[proc:.*]] = fir.address_of(@_QPchar_proc) : {{.*}} -> !fir.boxchar<1>
  ! CHECK: %[[boxproc:.*]] = fir.emboxproc %[[proc]] : ({{.*}} -> !fir.boxchar<1>) -> !fir.boxproc<() -> ()>
  ! CHECK: %[[callee:.*]] = fir.address_of(@_QPpass_logical_to_char_proc)
  ! CHECK: %[[convert:.*]] = fir.convert %[[callee]] : ((!fir.ref<!fir.logical<4>>) -> ()) -> ((tuple<!fir.boxproc<() -> ()>, i64>) -> ())
  ! CHECK: fir.call %[[convert]]({{.*}})
  call pass_logical_to_char_proc(char_proc)

  ! CHECK: %[[proc:.*]] = fir.address_of(@_QPchar_proc) : {{.*}} -> !fir.boxchar<1>
  ! CHECK: %[[boxproc:.*]] = fir.emboxproc %[[proc]] : ({{.*}} -> !fir.boxchar<1>) -> !fir.boxproc<() -> ()>
  ! CHECK: %[[callee:.*]] = fir.address_of(@_QPpass_real_to_char_proc)
  ! CHECK: %[[convert:.*]] = fir.convert %[[callee]] : ((!fir.ref<f64>) -> ()) -> ((tuple<!fir.boxproc<() -> ()>, i64>) -> ())
  ! CHECK: fir.call %[[convert]]({{.*}})
  call pass_real_to_char_proc(char_proc)

  ! CHECK: %[[proc:.*]] = fir.address_of(@_QPchar_proc) : {{.*}} -> !fir.boxchar<1>
  ! CHECK: %[[boxproc:.*]] = fir.emboxproc %[[proc]] : ({{.*}} -> !fir.boxchar<1>) -> !fir.boxproc<() -> ()>
  ! CHECK: %[[callee:.*]] = fir.address_of(@_QPpass_complex_to_char_proc)
  ! CHECK: %[[convert:.*]] = fir.convert %[[callee]] : ((!fir.ref<complex<f32>>) -> ()) -> ((tuple<!fir.boxproc<() -> ()>, i64>) -> ())
  ! CHECK: fir.call %[[convert]]({{.*}})
  call pass_complex_to_char_proc(char_proc)

  ! CHECK: %[[proc:.*]] = fir.address_of(@_QPchar_proc) : {{.*}} -> !fir.boxchar<1>
  ! CHECK: %[[boxproc:.*]] = fir.emboxproc %[[proc]] : ({{.*}} -> !fir.boxchar<1>) -> !fir.boxproc<() -> ()>
  ! CHECK: %[[callee:.*]] = fir.address_of(@_QPpass_char_to_char_proc)
  ! CHECK: %[[convert:.*]] = fir.convert %[[callee]] : ((!fir.boxchar<1>) -> ()) -> ((tuple<!fir.boxproc<() -> ()>, i64>) -> ())
  ! CHECK: fir.call %[[convert]]({{.*}})
  call pass_char_to_char_proc(char_proc)

  ! CHECK: %[[proc:.*]] = fir.address_of(@_QPchar_proc) : {{.*}} -> !fir.boxchar<1>
  ! CHECK: %[[boxproc:.*]] = fir.emboxproc %[[proc]] : ({{.*}} -> !fir.boxchar<1>) -> !fir.boxproc<() -> ()>
  ! CHECK: %[[callee:.*]] = fir.address_of(@_QPpass_dt_to_char_proc)
  ! CHECK: %[[convert:.*]] = fir.convert %[[callee]] : ((!fir.ref<!fir.type<{{.*}}>>) -> ()) -> ((tuple<!fir.boxproc<() -> ()>, i64>) -> ())
  ! CHECK: fir.call %[[convert]]({{.*}})
  call pass_dt_to_char_proc(char_proc)

  ! CHECK: %[[proc:.*]] = fir.address_of(@_QPchar_proc) : {{.*}} -> !fir.boxchar<1>
  ! CHECK: %[[boxproc:.*]] = fir.emboxproc %[[proc]] : ({{.*}} -> !fir.boxchar<1>) -> !fir.boxproc<() -> ()>
  ! CHECK: %[[callee:.*]] = fir.address_of(@_QPpass_array_to_char_proc)
  ! CHECK: %[[convert:.*]] = fir.convert %[[callee]] : ((!fir.ref<!fir.array<10x10xi32>>) -> ()) -> ((tuple<!fir.boxproc<() -> ()>, i64>) -> ())
  ! CHECK: fir.call %[[convert]]({{.*}})
  call pass_array_to_char_proc(char_proc)

  ! CHECK: %[[proc:.*]] = fir.address_of(@_QPchar_proc) : {{.*}} -> !fir.boxchar<1>
  ! CHECK: %[[boxproc:.*]] = fir.emboxproc %[[proc]] : ({{.*}} -> !fir.boxchar<1>) -> !fir.boxproc<() -> ()>
  ! CHECK: %[[callee:.*]] = fir.address_of(@_QPpass_proc_to_char_proc)
  ! CHECK: %[[convert:.*]] = fir.convert %[[callee]] : ((!fir.boxproc<() -> ()>) -> ()) -> ((tuple<!fir.boxproc<() -> ()>, i64>) -> ())
  ! CHECK: fir.call %[[convert]]({{.*}})
  call pass_proc_to_char_proc(char_proc)

  ! CHECK: %[[proc:.*]] = fir.address_of(@_QPchar_proc) : {{.*}} -> !fir.boxchar<1>
  ! CHECK: %[[len:.*]] = arith.constant 8 : i64
  ! CHECK: %[[boxProc:.*]] = fir.emboxproc %[[proc]] : ({{.*}} -> !fir.boxchar<1>) -> !fir.boxproc<() -> ()>
  ! CHECK: %[[tuple:.*]] = fir.undefined tuple<!fir.boxproc<() -> ()>, i64>
  ! CHECK: %[[tuple2:.*]] = fir.insert_value %[[tuple]], %[[boxProc]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxproc<() -> ()>) -> tuple<!fir.boxproc<() -> ()>, i64>
  ! CHECK: %[[tuple3:.*]] = fir.insert_value %[[tuple2]], %[[len]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, i64) -> tuple<!fir.boxproc<() -> ()>, i64>
  ! CHECK: fir.call @_QPpass_char_proc_to_char_proc(%[[tuple3]])
  call pass_char_proc_to_char_proc(char_proc)
end subroutine
