! Test questionable but existing abuses of implicit interfaces.
! Lowering must close the eyes and do as if it did not know
! about the function definition since semantic lets these
! programs through with a warning.
! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s

! Test reference to non-char procedure conversion.

subroutine takes_proc(proc)
  real(8), external :: proc
end subroutine

subroutine pass_int_to_proc(a)
  integer(4) :: a
  call takes_proc(a)
end subroutine
! CHECK-LABEL: func.func @_QPpass_int_to_proc(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<i32>
! CHECK: %[[procAddr:.*]] = fir.convert %[[arg0]] : (!fir.ref<i32>) -> (() -> ())
! CHECK: %[[boxProc:.*]] = fir.emboxproc %[[procAddr]] : (() -> ()) -> !fir.boxproc<() -> ()>
! CHECK: fir.call @_QPtakes_proc(%[[boxProc]]) {{.*}}: (!fir.boxproc<() -> ()>) -> ()

subroutine pass_logical_to_proc(a)
  logical(4) :: a
  call takes_proc(a)
end subroutine
! CHECK-LABEL: func.func @_QPpass_logical_to_proc(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.logical<4>>
! CHECK: %[[procAddr:.*]] = fir.convert %[[arg0]] : (!fir.ref<!fir.logical<4>>) -> (() -> ())
! CHECK: %[[boxProc:.*]] = fir.emboxproc %[[procAddr]] : (() -> ()) -> !fir.boxproc<() -> ()>
! CHECK: fir.call @_QPtakes_proc(%[[boxProc]]) {{.*}}: (!fir.boxproc<() -> ()>) -> ()

subroutine pass_real_to_proc(a)
  real(8) :: a
  call takes_proc(a)
end subroutine
! CHECK-LABEL: func.func @_QPpass_real_to_proc(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<f64>
! CHECK: %[[procAddr:.*]] = fir.convert %[[arg0]] : (!fir.ref<f64>) -> (() -> ())
! CHECK: %[[boxProc:.*]] = fir.emboxproc %[[procAddr]] : (() -> ()) -> !fir.boxproc<() -> ()>
! CHECK: fir.call @_QPtakes_proc(%[[boxProc]]) {{.*}}: (!fir.boxproc<() -> ()>) -> ()

subroutine pass_complex_to_proc(a)
  complex(4) :: a
  call takes_proc(a)
end subroutine
! CHECK-LABEL: func.func @_QPpass_complex_to_proc(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.complex<4>>
! CHECK: %[[procAddr:.*]] = fir.convert %[[arg0]] : (!fir.ref<!fir.complex<4>>) -> (() -> ())
! CHECK: %[[boxProc:.*]] = fir.emboxproc %[[procAddr]] : (() -> ()) -> !fir.boxproc<() -> ()>
! CHECK: fir.call @_QPtakes_proc(%[[boxProc]]) {{.*}}: (!fir.boxproc<() -> ()>) -> ()

subroutine pass_char_to_proc(a)
  character(8) :: a
  call takes_proc(a)
end subroutine
! CHECK-LABEL: func.func @_QPpass_char_to_proc(
! CHECK-SAME: %[[arg0:.*]]: !fir.boxchar<1>
! CHECK: %[[charAndLen:.*]]:2 = fir.unboxchar %[[arg0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK: %[[charRef:.*]] = fir.convert %[[charAndLen]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,8>>
! CHECK: %[[procAddr:.*]] = fir.convert %[[charRef]] : (!fir.ref<!fir.char<1,8>>) -> (() -> ())
! CHECK: %[[boxProc:.*]] = fir.emboxproc %[[procAddr]] : (() -> ()) -> !fir.boxproc<() -> ()>
! CHECK: fir.call @_QPtakes_proc(%[[boxProc]]) {{.*}}: (!fir.boxproc<() -> ()>) -> ()

subroutine pass_dt_to_proc(a)
  type :: dt
    integer(4) :: i, j
  end type
  type(dt) :: a

  call takes_proc(a)
end subroutine
! CHECK-LABEL: func.func @_QPpass_dt_to_proc(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.type<_QFpass_dt_to_procTdt{i:i32,j:i32}>>
! CHECK: %[[procAddr:.*]] = fir.convert %[[arg0]] : (!fir.ref<!fir.type<_QFpass_dt_to_procTdt{i:i32,j:i32}>>) -> (() -> ())
! CHECK: %[[boxProc:.*]] = fir.emboxproc %[[procAddr]] : (() -> ()) -> !fir.boxproc<() -> ()>
! CHECK: fir.call @_QPtakes_proc(%[[boxProc]]) {{.*}}: (!fir.boxproc<() -> ()>) -> ()

subroutine pass_array_to_proc(a)
  integer(4) :: a(10,10)
  call takes_proc(a)
end subroutine
! CHECK-LABEL: func.func @_QPpass_array_to_proc(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.array<10x10xi32>>
! CHECK: %[[procAddr:.*]] = fir.convert %[[arg0]] : (!fir.ref<!fir.array<10x10xi32>>) -> (() -> ())
! CHECK: %[[boxProc:.*]] = fir.emboxproc %[[procAddr]] : (() -> ()) -> !fir.boxproc<() -> ()>
! CHECK: fir.call @_QPtakes_proc(%[[boxProc]]) {{.*}}: (!fir.boxproc<() -> ()>) -> ()

! Test procedure conversion.

subroutine pass_char_proc_to_proc(a)
  character(8), external :: a
  call takes_proc(a)
end subroutine
! CHECK-LABEL: func.func @_QPpass_char_proc_to_proc(
! CHECK-SAME: %[[arg0:.*]]: tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: %[[extract:.*]] = fir.extract_value %[[arg0]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> !fir.boxproc<() -> ()>
! CHECK: %[[procAddr:.*]] = fir.box_addr %[[extract]] : (!fir.boxproc<() -> ()>) -> (() -> ())
! CHECK: %[[boxProc:.*]] = fir.emboxproc %[[procAddr]] : (() -> ()) -> !fir.boxproc<() -> ()>
! CHECK: fir.call @_QPtakes_proc(%[[boxProc]]) {{.*}}: (!fir.boxproc<() -> ()>) -> ()

subroutine pass_proc_to_proc(a)
  external :: a
  call takes_proc(a)
end subroutine
! CHECK-LABEL: func.func @_QPpass_proc_to_proc(
! CHECK-SAME: %[[arg0:.*]]: !fir.boxproc<() -> ()>
! CHECK: fir.call @_QPtakes_proc(%[[arg0]]) {{.*}}: (!fir.boxproc<() -> ()>) -> ()

! Test conversion from procedure to other data types.

! CHECK-LABEL: func.func @_QPtest_conversion_from_proc(
subroutine test_conversion_from_proc
  external :: proc

  ! CHECK: %[[proc:.*]] = fir.address_of(@_QPproc) : () -> ()
  ! CHECK: %[[convert:.*]] = fir.convert %[[proc]] : (() -> ()) -> !fir.ref<i32>
  ! CHECK: fir.call @_QPpass_int_to_proc(%[[convert]])
  call pass_int_to_proc(proc)

  ! CHECK: %[[proc:.*]] = fir.address_of(@_QPproc) : () -> ()
  ! CHECK: %[[convert:.*]] = fir.convert %[[proc]] : (() -> ()) -> !fir.ref<!fir.logical<4>>
  ! CHECK: fir.call @_QPpass_logical_to_proc(%[[convert]])
  call pass_logical_to_proc(proc)

  ! CHECK: %[[proc:.*]] = fir.address_of(@_QPproc) : () -> ()
  ! CHECK: %[[convert:.*]] = fir.convert %[[proc]] : (() -> ()) -> !fir.ref<f64>
  ! CHECK: fir.call @_QPpass_real_to_proc(%[[convert]])
  call pass_real_to_proc(proc)

  ! CHECK: %[[proc:.*]] = fir.address_of(@_QPproc) : () -> ()
  ! CHECK: %[[convert:.*]] = fir.convert %[[proc]] : (() -> ()) -> !fir.ref<!fir.complex<4>>
  ! CHECK: fir.call @_QPpass_complex_to_proc(%[[convert]])
  call pass_complex_to_proc(proc)

  ! CHECK: %[[proc:.*]] = fir.address_of(@_QPproc) : () -> ()
  ! CHECK: %[[convert:.*]] = fir.convert %[[proc]] : (() -> ()) -> !fir.ref<!fir.char<1,?>>
  ! CHECK: %[[len:.*]] = fir.undefined index
  ! CHECK: %[[box:.*]] = fir.emboxchar %[[convert]], %[[len]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  ! CHECK: fir.call @_QPpass_char_to_proc(%[[box]])
  call pass_char_to_proc(proc)

  ! CHECK: %[[proc:.*]] = fir.address_of(@_QPproc) : () -> ()
  ! CHECK: %[[convert:.*]] = fir.convert %[[proc]] : (() -> ()) -> !fir.ref<!fir.type<_QFpass_dt_to_procTdt{i:i32,j:i32}>>
  ! CHECK: fir.call @_QPpass_dt_to_proc(%[[convert]])
  call pass_dt_to_proc(proc)

  ! CHECK: %[[proc:.*]] = fir.address_of(@_QPproc) : () -> ()
  ! CHECK: %[[convert:.*]] = fir.convert %[[proc]] : (() -> ()) -> !fir.ref<!fir.array<10x10xi32>>
  ! CHECK: fir.call @_QPpass_array_to_proc(%[[convert]])
  call pass_array_to_proc(proc)

  ! CHECK: %[[proc:.*]] = fir.address_of(@_QPproc) : () -> ()
  ! CHECK: %[[boxProc:.*]] = fir.emboxproc %[[proc]] : (() -> ()) -> !fir.boxproc<() -> ()>
  ! CHECK: %[[len:.*]] = fir.undefined i64
  ! CHECK: %[[tuple:.*]] = fir.undefined tuple<!fir.boxproc<() -> ()>, i64>
  ! CHECK: %[[tuple2:.*]] = fir.insert_value %[[tuple]], %[[boxProc]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxproc<() -> ()>) -> tuple<!fir.boxproc<() -> ()>, i64>
  ! CHECK: %[[tuple3:.*]] = fir.insert_value %[[tuple2]], %[[len]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, i64) -> tuple<!fir.boxproc<() -> ()>, i64>
  ! CHECK: fir.call @_QPpass_char_proc_to_proc(%[[tuple3]]) {{.*}}: (tuple<!fir.boxproc<() -> ()>, i64>) -> ()
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
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<i32>
! CHECK: %[[procAddr:.*]] = fir.convert %[[arg0]] : (!fir.ref<i32>) -> (() -> ())
! CHECK: %[[boxProc:.*]] = fir.emboxproc %[[procAddr]] : (() -> ()) -> !fir.boxproc<() -> ()>
! CHECK: %[[charLen:.*]] = fir.undefined i64
! CHECK: %[[tuple:.*]] = fir.undefined tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: %[[tuple2:.*]] = fir.insert_value %[[tuple]], %[[boxProc]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxproc<() -> ()>) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: %[[tuple3:.*]] = fir.insert_value %[[tuple2]], %[[charLen]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, i64) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: fir.call @_QPtakes_char_proc(%[[tuple3]]) {{.*}}: (tuple<!fir.boxproc<() -> ()>, i64>) -> ()

subroutine pass_logical_to_char_proc(a)
  logical(4) :: a
  call takes_char_proc(a)
end subroutine
! CHECK-LABEL: func.func @_QPpass_logical_to_char_proc(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.logical<4>>
! CHECK: %[[procAddr:.*]] = fir.convert %[[arg0]] : (!fir.ref<!fir.logical<4>>) -> (() -> ())
! CHECK: %[[boxProc:.*]] = fir.emboxproc %[[procAddr]] : (() -> ()) -> !fir.boxproc<() -> ()>
! CHECK: %[[charLen:.*]] = fir.undefined i64
! CHECK: %[[tuple:.*]] = fir.undefined tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: %[[tuple2:.*]] = fir.insert_value %[[tuple]], %[[boxProc]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxproc<() -> ()>) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: %[[tuple3:.*]] = fir.insert_value %[[tuple2]], %[[charLen]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, i64) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: fir.call @_QPtakes_char_proc(%[[tuple3]]) {{.*}}: (tuple<!fir.boxproc<() -> ()>, i64>) -> ()

subroutine pass_real_to_char_proc(a)
  real(8) :: a
  call takes_char_proc(a)
end subroutine
! CHECK-LABEL: func.func @_QPpass_real_to_char_proc(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<f64>
! CHECK: %[[procAddr:.*]] = fir.convert %[[arg0]] : (!fir.ref<f64>) -> (() -> ())
! CHECK: %[[boxProc:.*]] = fir.emboxproc %[[procAddr]] : (() -> ()) -> !fir.boxproc<() -> ()>
! CHECK: %[[charLen:.*]] = fir.undefined i64
! CHECK: %[[tuple:.*]] = fir.undefined tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: %[[tuple2:.*]] = fir.insert_value %[[tuple]], %[[boxProc]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxproc<() -> ()>) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: %[[tuple3:.*]] = fir.insert_value %[[tuple2]], %[[charLen]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, i64) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: fir.call @_QPtakes_char_proc(%[[tuple3]]) {{.*}}: (tuple<!fir.boxproc<() -> ()>, i64>) -> ()

subroutine pass_complex_to_char_proc(a)
  complex(4) :: a
  call takes_char_proc(a)
end subroutine
! CHECK-LABEL: func.func @_QPpass_complex_to_char_proc(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.complex<4>>
! CHECK: %[[procAddr:.*]] = fir.convert %[[arg0]] : (!fir.ref<!fir.complex<4>>) -> (() -> ())
! CHECK: %[[boxProc:.*]] = fir.emboxproc %[[procAddr]] : (() -> ()) -> !fir.boxproc<() -> ()>
! CHECK: %[[charLen:.*]] = fir.undefined i64
! CHECK: %[[tuple:.*]] = fir.undefined tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: %[[tuple2:.*]] = fir.insert_value %[[tuple]], %[[boxProc]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxproc<() -> ()>) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: %[[tuple3:.*]] = fir.insert_value %[[tuple2]], %[[charLen]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, i64) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: fir.call @_QPtakes_char_proc(%[[tuple3]]) {{.*}}: (tuple<!fir.boxproc<() -> ()>, i64>) -> ()

subroutine pass_char_to_char_proc(a)
  character(8) :: a
  call takes_char_proc(a)
end subroutine
! CHECK-LABEL: func.func @_QPpass_char_to_char_proc(
! CHECK-SAME: %[[arg0:.*]]: !fir.boxchar<1>
! CHECK: %[[charRefAndLen:.*]]:2 = fir.unboxchar %[[arg0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK: %[[charRef:.*]] = fir.convert %[[charRefAndLen]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,8>>
! CHECK: %[[charLen:.*]] = arith.constant 8 : index
! CHECK: %[[procAddr:.*]] = fir.convert %[[charRef]] : (!fir.ref<!fir.char<1,8>>) -> (() -> ())
! CHECK: %[[boxProc:.*]] = fir.emboxproc %[[procAddr]] : (() -> ()) -> !fir.boxproc<() -> ()>
! CHECK: %[[charLen2:.*]] = fir.convert %[[charLen]] : (index) -> i64
! CHECK: %[[tuple:.*]] = fir.undefined tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: %[[tuple2:.*]] = fir.insert_value %[[tuple]], %[[boxProc]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxproc<() -> ()>) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: %[[tuple3:.*]] = fir.insert_value %[[tuple2]], %[[charLen2]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, i64) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: fir.call @_QPtakes_char_proc(%[[tuple3]]) {{.*}}: (tuple<!fir.boxproc<() -> ()>, i64>) -> ()

subroutine pass_dt_to_char_proc(a)
  type :: dt
    integer(4) :: i, j
  end type
  type(dt) :: a

  call takes_char_proc(a)
end subroutine
! CHECK-LABEL: func.func @_QPpass_dt_to_char_proc(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.type<_QFpass_dt_to_char_procTdt{i:i32,j:i32}>>
! CHECK: %[[procAddr:.*]] = fir.convert %[[arg0]] : (!fir.ref<!fir.type<_QFpass_dt_to_char_procTdt{i:i32,j:i32}>>) -> (() -> ())
! CHECK: %[[boxProc:.*]] = fir.emboxproc %[[procAddr]] : (() -> ()) -> !fir.boxproc<() -> ()>
! CHECK: %[[charLen:.*]] = fir.undefined i64
! CHECK: %[[tuple:.*]] = fir.undefined tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: %[[tuple2:.*]] = fir.insert_value %[[tuple]], %[[boxProc]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxproc<() -> ()>) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: %[[tuple3:.*]] = fir.insert_value %[[tuple2]], %[[charLen]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, i64) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: fir.call @_QPtakes_char_proc(%[[tuple3]]) {{.*}}: (tuple<!fir.boxproc<() -> ()>, i64>) -> ()

subroutine pass_array_to_char_proc(a)
  integer(4) :: a(10,10)
  call takes_char_proc(a)
end subroutine
! CHECK-LABEL: func.func @_QPpass_array_to_char_proc(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.array<10x10xi32>>
! CHECK: %[[procAddr:.*]] = fir.convert %[[arg0]] : (!fir.ref<!fir.array<10x10xi32>>) -> (() -> ())
! CHECK: %[[boxProc:.*]] = fir.emboxproc %[[procAddr]] : (() -> ()) -> !fir.boxproc<() -> ()>
! CHECK: %[[charLen:.*]] = fir.undefined i64
! CHECK: %[[tuple:.*]] = fir.undefined tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: %[[tuple2:.*]] = fir.insert_value %[[tuple]], %[[boxProc]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxproc<() -> ()>) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: %[[tuple3:.*]] = fir.insert_value %[[tuple2]], %[[charLen]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, i64) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: fir.call @_QPtakes_char_proc(%[[tuple3]]) {{.*}}: (tuple<!fir.boxproc<() -> ()>, i64>) -> ()

! Test procedure conversion.

subroutine pass_proc_to_char_proc(a)
  external :: a
  call takes_char_proc(a)
end subroutine
! CHECK-LABEL: func.func @_QPpass_proc_to_char_proc(
! CHECK-SAME: %[[arg0:.*]]: !fir.boxproc<() -> ()>
! CHECK: %[[procAddr:.*]] = fir.box_addr %[[arg0]] : (!fir.boxproc<() -> ()>) -> (() -> ())
! CHECK: %[[boxProc:.*]] = fir.emboxproc %[[procAddr]] : (() -> ()) -> !fir.boxproc<() -> ()>
! CHECK: %[[charLen:.*]] = fir.undefined i64
! CHECK: %[[tuple:.*]] = fir.undefined tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: %[[tuple2:.*]] = fir.insert_value %[[tuple]], %[[boxProc]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxproc<() -> ()>) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: %[[tuple3:.*]] = fir.insert_value %[[tuple2]], %[[charLen]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, i64) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: fir.call @_QPtakes_char_proc(%[[tuple3]]) {{.*}}: (tuple<!fir.boxproc<() -> ()>, i64>) -> ()

subroutine pass_char_proc_to_char_proc(a)
  character(8), external :: a
  call takes_char_proc(a)
end subroutine
! CHECK-LABEL: func.func @_QPpass_char_proc_to_char_proc(
! CHECK-SAME: %[[arg0:.*]]: tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: %[[boxProc:.*]] = fir.extract_value %arg0, [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> !fir.boxproc<() -> ()>
! CHECK: %[[procAddr:.*]] = fir.box_addr %[[boxProc]] : (!fir.boxproc<() -> ()>) -> (() -> ())
! CHECK: %[[charLen:.*]] = arith.constant 8 : i64
! CHECK: %[[boxProc2:.*]] = fir.emboxproc %[[procAddr]] : (() -> ()) -> !fir.boxproc<() -> ()>
! CHECK: %[[tuple:.*]] = fir.undefined tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: %[[tuple2:.*]] = fir.insert_value %[[tuple]], %[[boxProc2]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxproc<() -> ()>) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: %[[tuple3:.*]] = fir.insert_value %[[tuple2]], %[[charLen]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, i64) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: fir.call @_QPtakes_char_proc(%[[tuple3]]) {{.*}}: (tuple<!fir.boxproc<() -> ()>, i64>) -> ()

! Test conversion from character procedure to other data types.

! CHECK-LABEL: func.func @_QPtest_conversion_from_char_proc(
subroutine test_conversion_from_char_proc
  character(8), external :: char_proc

  ! CHECK: %[[proc:.*]] = fir.address_of(@_QPchar_proc) : (!fir.ref<!fir.char<1,8>>, index) -> !fir.boxchar<1>
  ! CHECK: %[[convert:.*]] = fir.convert %[[proc]] : ((!fir.ref<!fir.char<1,8>>, index) -> !fir.boxchar<1>) -> !fir.ref<i32>
  ! CHECK: fir.call @_QPpass_int_to_char_proc(%[[convert]])
  call pass_int_to_char_proc(char_proc)

  ! CHECK: %[[proc:.*]] = fir.address_of(@_QPchar_proc) : (!fir.ref<!fir.char<1,8>>, index) -> !fir.boxchar<1>
  ! CHECK: %[[convert:.*]] = fir.convert %[[proc]] : ((!fir.ref<!fir.char<1,8>>, index) -> !fir.boxchar<1>) -> !fir.ref<!fir.logical<4>>
  ! CHECK: fir.call @_QPpass_logical_to_char_proc(%[[convert]])
  call pass_logical_to_char_proc(char_proc)

  ! CHECK: %[[proc:.*]] = fir.address_of(@_QPchar_proc) : (!fir.ref<!fir.char<1,8>>, index) -> !fir.boxchar<1>
  ! CHECK: %[[convert:.*]] = fir.convert %[[proc]] : ((!fir.ref<!fir.char<1,8>>, index) -> !fir.boxchar<1>) -> !fir.ref<f64>
  ! CHECK: fir.call @_QPpass_real_to_char_proc(%[[convert]])
  call pass_real_to_char_proc(char_proc)

  ! CHECK: %[[proc:.*]] = fir.address_of(@_QPchar_proc) : (!fir.ref<!fir.char<1,8>>, index) -> !fir.boxchar<1>
  ! CHECK: %[[convert:.*]] = fir.convert %[[proc]] : ((!fir.ref<!fir.char<1,8>>, index) -> !fir.boxchar<1>) -> !fir.ref<!fir.complex<4>>
  ! CHECK: fir.call @_QPpass_complex_to_char_proc(%[[convert]])
  call pass_complex_to_char_proc(char_proc)

  ! CHECK: %[[proc:.*]] = fir.address_of(@_QPchar_proc) : (!fir.ref<!fir.char<1,8>>, index) -> !fir.boxchar<1>
  ! CHECK: %[[convert:.*]] = fir.convert %[[proc]] : ((!fir.ref<!fir.char<1,8>>, index) -> !fir.boxchar<1>) -> !fir.ref<!fir.char<1,?>>
  ! CHECK: %[[len:.*]] = fir.undefined index
  ! CHECK: %[[box:.*]] = fir.emboxchar %[[convert]], %[[len]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  ! CHECK: fir.call @_QPpass_char_to_char_proc(%[[box]])
  call pass_char_to_char_proc(char_proc)

  ! CHECK: %[[proc:.*]] = fir.address_of(@_QPchar_proc) : (!fir.ref<!fir.char<1,8>>, index) -> !fir.boxchar<1>
  ! CHECK: %[[convert:.*]] = fir.convert %[[proc]] : ((!fir.ref<!fir.char<1,8>>, index) -> !fir.boxchar<1>) -> !fir.ref<!fir.type<_QFpass_dt_to_char_procTdt{i:i32,j:i32}>>
  ! CHECK: fir.call @_QPpass_dt_to_char_proc(%[[convert]])
  call pass_dt_to_char_proc(char_proc)

  ! CHECK: %[[proc:.*]] = fir.address_of(@_QPchar_proc) : (!fir.ref<!fir.char<1,8>>, index) -> !fir.boxchar<1>
  ! CHECK: %[[convert:.*]] = fir.convert %[[proc]] : ((!fir.ref<!fir.char<1,8>>, index) -> !fir.boxchar<1>) -> !fir.ref<!fir.array<10x10xi32>>
  ! CHECK: fir.call @_QPpass_array_to_char_proc(%[[convert]])
  call pass_array_to_char_proc(char_proc)

  ! CHECK: %[[proc:.*]] = fir.address_of(@_QPchar_proc) : (!fir.ref<!fir.char<1,8>>, index) -> !fir.boxchar<1>
  ! CHECK: %[[boxProc:.*]] = fir.emboxproc %[[proc]] : ((!fir.ref<!fir.char<1,8>>, index) -> !fir.boxchar<1>) -> !fir.boxproc<() -> ()>
  ! CHECK: fir.call @_QPpass_proc_to_char_proc(%[[boxProc]])
  call pass_proc_to_char_proc(char_proc)

  ! CHECK: %[[proc:.*]] = fir.address_of(@_QPchar_proc) : (!fir.ref<!fir.char<1,8>>, index) -> !fir.boxchar<1>
  ! CHECK: %[[len:.*]] = arith.constant 8 : i64
  ! CHECK: %[[boxProc:.*]] = fir.emboxproc %[[proc]] : ((!fir.ref<!fir.char<1,8>>, index) -> !fir.boxchar<1>) -> !fir.boxproc<() -> ()>
  ! CHECK: %[[tuple:.*]] = fir.undefined tuple<!fir.boxproc<() -> ()>, i64>
  ! CHECK: %[[tuple2:.*]] = fir.insert_value %[[tuple]], %[[boxProc]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxproc<() -> ()>) -> tuple<!fir.boxproc<() -> ()>, i64>
  ! CHECK: %[[tuple3:.*]] = fir.insert_value %[[tuple2]], %[[len]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, i64) -> tuple<!fir.boxproc<() -> ()>, i64>
  ! CHECK: fir.call @_QPpass_char_proc_to_char_proc(%[[tuple3]])
  call pass_char_proc_to_char_proc(char_proc)
end subroutine
