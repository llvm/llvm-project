! This test checks lowering of `COPYIN` clause.
! RUN: bbc -fopenmp -emit-fir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPcopyin_scalar_array() {
! CHECK:         %[[VAL_0:.*]] = fir.address_of(@_QFcopyin_scalar_arrayEx1) : !fir.ref<i32>
! CHECK:         %[[VAL_1:.*]] = omp.threadprivate %[[VAL_0]] : !fir.ref<i32> -> !fir.ref<i32>
! CHECK:         %[[VAL_2:.*]] = fir.address_of(@_QFcopyin_scalar_arrayEx2) : !fir.ref<!fir.array<10xi64>>
! CHECK:         %[[VAL_3:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_4:.*]] = omp.threadprivate %[[VAL_2]] : !fir.ref<!fir.array<10xi64>> -> !fir.ref<!fir.array<10xi64>>
! CHECK:         omp.parallel   {
! CHECK:           %[[VAL_5:.*]] = omp.threadprivate %[[VAL_0]] : !fir.ref<i32> -> !fir.ref<i32>
! CHECK:           %[[VAL_6:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:           fir.store %[[VAL_6]] to %[[VAL_5]] : !fir.ref<i32>
! CHECK:           %[[VAL_7:.*]] = omp.threadprivate %[[VAL_2]] : !fir.ref<!fir.array<10xi64>> -> !fir.ref<!fir.array<10xi64>>
! CHECK:           %[[VAL_8:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_9:.*]] = fir.array_load %[[VAL_7]](%[[VAL_8]]) : (!fir.ref<!fir.array<10xi64>>, !fir.shape<1>) -> !fir.array<10xi64>
! CHECK:           %[[VAL_10:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_11:.*]] = fir.array_load %[[VAL_4]](%[[VAL_10]]) : (!fir.ref<!fir.array<10xi64>>, !fir.shape<1>) -> !fir.array<10xi64>
! CHECK:           %[[VAL_12:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_13:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_14:.*]] = arith.subi %[[VAL_3]], %[[VAL_12]] : index
! CHECK:           %[[VAL_15:.*]] = fir.do_loop %[[VAL_16:.*]] = %[[VAL_13]] to %[[VAL_14]] step %[[VAL_12]] unordered iter_args(%[[VAL_17:.*]] = %[[VAL_9]]) -> (!fir.array<10xi64>) {
! CHECK:             %[[VAL_18:.*]] = fir.array_fetch %[[VAL_11]], %[[VAL_16]] : (!fir.array<10xi64>, index) -> i64
! CHECK:             %[[VAL_19:.*]] = fir.array_update %[[VAL_17]], %[[VAL_18]], %[[VAL_16]] : (!fir.array<10xi64>, i64, index) -> !fir.array<10xi64>
! CHECK:             fir.result %[[VAL_19]] : !fir.array<10xi64>
! CHECK:           }
! CHECK:           fir.array_merge_store %[[VAL_9]], %[[VAL_20:.*]] to %[[VAL_7]] : !fir.array<10xi64>, !fir.array<10xi64>, !fir.ref<!fir.array<10xi64>>
! CHECK:           omp.barrier
! CHECK:           fir.call @_QPsub1(%[[VAL_5]], %[[VAL_7]]) {{.*}}: (!fir.ref<i32>, !fir.ref<!fir.array<10xi64>>) -> ()
! CHECK:           omp.terminator
! CHECK:         }
! CHECK:         return
! CHECK:       }

subroutine copyin_scalar_array()
  integer(kind=4), save :: x1
  integer(kind=8), save :: x2(10)
  !$omp threadprivate(x1, x2)

  !$omp parallel copyin(x1) copyin(x2)
    call sub1(x1, x2)
  !$omp end parallel

end

! CHECK-LABEL: func.func @_QPcopyin_char_chararray() {
! CHECK:         %[[VAL_0:.*]] = fir.address_of(@_QFcopyin_char_chararrayEx3) : !fir.ref<!fir.char<1,5>>
! CHECK:         %[[VAL_1:.*]] = arith.constant 5 : index
! CHECK:         %[[VAL_2:.*]] = omp.threadprivate %[[VAL_0]] : !fir.ref<!fir.char<1,5>> -> !fir.ref<!fir.char<1,5>>
! CHECK:         %[[VAL_3:.*]] = fir.address_of(@_QFcopyin_char_chararrayEx4) : !fir.ref<!fir.array<10x!fir.char<1,5>>>
! CHECK:         %[[VAL_4:.*]] = arith.constant 5 : index
! CHECK:         %[[VAL_5:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_6:.*]] = omp.threadprivate %[[VAL_3]] : !fir.ref<!fir.array<10x!fir.char<1,5>>> -> !fir.ref<!fir.array<10x!fir.char<1,5>>>
! CHECK:         omp.parallel   {
! CHECK:           %[[VAL_7:.*]] = omp.threadprivate %[[VAL_0]] : !fir.ref<!fir.char<1,5>> -> !fir.ref<!fir.char<1,5>>
! CHECK:           %[[VAL_8:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_1]] : (index) -> i64
! CHECK:           %[[VAL_10:.*]] = arith.muli %[[VAL_8]], %[[VAL_9]] : i64
! CHECK:           %[[VAL_11:.*]] = arith.constant false
! CHECK:           %[[VAL_12:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<!fir.char<1,5>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.char<1,5>>) -> !fir.ref<i8>
! CHECK:           fir.call @llvm.memmove.p0.p0.i64(%[[VAL_12]], %[[VAL_13]], %[[VAL_10]], %[[VAL_11]]) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
! CHECK:           %[[VAL_14:.*]] = omp.threadprivate %[[VAL_3]] : !fir.ref<!fir.array<10x!fir.char<1,5>>> -> !fir.ref<!fir.array<10x!fir.char<1,5>>>
! CHECK:           %[[VAL_15:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_16:.*]] = fir.array_load %[[VAL_14]](%[[VAL_15]]) : (!fir.ref<!fir.array<10x!fir.char<1,5>>>, !fir.shape<1>) -> !fir.array<10x!fir.char<1,5>>
! CHECK:           %[[VAL_17:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_18:.*]] = fir.array_load %[[VAL_6]](%[[VAL_17]]) : (!fir.ref<!fir.array<10x!fir.char<1,5>>>, !fir.shape<1>) -> !fir.array<10x!fir.char<1,5>>
! CHECK:           %[[VAL_19:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_20:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_21:.*]] = arith.subi %[[VAL_5]], %[[VAL_19]] : index
! CHECK:           %[[VAL_22:.*]] = fir.do_loop %[[VAL_23:.*]] = %[[VAL_20]] to %[[VAL_21]] step %[[VAL_19]] unordered iter_args(%[[VAL_24:.*]] = %[[VAL_16]]) -> (!fir.array<10x!fir.char<1,5>>) {
! CHECK:             %[[VAL_25:.*]] = fir.array_access %[[VAL_18]], %[[VAL_23]] : (!fir.array<10x!fir.char<1,5>>, index) -> !fir.ref<!fir.char<1,5>>
! CHECK:             %[[VAL_26:.*]] = fir.array_access %[[VAL_24]], %[[VAL_23]] : (!fir.array<10x!fir.char<1,5>>, index) -> !fir.ref<!fir.char<1,5>>
! CHECK:             %[[VAL_27:.*]] = arith.constant 5 : index
! CHECK:             %[[VAL_28:.*]] = arith.constant 1 : i64
! CHECK:             %[[VAL_29:.*]] = fir.convert %[[VAL_27]] : (index) -> i64
! CHECK:             %[[VAL_30:.*]] = arith.muli %[[VAL_28]], %[[VAL_29]] : i64
! CHECK:             %[[VAL_31:.*]] = arith.constant false
! CHECK:             %[[VAL_32:.*]] = fir.convert %[[VAL_26]] : (!fir.ref<!fir.char<1,5>>) -> !fir.ref<i8>
! CHECK:             %[[VAL_33:.*]] = fir.convert %[[VAL_25]] : (!fir.ref<!fir.char<1,5>>) -> !fir.ref<i8>
! CHECK:             fir.call @llvm.memmove.p0.p0.i64(%[[VAL_32]], %[[VAL_33]], %[[VAL_30]], %[[VAL_31]]) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
! CHECK:             %[[VAL_34:.*]] = fir.array_amend %[[VAL_24]], %[[VAL_26]] : (!fir.array<10x!fir.char<1,5>>, !fir.ref<!fir.char<1,5>>) -> !fir.array<10x!fir.char<1,5>>
! CHECK:             fir.result %[[VAL_34]] : !fir.array<10x!fir.char<1,5>>
! CHECK:           }
! CHECK:           fir.array_merge_store %[[VAL_16]], %[[VAL_35:.*]] to %[[VAL_14]] : !fir.array<10x!fir.char<1,5>>, !fir.array<10x!fir.char<1,5>>, !fir.ref<!fir.array<10x!fir.char<1,5>>>
! CHECK:           omp.barrier
! CHECK:           %[[VAL_36:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<!fir.char<1,5>>) -> !fir.ref<!fir.char<1,?>>
! CHECK:           %[[VAL_37:.*]] = fir.emboxchar %[[VAL_36]], %[[VAL_1]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:           %[[VAL_38:.*]] = fir.convert %[[VAL_14]] : (!fir.ref<!fir.array<10x!fir.char<1,5>>>) -> !fir.ref<!fir.char<1,?>>
! CHECK:           %[[VAL_39:.*]] = fir.emboxchar %[[VAL_38]], %[[VAL_4]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:           fir.call @_QPsub2(%[[VAL_37]], %[[VAL_39]]) {{.*}}: (!fir.boxchar<1>, !fir.boxchar<1>) -> ()
! CHECK:           omp.terminator
! CHECK:         }
! CHECK:         return
! CHECK:       }

subroutine copyin_char_chararray()
  character(5), save :: x3, x4(10)
  !$omp threadprivate(x3, x4)

  !$omp parallel copyin(x3) copyin(x4)
    call sub2(x3, x4)
  !$omp end parallel

end

! CHECK-LABEL: func.func @_QPcopyin_derived_type() {
! CHECK:         %[[VAL_0:.*]] = fir.address_of(@_QFcopyin_derived_typeEx5) : !fir.ref<!fir.type<_QFcopyin_derived_typeTmy_type{t_i:i32,t_arr:!fir.array<5xi32>}>>
! CHECK:         %[[VAL_1:.*]] = omp.threadprivate %[[VAL_0]] : !fir.ref<!fir.type<_QFcopyin_derived_typeTmy_type{t_i:i32,t_arr:!fir.array<5xi32>}>> -> !fir.ref<!fir.type<_QFcopyin_derived_typeTmy_type{t_i:i32,t_arr:!fir.array<5xi32>}>>
! CHECK:         omp.parallel   {
! CHECK:           %[[VAL_2:.*]] = omp.threadprivate %[[VAL_0]] : !fir.ref<!fir.type<_QFcopyin_derived_typeTmy_type{t_i:i32,t_arr:!fir.array<5xi32>}>> -> !fir.ref<!fir.type<_QFcopyin_derived_typeTmy_type{t_i:i32,t_arr:!fir.array<5xi32>}>>
! CHECK:           %[[VAL_3:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.type<_QFcopyin_derived_typeTmy_type{t_i:i32,t_arr:!fir.array<5xi32>}>>
! CHECK:           fir.store %[[VAL_3]] to %[[VAL_2]] : !fir.ref<!fir.type<_QFcopyin_derived_typeTmy_type{t_i:i32,t_arr:!fir.array<5xi32>}>>
! CHECK:           omp.barrier
! CHECK:           fir.call @_QPsub3(%[[VAL_2]]) {{.*}}: (!fir.ref<!fir.type<_QFcopyin_derived_typeTmy_type{t_i:i32,t_arr:!fir.array<5xi32>}>>) -> ()
! CHECK:           omp.terminator
! CHECK:         }
! CHECK:         return
! CHECK:       }

subroutine copyin_derived_type()
  type my_type
    integer :: t_i
    integer :: t_arr(5)
  end type my_type
  type(my_type), save :: x5
  !$omp threadprivate(x5)

  !$omp parallel copyin(x5)
    call sub3(x5)
  !$omp end parallel

end

! CHECK-LABEL: func.func @_QPcombined_parallel_worksharing_loop() {
! CHECK:         %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFcombined_parallel_worksharing_loopEi"}
! CHECK:         %[[VAL_1:.*]] = fir.address_of(@_QFcombined_parallel_worksharing_loopEx6) : !fir.ref<i32>
! CHECK:         %[[VAL_2:.*]] = omp.threadprivate %[[VAL_1]] : !fir.ref<i32> -> !fir.ref<i32>
! CHECK:         omp.parallel   {
! CHECK:           %[[VAL_3:.*]] = fir.alloca i32 {adapt.valuebyref, pinned}
! CHECK:           %[[VAL_4:.*]] = omp.threadprivate %[[VAL_1]] : !fir.ref<i32> -> !fir.ref<i32>
! CHECK:           %[[VAL_5:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:           fir.store %[[VAL_5]] to %[[VAL_4]] : !fir.ref<i32>
! CHECK:           omp.barrier
! CHECK:           %[[VAL_6:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_7:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
! CHECK:           %[[VAL_8:.*]] = arith.constant 1 : i32
! CHECK:           omp.wsloop   for  (%[[VAL_9:.*]]) : i32 = (%[[VAL_6]]) to (%[[VAL_7]]) inclusive step (%[[VAL_8]]) {
! CHECK:             fir.store %[[VAL_9]] to %[[VAL_3]] : !fir.ref<i32>
! CHECK:             fir.call @_QPsub4(%[[VAL_4]]) {{.*}}: (!fir.ref<i32>) -> ()
! CHECK:             omp.yield
! CHECK:           }
! CHECK:           omp.terminator
! CHECK:         }
! CHECK:         return
! CHECK:       }

subroutine combined_parallel_worksharing_loop()
  integer, save :: x6
  !$omp threadprivate(x6)

  !$omp parallel do copyin(x6)
    do i=1, x6
      call sub4(x6)
    end do
  !$omp end parallel do

end

! CHECK-LABEL: func.func @_QPcombined_parallel_sections() {
! CHECK:         %[[VAL_0:.*]] = fir.address_of(@_QFcombined_parallel_sectionsEx7) : !fir.ref<i32>
! CHECK:         %[[VAL_1:.*]] = omp.threadprivate %[[VAL_0]] : !fir.ref<i32> -> !fir.ref<i32>
! CHECK:         omp.parallel   {
! CHECK:           %[[VAL_2:.*]] = omp.threadprivate %[[VAL_0]] : !fir.ref<i32> -> !fir.ref<i32>
! CHECK:           %[[VAL_3:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:           fir.store %[[VAL_3]] to %[[VAL_2]] : !fir.ref<i32>
! CHECK:           omp.barrier
! CHECK:           omp.sections   {
! CHECK:             omp.section {
! CHECK:               fir.call @_QPsub5(%[[VAL_2]]) {{.*}}: (!fir.ref<i32>) -> ()
! CHECK:               omp.terminator
! CHECK:             }
! CHECK:             omp.section {
! CHECK:               fir.call @_QPsub6(%[[VAL_2]]) {{.*}}: (!fir.ref<i32>) -> ()
! CHECK:               omp.terminator
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           omp.terminator
! CHECK:         }
! CHECK:         return
! CHECK:       }

subroutine combined_parallel_sections()
  integer, save :: x7
  !$omp threadprivate(x7)

  !$omp parallel sections copyin(x7)
    !$omp section
      call sub5(x7)
    !$omp section
      call sub6(x7)
  !$omp end parallel sections

end


!CHECK: func.func @_QPcommon_1() {
!CHECK: %[[val_0:.*]] = fir.address_of(@c_) : !fir.ref<!fir.array<4xi8>>
!CHECK: %[[val_1:.*]] = omp.threadprivate %[[val_0]] : !fir.ref<!fir.array<4xi8>> -> !fir.ref<!fir.array<4xi8>>
!CHECK: %[[val_2:.*]] = fir.convert %[[val_1]] : (!fir.ref<!fir.array<4xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK: %[[val_c0:.*]] = arith.constant 0 : index
!CHECK: %[[val_3:.*]] = fir.coordinate_of %[[val_2]], %[[val_c0]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK: %[[val_4:.*]] = fir.convert %[[val_3]] : (!fir.ref<i8>) -> !fir.ref<i32>
!CHECK: %[[val_5:.*]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFcommon_1Ey"}
!CHECK: omp.parallel {
!CHECK: %[[val_6:.*]] = omp.threadprivate %[[val_0]] : !fir.ref<!fir.array<4xi8>> -> !fir.ref<!fir.array<4xi8>>
!CHECK: %[[val_7:.*]] = fir.convert %[[val_6]] : (!fir.ref<!fir.array<4xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK: %[[val_c0_0:.*]] = arith.constant 0 : index
!CHECK: %[[val_8:.*]] = fir.coordinate_of %[[val_7]], %[[val_c0_0]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK: %[[val_9:.*]] = fir.convert %[[val_8]] : (!fir.ref<i8>) -> !fir.ref<i32>
!CHECK: %[[val_10:.*]] = fir.load %[[val_4]] : !fir.ref<i32>
!CHECK: fir.store %[[val_10]] to %[[val_9]] : !fir.ref<i32>
!CHECK: omp.barrier
!CHECK: omp.sections   {
!CHECK: omp.section {
!CHECK: %[[val_11:.*]] = fir.load %[[val_9]] : !fir.ref<i32>
!CHECK: %[[val_c1_i32:.*]] = arith.constant 1 : i32
!CHECK: %[[val_12:.*]] = arith.addi %[[val_11]], %[[val_c1_i32]] : i32
!CHECK: fir.store %[[val_12]] to %[[val_5]] : !fir.ref<i32>
!CHECK: omp.terminator
!CHECK: }
!CHECK: omp.section {
!CHECK: %[[val_11:.*]] = fir.load %[[val_5]] : !fir.ref<i32>
!CHECK: %[[val_12:.*]] = fir.load %[[val_5]] : !fir.ref<i32>
!CHECK: %[[val_13:.*]] = arith.muli %[[val_11]], %[[val_12]] : i32
!CHECK: fir.store %[[val_13]] to %[[val_9]] : !fir.ref<i32>
!CHECK: omp.terminator
!CHECK: }
!CHECK: omp.terminator
!CHECK: }
!CHECK: return
!CHECK: }
subroutine common_1()
  integer :: x
  integer :: y
  common /c/ x
  !$omp threadprivate(/c/)

  !$omp parallel sections copyin(/c/)
      !$omp section
        y = x + 1
      !$omp section
        x = y * y
  !$omp end parallel sections
end subroutine

!CHECK: func.func @_QPcommon_2() {
!CHECK: %[[val_0:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFcommon_2Ei"}
!CHECK: %[[val_1:.*]] = fir.address_of(@d_) : !fir.ref<!fir.array<8xi8>>
!CHECK: %[[val_2:.*]] = omp.threadprivate %[[val_1]] : !fir.ref<!fir.array<8xi8>> -> !fir.ref<!fir.array<8xi8>>
!CHECK: %[[val_3:.*]] = fir.convert %[[val_2]] : (!fir.ref<!fir.array<8xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK: %[[val_c0:.*]] = arith.constant 0 : index
!CHECK: %[[val_4:.*]] = fir.coordinate_of %[[val_3]], %[[val_c0]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK: %[[val_5:.*]] = fir.convert %[[val_4]] : (!fir.ref<i8>) -> !fir.ref<i32>
!CHECK: %[[val_6:.*]] = fir.convert %[[val_2]] : (!fir.ref<!fir.array<8xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK: %[[val_c4:.*]] = arith.constant 4 : index
!CHECK: %[[val_7:.*]] = fir.coordinate_of %[[val_6]], %[[val_c4]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK: %[[val_8:.*]] = fir.convert %[[val_7]] : (!fir.ref<i8>) -> !fir.ref<i32>
!CHECK: omp.parallel {
!CHECK: %[[val_9:.*]] = fir.alloca i32 {adapt.valuebyref, pinned}
!CHECK: %[[val_10:.*]] = omp.threadprivate %[[val_1]] : !fir.ref<!fir.array<8xi8>> -> !fir.ref<!fir.array<8xi8>>
!CHECK: %[[val_11:.*]] = fir.convert %[[val_10]] : (!fir.ref<!fir.array<8xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK: %[[val_c0_0:.*]] = arith.constant 0 : index
!CHECK: %[[val_12:.*]] = fir.coordinate_of %[[val_11]], %[[val_c0_0]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK: %[[val_13:.*]] = fir.convert %[[val_12]] : (!fir.ref<i8>) -> !fir.ref<i32>
!CHECK: %[[val_14:.*]] = fir.convert %[[val_10]] : (!fir.ref<!fir.array<8xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK: %[[val_c4_1:.*]] = arith.constant 4 : index
!CHECK: %[[val_15:.*]] = fir.coordinate_of %[[val_14]], %[[val_c4_1]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK: %[[val_16:.*]] = fir.convert %[[val_15]] : (!fir.ref<i8>) -> !fir.ref<i32>
!CHECK: %[[val_17:.*]] = fir.load %[[val_5]] : !fir.ref<i32>
!CHECK: fir.store %[[val_17]] to %[[val_13]] : !fir.ref<i32>
!CHECK: %[[val_18:.*]] = fir.load %[[val_8]] : !fir.ref<i32>
!CHECK: fir.store %[[val_18]] to %[[val_16]] : !fir.ref<i32>
!CHECK: omp.barrier
!CHECK: %[[val_c1_i32:.*]] = arith.constant 1 : i32
!CHECK: %[[val_19:.*]] = fir.load %[[val_13]] : !fir.ref<i32>
!CHECK: %[[val_c1_i32_2:.*]] = arith.constant 1 : i32
!CHECK: omp.wsloop   for (%[[arg:.*]]) : i32 = (%[[val_c1_i32]]) to (%[[val_19]]) inclusive step (%[[val_c1_i32_2]]) {
!CHECK: fir.store %[[arg]] to %[[val_9]] : !fir.ref<i32>
!CHECK: %[[val_20:.*]] = fir.load %[[val_16]] : !fir.ref<i32>
!CHECK: %[[val_21:.*]] = fir.load %[[val_9]] : !fir.ref<i32>
!CHECK: %[[val_22:.*]] = arith.addi %[[val_20]], %[[val_21]] : i32
!CHECK: fir.store %[[val_22]] to %[[val_16]] : !fir.ref<i32>
!CHECK: omp.yield
!CHECK: }
!CHECK: omp.terminator
!CHECK: }
!CHECK: return
!CHECK: }
subroutine common_2()
  integer :: x
  integer :: y
  common /d/ x, y
  !$omp threadprivate(/d/)
  
  !$omp parallel do copyin(/d/)
     do i = 1, x
        y = y + i
     end do
  !$omp end parallel do
end subroutine
