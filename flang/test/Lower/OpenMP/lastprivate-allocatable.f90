! RUN: %flang_fc1 -emit-hlfir -o - -fopenmp %s | FileCheck %s
! RUN: bbc -emit-hlfir -o - -fopenmp %s | FileCheck %s

! CHECK-LABEL:   func.func @_QQmain() attributes {fir.bindc_name = "lastprivate_allocatable"} {
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.heap<i32>> {bindc_name = "a", uniq_name = "_QFEa"}
! CHECK:           %[[VAL_1:.*]] = fir.zero_bits !fir.heap<i32>
! CHECK:           %[[VAL_2:.*]] = fir.embox %[[VAL_1]] : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
! CHECK:           fir.store %[[VAL_2]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = {{.*}}<allocatable>, uniq_name = "_QFEa"} : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>)
! CHECK:           %[[VAL_4:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]] {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           omp.parallel {
! CHECK:             omp.barrier
! CHECK:             %[[VAL_6:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_7:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_8:.*]] = arith.constant 1 : i32
! CHECK:             omp.wsloop private(@_QFEa_private_box_heap_i32 %[[VAL_3]]#0 -> %[[VAL_9:.*]], @_QFEi_private_i32 %[[VAL_5]]#0 -> %[[VAL_10:.*]] : !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<i32>) {
! CHECK:               omp.loop_nest (%[[VAL_11:.*]]) : i32 = (%[[VAL_6]]) to (%[[VAL_7]]) inclusive step (%[[VAL_8]]) {
! CHECK:                 %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_9]] {fortran_attrs = {{.*}}<allocatable>, uniq_name = "_QFEa"} : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>)
! CHECK:                 %[[VAL_13:.*]]:2 = hlfir.declare %[[VAL_10]] {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:                 hlfir.assign %[[VAL_11]] to %[[VAL_13]]#0 : i32, !fir.ref<i32>
! CHECK:                 %[[VAL_14:.*]] = arith.constant 42 : i32
! CHECK:                 hlfir.assign %[[VAL_14]] to %[[VAL_12]]#0 realloc : i32, !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:                 %[[VAL_15:.*]] = arith.constant 1 : i32
! CHECK:                 %[[VAL_16:.*]] = arith.constant 1 : i32
! CHECK:                 %[[VAL_17:.*]] = arith.addi %[[VAL_11]], %[[VAL_16]] : i32
! CHECK:                 %[[VAL_18:.*]] = arith.constant 0 : i32
! CHECK:                 %[[VAL_19:.*]] = arith.cmpi slt, %[[VAL_16]], %[[VAL_18]] : i32
! CHECK:                 %[[VAL_20:.*]] = arith.cmpi slt, %[[VAL_17]], %[[VAL_15]] : i32
! CHECK:                 %[[VAL_21:.*]] = arith.cmpi sgt, %[[VAL_17]], %[[VAL_15]] : i32
! CHECK:                 %[[VAL_22:.*]] = arith.select %[[VAL_19]], %[[VAL_20]], %[[VAL_21]] : i1
! CHECK:                 fir.if %[[VAL_22]] {
! CHECK:                   hlfir.assign %[[VAL_17]] to %[[VAL_13]]#0 : i32, !fir.ref<i32>
! CHECK:                   %[[VAL_23:.*]] = fir.load %[[VAL_12]]#0 : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:                   %[[VAL_24:.*]] = fir.box_addr %[[VAL_23]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
! CHECK:                   %[[VAL_25:.*]] = fir.load %[[VAL_24]] : !fir.heap<i32>
! CHECK:                   hlfir.assign %[[VAL_25]] to %[[VAL_3]]#0 realloc : i32, !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:                 }
! CHECK:                 omp.yield
! CHECK:               }
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }
program lastprivate_allocatable
  integer, allocatable :: a
  integer :: i
  ! a is unallocated here
  !$omp parallel do lastprivate(a)
  do i=1,1
    a = 42
  enddo
  !$omp end parallel do
  ! a should be allocated here
end program

! CHECK-LABEL:   func.func @_QPlastprivate_realloc() {
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>> {bindc_name = "a", uniq_name = "_QFlastprivate_reallocEa"}
! CHECK:           %[[VAL_1:.*]] = fir.zero_bits !fir.heap<!fir.array<?xcomplex<f32>>>
! CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_4:.*]] = fir.embox %[[VAL_1]](%[[VAL_3]]) : (!fir.heap<!fir.array<?xcomplex<f32>>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>
! CHECK:           fir.store %[[VAL_4]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>>
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = {{.*}}<allocatable>, uniq_name = "_QFlastprivate_reallocEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>>)
! CHECK:           %[[VAL_6:.*]] = arith.constant false
! CHECK:           %[[VAL_7:.*]] = fir.absent !fir.box<none>
! CHECK:           %[[VAL_8:.*]] = fir.address_of(@_QQclX9c577f22af2c5c17f89170ff454cf10e) : !fir.ref<!fir.char<1,83>>
! CHECK:           %[[VAL_9:.*]] = arith.constant {{.*}} : i32
! CHECK:           %[[VAL_10:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_11:.*]] = arith.constant 2 : i32
! CHECK:           %[[VAL_12:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_5]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_10]] : (index) -> i64
! CHECK:           %[[VAL_15:.*]] = fir.convert %[[VAL_11]] : (i32) -> i64
! CHECK:           fir.call @_FortranAAllocatableSetBounds(%[[VAL_13]], %[[VAL_12]], %[[VAL_14]], %[[VAL_15]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
! CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_5]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_8]] : (!fir.ref<!fir.char<1,83>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_18:.*]] = fir.call @_FortranAAllocatableAllocate(%[[VAL_16]], %[[VAL_6]], %[[VAL_7]], %[[VAL_17]], %[[VAL_9]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           omp.parallel {
! CHECK:             %[[VAL_19:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>> {bindc_name = "a", pinned, uniq_name = "_QFlastprivate_reallocEa"}
! CHECK:             %[[VAL_20:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>>
! CHECK:             %[[VAL_21:.*]] = fir.box_addr %[[VAL_20]] : (!fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>) -> !fir.heap<!fir.array<?xcomplex<f32>>>
! CHECK:             %[[VAL_22:.*]] = fir.convert %[[VAL_21]] : (!fir.heap<!fir.array<?xcomplex<f32>>>) -> i64
! CHECK:             %[[VAL_23:.*]] = arith.constant 0 : i64
! CHECK:             %[[VAL_24:.*]] = arith.cmpi ne, %[[VAL_22]], %[[VAL_23]] : i64
! CHECK:             fir.if %[[VAL_24]] {
! CHECK:               %[[VAL_25:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>>
! CHECK:               %[[VAL_26:.*]] = arith.constant 0 : index
! CHECK:               %[[VAL_27:.*]]:3 = fir.box_dims %[[VAL_25]], %[[VAL_26]] : (!fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>, index) -> (index, index, index)
! CHECK:               %[[VAL_28:.*]] = arith.constant 0 : index
! CHECK:               %[[VAL_29:.*]] = arith.cmpi sgt, %[[VAL_27]]#1, %[[VAL_28]] : index
! CHECK:               %[[VAL_30:.*]] = arith.select %[[VAL_29]], %[[VAL_27]]#1, %[[VAL_28]] : index
! CHECK:               %[[VAL_31:.*]] = fir.allocmem !fir.array<?xcomplex<f32>>, %[[VAL_30]] {fir.must_be_heap = true, uniq_name = "_QFlastprivate_reallocEa.alloc"}
! CHECK:               %[[VAL_32:.*]] = fir.shape_shift %[[VAL_27]]#0, %[[VAL_30]] : (index, index) -> !fir.shapeshift<1>
! CHECK:               %[[VAL_33:.*]] = fir.embox %[[VAL_31]](%[[VAL_32]]) : (!fir.heap<!fir.array<?xcomplex<f32>>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>
! CHECK:               fir.store %[[VAL_33]] to %[[VAL_19]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>>
! CHECK:             } else {
! CHECK:               %[[VAL_34:.*]] = fir.zero_bits !fir.heap<!fir.array<?xcomplex<f32>>>
! CHECK:               %[[VAL_35:.*]] = arith.constant 0 : index
! CHECK:               %[[VAL_36:.*]] = fir.shape %[[VAL_35]] : (index) -> !fir.shape<1>
! CHECK:               %[[VAL_37:.*]] = fir.embox %[[VAL_34]](%[[VAL_36]]) : (!fir.heap<!fir.array<?xcomplex<f32>>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>
! CHECK:               fir.store %[[VAL_37]] to %[[VAL_19]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>>
! CHECK:             }
! CHECK:             %[[VAL_38:.*]]:2 = hlfir.declare %[[VAL_19]] {fortran_attrs = {{.*}}<allocatable>, uniq_name = "_QFlastprivate_reallocEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>>)
! CHECK:             omp.sections {
! CHECK:               omp.section {
! CHECK:                 %[[VAL_39:.*]] = arith.constant false
! CHECK:                 %[[VAL_40:.*]] = fir.absent !fir.box<none>
! CHECK:                 %[[VAL_41:.*]] = fir.address_of(@_QQclX9c577f22af2c5c17f89170ff454cf10e) : !fir.ref<!fir.char<1,83>>
! CHECK:                 %[[VAL_42:.*]] = arith.constant {{.*}} : i32
! CHECK:                 %[[VAL_43:.*]] = fir.convert %[[VAL_38]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:                 %[[VAL_44:.*]] = fir.convert %[[VAL_41]] : (!fir.ref<!fir.char<1,83>>) -> !fir.ref<i8>
! CHECK:                 %[[VAL_45:.*]] = fir.call @_FortranAAllocatableDeallocate(%[[VAL_43]], %[[VAL_39]], %[[VAL_40]], %[[VAL_44]], %[[VAL_42]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:                 %[[VAL_46:.*]] = arith.constant false
! CHECK:                 %[[VAL_47:.*]] = fir.absent !fir.box<none>
! CHECK:                 %[[VAL_48:.*]] = fir.address_of(@_QQclX9c577f22af2c5c17f89170ff454cf10e) : !fir.ref<!fir.char<1,83>>
! CHECK:                 %[[VAL_49:.*]] = arith.constant {{.*}} : i32
! CHECK:                 %[[VAL_50:.*]] = arith.constant 1 : index
! CHECK:                 %[[VAL_51:.*]] = arith.constant 3 : i32
! CHECK:                 %[[VAL_52:.*]] = arith.constant 0 : i32
! CHECK:                 %[[VAL_53:.*]] = fir.convert %[[VAL_38]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:                 %[[VAL_54:.*]] = fir.convert %[[VAL_50]] : (index) -> i64
! CHECK:                 %[[VAL_55:.*]] = fir.convert %[[VAL_51]] : (i32) -> i64
! CHECK:                 fir.call @_FortranAAllocatableSetBounds(%[[VAL_53]], %[[VAL_52]], %[[VAL_54]], %[[VAL_55]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
! CHECK:                 %[[VAL_56:.*]] = fir.convert %[[VAL_38]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:                 %[[VAL_57:.*]] = fir.convert %[[VAL_48]] : (!fir.ref<!fir.char<1,83>>) -> !fir.ref<i8>
! CHECK:                 %[[VAL_58:.*]] = fir.call @_FortranAAllocatableAllocate(%[[VAL_56]], %[[VAL_46]], %[[VAL_47]], %[[VAL_57]], %[[VAL_49]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:                 %[[VAL_59:.*]] = fir.load %[[VAL_38]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>>
! CHECK:                 hlfir.assign %[[VAL_59]] to %[[VAL_5]]#0 realloc : !fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>>
! CHECK:                 omp.terminator
! CHECK:               }
! CHECK:               omp.terminator
! CHECK:             }
! CHECK:             %[[VAL_60:.*]] = fir.load %[[VAL_38]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>>
! CHECK:             %[[VAL_61:.*]] = fir.box_addr %[[VAL_60]] : (!fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>) -> !fir.heap<!fir.array<?xcomplex<f32>>>
! CHECK:             %[[VAL_62:.*]] = fir.convert %[[VAL_61]] : (!fir.heap<!fir.array<?xcomplex<f32>>>) -> i64
! CHECK:             %[[VAL_63:.*]] = arith.constant 0 : i64
! CHECK:             %[[VAL_64:.*]] = arith.cmpi ne, %[[VAL_62]], %[[VAL_63]] : i64
! CHECK:             fir.if %[[VAL_64]] {
! CHECK:               %[[VAL_65:.*]] = arith.constant false
! CHECK:               %[[VAL_66:.*]] = fir.absent !fir.box<none>
! CHECK:               %[[VAL_67:.*]] = fir.address_of(@_QQclX9c577f22af2c5c17f89170ff454cf10e) : !fir.ref<!fir.char<1,83>>
! CHECK:               %[[VAL_68:.*]] = arith.constant {{.*}} : i32
! CHECK:               %[[VAL_69:.*]] = fir.convert %[[VAL_38]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:               %[[VAL_70:.*]] = fir.convert %[[VAL_67]] : (!fir.ref<!fir.char<1,83>>) -> !fir.ref<i8>
! CHECK:               %[[VAL_71:.*]] = fir.call @_FortranAAllocatableDeallocate(%[[VAL_69]], %[[VAL_65]], %[[VAL_66]], %[[VAL_70]], %[[VAL_68]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           %[[VAL_72:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>>
! CHECK:           %[[VAL_73:.*]] = fir.box_addr %[[VAL_72]] : (!fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>) -> !fir.heap<!fir.array<?xcomplex<f32>>>
! CHECK:           %[[VAL_74:.*]] = fir.convert %[[VAL_73]] : (!fir.heap<!fir.array<?xcomplex<f32>>>) -> i64
! CHECK:           %[[VAL_75:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_76:.*]] = arith.cmpi ne, %[[VAL_74]], %[[VAL_75]] : i64
! CHECK:           fir.if %[[VAL_76]] {
! CHECK:             %[[VAL_77:.*]] = arith.constant false
! CHECK:             %[[VAL_78:.*]] = fir.absent !fir.box<none>
! CHECK:             %[[VAL_79:.*]] = fir.address_of(@_QQclX9c577f22af2c5c17f89170ff454cf10e) : !fir.ref<!fir.char<1,83>>
! CHECK:             %[[VAL_80:.*]] = arith.constant {{.*}} : i32
! CHECK:             %[[VAL_81:.*]] = fir.convert %[[VAL_5]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:             %[[VAL_82:.*]] = fir.convert %[[VAL_79]] : (!fir.ref<!fir.char<1,83>>) -> !fir.ref<i8>
! CHECK:             %[[VAL_83:.*]] = fir.call @_FortranAAllocatableDeallocate(%[[VAL_81]], %[[VAL_77]], %[[VAL_78]], %[[VAL_82]], %[[VAL_80]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           }
! CHECK:           return
! CHECK:         }
subroutine lastprivate_realloc()
  complex, allocatable :: a(:)

  allocate(a(2))
  !$omp parallel
    !$omp sections lastprivate(a)
      !$omp section
        deallocate(a)
        allocate(a(3))
    !$omp end sections
  !$omp end parallel
end subroutine
