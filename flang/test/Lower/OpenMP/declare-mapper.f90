! This test checks lowering of OpenMP declare mapper Directive.

! RUN: split-file %s %t
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 %t/omp-declare-mapper-1.f90 -o - | FileCheck %t/omp-declare-mapper-1.f90
! RUN  %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 %t/omp-declare-mapper-2.f90 -o - | FileCheck %t/omp-declare-mapper-2.f90

!--- omp-declare-mapper-1.f90
subroutine declare_mapper_1
   integer, parameter      :: nvals = 250
   type my_type
      integer              :: num_vals
      integer, allocatable :: values(:)
   end type

   type my_type2
      type(my_type)        :: my_type_var
      type(my_type)        :: temp
      real, dimension(nvals) :: unmapped
      real, dimension(nvals) :: arr
   end type
   type(my_type2)        :: t
   real                   :: x, y(nvals)
   !CHECK: omp.declare_mapper @default_my_type : ![[VAR_TYPE:.*]] {
   !CHECK:   %[[VAL_5:.*]] = fir.alloca ![[VAR_TYPE]] {bindc_name = "var"}
   !CHECK:   %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]] {uniq_name = "_QFdeclare_mapper_1Evar"} : (!fir.ref<![[VAR_TYPE]]>) -> (!fir.ref<![[VAR_TYPE]]>, !fir.ref<![[VAR_TYPE]]>)
   !CHECK:   %[[VAL_7:.*]] = hlfir.designate %[[VAL_6]]#0{{.*}} : (!fir.ref<![[VAR_TYPE]]>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
   !CHECK:   %[[VAL_8:.*]] = fir.load %[[VAL_7]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
   !CHECK:   %[[VAL_9:.*]] = arith.constant 0 : index
   !CHECK:   %[[VAL_10:.*]]:3 = fir.box_dims %[[VAL_8]], %[[VAL_9]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
   !CHECK:   %[[VAL_11:.*]] = arith.constant 1 : index
   !CHECK:   %[[VAL_12:.*]] = arith.constant 1 : index
   !CHECK:   %[[VAL_13:.*]] = arith.subi %[[VAL_12]], %[[VAL_10]]#0 : index
   !CHECK:   %[[VAL_14:.*]] = hlfir.designate %[[VAL_6]]#0{"num_vals"}   : (!fir.ref<![[VAR_TYPE]]>) -> !fir.ref<i32>
   !CHECK:   %[[VAL_15:.*]] = fir.load %[[VAL_14]] : !fir.ref<i32>
   !CHECK:   %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (i32) -> i64
   !CHECK:   %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (i64) -> index
   !CHECK:   %[[VAL_18:.*]] = arith.subi %[[VAL_17]], %[[VAL_10]]#0 : index
   !CHECK:   %[[VAL_19:.*]] = omp.map.bounds lower_bound(%[[VAL_13]] : index) upper_bound(%[[VAL_18]] : index) extent(%[[VAL_10]]#1 : index) stride(%[[VAL_11]] : index) start_idx(%[[VAL_10]]#0 : index)
   !CHECK:   %[[VAL_20:.*]] = arith.constant 1 : index
   !CHECK:   %[[VAL_21:.*]] = fir.coordinate_of %[[VAL_6]]#0, %[[VAL_20]] : (!fir.ref<![[VAR_TYPE]]>, index) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
   !CHECK:   %[[VAL_22:.*]] = fir.box_offset %[[VAL_21]] base_addr : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
   !CHECK:   %[[VAL_23:.*]] = omp.map.info var_ptr(%[[VAL_21]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.array<?xi32>) var_ptr_ptr(%[[VAL_22]] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) map_clauses(tofrom) capture(ByRef) bounds(%[[VAL_19]]) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>> {name = ""}
   !CHECK:   %[[VAL_24:.*]] = omp.map.info var_ptr(%[[VAL_21]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(to) capture(ByRef) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {name = "var%[[VAL_25:.*]](1:var%[[VAL_26:.*]])"}
   !CHECK:   %[[VAL_27:.*]] = omp.map.info var_ptr(%[[VAL_6]]#1 : !fir.ref<![[VAR_TYPE]]>, ![[VAR_TYPE]]) map_clauses(tofrom) capture(ByRef) members(%[[VAL_24]], %[[VAL_23]] : [1], [1, 0] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) -> !fir.ref<![[VAR_TYPE]]> {name = "var"}
   !CHECK:   omp.declare_mapper_info map_entries(%[[VAL_27]], %[[VAL_24]], %[[VAL_23]] : !fir.ref<![[VAR_TYPE]]>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>)
   !CHECK:   omp.terminator
   !CHECK: }
   !$omp declare mapper (my_type :: var) map (var, var%values (1:var%num_vals))
end subroutine declare_mapper_1

!--- omp-declare-mapper-2.f90
subroutine declare_mapper_2
   integer, parameter      :: nvals = 250
   type my_type
      integer              :: num_vals
      integer, allocatable :: values(:)
   end type

   type my_type2
      type(my_type)        :: my_type_var
      type(my_type)        :: temp
      real, dimension(nvals) :: unmapped
      real, dimension(nvals) :: arr
   end type
   type(my_type2)        :: t
   real                   :: x, y(nvals)
   !CHECK: omp.declare_mapper @my_mapper : ![[VAR_TYPE:.*]] {
   !CHECK:   %[[VAL_0:.*]] = fir.alloca ![[VAR_TYPE]] {bindc_name = "v"}
   !CHECK:   %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFdeclare_mapper_2Ev"} : (!fir.ref<![[VAR_TYPE]]>) -> (!fir.ref<![[VAR_TYPE]]>, !fir.ref<![[VAR_TYPE]]>)
   !CHECK:   %[[VAL_2:.*]] = arith.constant 250 : index
   !CHECK:   %[[VAL_3:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
   !CHECK:   %[[VAL_4:.*]] = hlfir.designate %[[VAL_1]]#0{"arr"}   shape %[[VAL_3]] : (!fir.ref<![[VAR_TYPE]]>, !fir.shape<1>) -> !fir.ref<!fir.array<250xf32>>
   !CHECK:   %[[VAL_5:.*]] = arith.constant 1 : index
   !CHECK:   %[[VAL_6:.*]] = arith.constant 0 : index
   !CHECK:   %[[VAL_7:.*]] = arith.subi %[[VAL_2]], %[[VAL_5]] : index
   !CHECK:   %[[VAL_8:.*]] = omp.map.bounds lower_bound(%[[VAL_6]] : index) upper_bound(%[[VAL_7]] : index) extent(%[[VAL_2]] : index) stride(%[[VAL_5]] : index) start_idx(%[[VAL_5]] : index)
   !CHECK:   %[[VAL_9:.*]] = omp.map.info var_ptr(%[[VAL_4]] : !fir.ref<!fir.array<250xf32>>, !fir.array<250xf32>) map_clauses(tofrom) capture(ByRef) bounds(%[[VAL_8]]) -> !fir.ref<!fir.array<250xf32>> {name = "v%[[VAL_10:.*]]"}
   !CHECK:   %[[VAL_11:.*]] = omp.map.info var_ptr(%[[VAL_12:.*]]#1 : !fir.ref<f32>, f32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<f32> {name = "x"}
   !CHECK:   %[[VAL_13:.*]] = arith.constant 0 : index
   !CHECK:   %[[VAL_14:.*]] = arith.constant 1 : index
   !CHECK:   %[[VAL_15:.*]] = arith.subi %[[VAL_16:.*]], %[[VAL_14]] : index
   !CHECK:   %[[VAL_17:.*]] = omp.map.bounds lower_bound(%[[VAL_13]] : index) upper_bound(%[[VAL_15]] : index) extent(%[[VAL_16]] : index) stride(%[[VAL_14]] : index) start_idx(%[[VAL_14]] : index)
   !CHECK:   %[[VAL_18:.*]] = omp.map.info var_ptr(%[[VAL_19:.*]]#1 : !fir.ref<!fir.array<250xf32>>, !fir.array<250xf32>) map_clauses(tofrom) capture(ByRef) bounds(%[[VAL_17]]) -> !fir.ref<!fir.array<250xf32>> {name = "y(:)"}
   !CHECK:   %[[VAL_20:.*]] = hlfir.designate %[[VAL_1]]#0{"temp"}   : (!fir.ref<![[VAR_TYPE]]>) -> !fir.ref<!fir.type<_QFdeclare_mapper_2Tmy_type{num_vals:i32,values:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>
   !CHECK:   %[[VAL_21:.*]] = omp.map.info var_ptr(%[[VAL_20]] : !fir.ref<!fir.type<_QFdeclare_mapper_2Tmy_type{num_vals:i32,values:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>, !fir.type<_QFdeclare_mapper_2Tmy_type{num_vals:i32,values:!fir.box<!fir.heap<!fir.array<?xi32>>>}>) map_clauses(exit_release_or_enter_alloc) capture(ByRef) -> !fir.ref<!fir.type<_QFdeclare_mapper_2Tmy_type{num_vals:i32,values:!fir.box<!fir.heap<!fir.array<?xi32>>>}>> {name = "v%[[VAL_22:.*]]"}
   !CHECK:   %[[VAL_23:.*]] = omp.map.info var_ptr(%[[VAL_1]]#1 : !fir.ref<![[VAR_TYPE]]>, ![[VAR_TYPE]]) map_clauses(tofrom) capture(ByRef) members(%[[VAL_9]], %[[VAL_21]] : [3], [1] : !fir.ref<!fir.array<250xf32>>, !fir.ref<!fir.type<_QFdeclare_mapper_2Tmy_type{num_vals:i32,values:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> !fir.ref<![[VAR_TYPE]]> {name = "v", partial_map = true}
   !CHECK:   omp.declare_mapper_info map_entries(%[[VAL_11]], %[[VAL_18]], %[[VAL_23]], %[[VAL_9]], %[[VAL_21]] : !fir.ref<f32>, !fir.ref<!fir.array<250xf32>>, !fir.ref<![[VAR_TYPE]]>, !fir.ref<!fir.array<250xf32>>, !fir.ref<!fir.type<_QFdeclare_mapper_2Tmy_type{num_vals:i32,values:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>)
   !CHECK:   omp.terminator
   !CHECK: }
   !$omp declare mapper (my_mapper : my_type2 :: v) map (v%arr, x, y(:)) &
   !$omp&                map (alloc : v%temp)
end subroutine declare_mapper_2
