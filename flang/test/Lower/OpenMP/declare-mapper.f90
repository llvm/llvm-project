! This test checks lowering of OpenMP declare mapper Directive.

! RUN: split-file %s %t
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 %t/omp-declare-mapper-1.f90 -o - | FileCheck %t/omp-declare-mapper-1.f90
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 %t/omp-declare-mapper-2.f90 -o - | FileCheck %t/omp-declare-mapper-2.f90

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
   !CHECK:omp.declare_mapper @[[MY_TYPE_MAPPER:_QQFdeclare_mapper_1my_type\.default]] : [[MY_TYPE:!fir\.type<_QFdeclare_mapper_1Tmy_type\{num_vals:i32,values:!fir\.box<!fir\.heap<!fir\.array<\?xi32>>>\}>]] {
   !CHECK:      ^bb0(%[[VAL_0:.*]]: !fir.ref<[[MY_TYPE]]>):
   !CHECK:        %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFdeclare_mapper_1Evar"} : (!fir.ref<[[MY_TYPE]]>) -> (!fir.ref<[[MY_TYPE]]>, !fir.ref<[[MY_TYPE]]>)
   !CHECK:        %[[VAL_2:.*]] = hlfir.designate %[[VAL_1]]#0{"values"}   {fortran_attrs = #fir.var_attrs<allocatable>} : (!fir.ref<[[MY_TYPE]]>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
   !CHECK:        %[[VAL_3:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
   !CHECK:        %[[VAL_4:.*]] = fir.box_addr %[[VAL_3]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
   !CHECK:        %[[VAL_5:.*]] = arith.constant 0 : index
   !CHECK:        %[[VAL_6:.*]]:3 = fir.box_dims %[[VAL_3]], %[[VAL_5]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
   !CHECK:        %[[VAL_7:.*]] = arith.constant 0 : index
   !CHECK:        %[[VAL_8:.*]] = arith.constant 1 : index
   !CHECK:        %[[VAL_9:.*]] = arith.constant 1 : index
   !CHECK:        %[[VAL_10:.*]] = arith.subi %[[VAL_9]], %[[VAL_6]]#0 : index
   !CHECK:        %[[VAL_11:.*]] = hlfir.designate %[[VAL_1]]#0{"num_vals"}   : (!fir.ref<[[MY_TYPE]]>) -> !fir.ref<i32>
   !CHECK:        %[[VAL_12:.*]] = fir.load %[[VAL_11]] : !fir.ref<i32>
   !CHECK:        %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (i32) -> i64
   !CHECK:        %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i64) -> index
   !CHECK:        %[[VAL_15:.*]] = arith.subi %[[VAL_14]], %[[VAL_6]]#0 : index
   !CHECK:        %[[VAL_16:.*]] = omp.map.bounds lower_bound(%[[VAL_10]] : index) upper_bound(%[[VAL_15]] : index) extent(%[[VAL_6]]#1 : index) stride(%[[VAL_8]] : index) start_idx(%[[VAL_6]]#0 : index)
   !CHECK:        %[[VAL_17:.*]] = arith.constant 1 : index
   !CHECK:        %[[VAL_18:.*]] = fir.coordinate_of %[[VAL_1]]#0, %[[VAL_17]] : (!fir.ref<[[MY_TYPE]]>, index) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
   !CHECK:        %[[VAL_19:.*]] = fir.box_offset %[[VAL_18]] base_addr : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
   !CHECK:        %[[VAL_20:.*]] = omp.map.info var_ptr(%[[VAL_18]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.array<?xi32>) var_ptr_ptr(%[[VAL_19]] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) map_clauses(tofrom) capture(ByRef) bounds(%[[VAL_16]]) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>> {name = ""}
   !CHECK:        %[[VAL_21:.*]] = omp.map.info var_ptr(%[[VAL_18]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(to) capture(ByRef) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {name = "var%[[VAL_22:.*]](1:var%[[VAL_23:.*]])"}
   !CHECK:        %[[VAL_24:.*]] = omp.map.info var_ptr(%[[VAL_1]]#1 : !fir.ref<[[MY_TYPE]]>, [[MY_TYPE]]) map_clauses(tofrom) capture(ByRef) members(%[[VAL_21]], %[[VAL_20]] : [1], [1, 0] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) -> !fir.ref<[[MY_TYPE]]> {name = "var"}
   !CHECK:        omp.declare_mapper.info map_entries(%[[VAL_24]], %[[VAL_21]], %[[VAL_20]] : !fir.ref<[[MY_TYPE]]>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>)
   !CHECK:      }
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
   real                  :: x, y(nvals)
   !CHECK:omp.declare_mapper @[[MY_TYPE_MAPPER:_QQFdeclare_mapper_2my_mapper]] : [[MY_TYPE:!fir\.type<_QFdeclare_mapper_2Tmy_type2\{my_type_var:!fir\.type<_QFdeclare_mapper_2Tmy_type\{num_vals:i32,values:!fir\.box<!fir\.heap<!fir\.array<\?xi32>>>\}>,temp:!fir\.type<_QFdeclare_mapper_2Tmy_type\{num_vals:i32,values:!fir\.box<!fir\.heap<!fir\.array<\?xi32>>>\}>,unmapped:!fir\.array<250xf32>,arr:!fir\.array<250xf32>\}>]] {
   !CHECK:      ^bb0(%[[VAL_0:.*]]: !fir.ref<[[MY_TYPE]]>):
   !CHECK:        %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFdeclare_mapper_2Ev"} : (!fir.ref<[[MY_TYPE]]>) -> (!fir.ref<[[MY_TYPE]]>, !fir.ref<[[MY_TYPE]]>)
   !CHECK:        %[[VAL_2:.*]] = arith.constant 250 : index
   !CHECK:        %[[VAL_3:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
   !CHECK:        %[[VAL_4:.*]] = hlfir.designate %[[VAL_1]]#0{"arr"}   shape %[[VAL_3]] : (!fir.ref<[[MY_TYPE]]>, !fir.shape<1>) -> !fir.ref<!fir.array<250xf32>>
   !CHECK:        %[[VAL_5:.*]] = arith.constant 1 : index
   !CHECK:        %[[VAL_6:.*]] = arith.constant 0 : index
   !CHECK:        %[[VAL_7:.*]] = arith.subi %[[VAL_2]], %[[VAL_5]] : index
   !CHECK:        %[[VAL_8:.*]] = omp.map.bounds lower_bound(%[[VAL_6]] : index) upper_bound(%[[VAL_7]] : index) extent(%[[VAL_2]] : index) stride(%[[VAL_5]] : index) start_idx(%[[VAL_5]] : index)
   !CHECK:        %[[VAL_9:.*]] = omp.map.info var_ptr(%[[VAL_4]] : !fir.ref<!fir.array<250xf32>>, !fir.array<250xf32>) map_clauses(tofrom) capture(ByRef) bounds(%[[VAL_8]]) -> !fir.ref<!fir.array<250xf32>> {name = "v%[[VAL_10:.*]]"}
   !CHECK:        %[[VAL_11:.*]] = hlfir.designate %[[VAL_1]]#0{"temp"}   : (!fir.ref<[[MY_TYPE]]>) -> !fir.ref<!fir.type<_QFdeclare_mapper_2Tmy_type{num_vals:i32,values:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>
   !CHECK:        %[[VAL_12:.*]] = omp.map.info var_ptr(%[[VAL_11]] : !fir.ref<!fir.type<_QFdeclare_mapper_2Tmy_type{num_vals:i32,values:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>, !fir.type<_QFdeclare_mapper_2Tmy_type{num_vals:i32,values:!fir.box<!fir.heap<!fir.array<?xi32>>>}>) map_clauses(exit_release_or_enter_alloc) capture(ByRef) -> !fir.ref<!fir.type<_QFdeclare_mapper_2Tmy_type{num_vals:i32,values:!fir.box<!fir.heap<!fir.array<?xi32>>>}>> {name = "v%[[VAL_13:.*]]"}
   !CHECK:        %[[VAL_14:.*]] = omp.map.info var_ptr(%[[VAL_1]]#1 : !fir.ref<[[MY_TYPE]]>, [[MY_TYPE]]) map_clauses(tofrom) capture(ByRef) members(%[[VAL_9]], %[[VAL_12]] : [3], [1] : !fir.ref<!fir.array<250xf32>>, !fir.ref<!fir.type<_QFdeclare_mapper_2Tmy_type{num_vals:i32,values:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> !fir.ref<[[MY_TYPE]]> {name = "v", partial_map = true}
   !CHECK:        omp.declare_mapper.info map_entries(%[[VAL_14]], %[[VAL_9]], %[[VAL_12]] : !fir.ref<[[MY_TYPE]]>, !fir.ref<!fir.array<250xf32>>, !fir.ref<!fir.type<_QFdeclare_mapper_2Tmy_type{num_vals:i32,values:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>)
   !CHECK:      }
   !$omp declare mapper (my_mapper : my_type2 :: v) map (v%arr) map (alloc : v%temp)
end subroutine declare_mapper_2
