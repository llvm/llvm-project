! This test checks lowering of OpenMP declare mapper Directive.

! RUN: split-file %s %t
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 %t/omp-declare-mapper-1.f90 -o - | FileCheck %t/omp-declare-mapper-1.f90
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 %t/omp-declare-mapper-2.f90 -o - | FileCheck %t/omp-declare-mapper-2.f90
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 %t/omp-declare-mapper-3.f90 -o - | FileCheck %t/omp-declare-mapper-3.f90
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 %t/omp-declare-mapper-4.f90 -o - | FileCheck %t/omp-declare-mapper-4.f90
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 %t/omp-declare-mapper-5.f90 -o - | FileCheck %t/omp-declare-mapper-5.f90
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 %t/omp-declare-mapper-6.f90 -o - | FileCheck %t/omp-declare-mapper-6.f90

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
   !CHECK:omp.declare_mapper @[[MY_TYPE_MAPPER:_QQFdeclare_mapper_1my_type\.omp\.default\.mapper]] : [[MY_TYPE:!fir\.type<_QFdeclare_mapper_1Tmy_type\{num_vals:i32,values:!fir\.box<!fir\.heap<!fir\.array<\?xi32>>>\}>]] {
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
   !CHECK:        %[[VAL_18:.*]] = fir.coordinate_of %[[VAL_1]]#0, values : (!fir.ref<[[MY_TYPE]]>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
   !CHECK:        %[[VAL_19:.*]] = fir.box_offset %[[VAL_18]] base_addr : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
   !CHECK:        %[[VAL_20:.*]] = omp.map.info var_ptr(%[[VAL_18]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, i32) map_clauses(tofrom) capture(ByRef) var_ptr_ptr(%[[VAL_19]] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) bounds(%[[VAL_16]]) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>> {name = ""}
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

!--- omp-declare-mapper-3.f90
subroutine declare_mapper_3
   type my_type
      integer              :: num_vals
      integer, allocatable :: values(:)
   end type

   type my_type2
      type(my_type)        :: my_type_var
      real, dimension(250) :: arr
   end type

   !CHECK:  omp.declare_mapper @[[MY_TYPE_MAPPER2:_QQFdeclare_mapper_3my_mapper2]] : [[MY_TYPE2:!fir\.type<_QFdeclare_mapper_3Tmy_type2\{my_type_var:!fir\.type<_QFdeclare_mapper_3Tmy_type\{num_vals:i32,values:!fir\.box<!fir\.heap<!fir\.array<\?xi32>>>}>,arr:!fir\.array<250xf32>}>]] {
   !CHECK:   ^bb0(%[[VAL_0:.*]]: !fir.ref<[[MY_TYPE2]]>):
   !CHECK:     %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFdeclare_mapper_3Ev"} : (!fir.ref<[[MY_TYPE2]]>) -> (!fir.ref<[[MY_TYPE2]]>, !fir.ref<[[MY_TYPE2]]>)
   !CHECK:     %[[VAL_2:.*]] = hlfir.designate %[[VAL_1]]#0{"my_type_var"}   : (!fir.ref<[[MY_TYPE2]]>) -> !fir.ref<[[MY_TYPE:!fir\.type<_QFdeclare_mapper_3Tmy_type\{num_vals:i32,values:!fir\.box<!fir\.heap<!fir\.array<\?xi32>>>}>]]>
   !CHECK:     %[[VAL_3:.*]] = omp.map.info var_ptr(%[[VAL_2]] : !fir.ref<[[MY_TYPE]]>, [[MY_TYPE]]) map_clauses(tofrom) capture(ByRef) mapper(@[[MY_TYPE_MAPPER:_QQFdeclare_mapper_3my_mapper]]) -> !fir.ref<[[MY_TYPE]]> {name = "v%[[VAL_4:.*]]"}
   !CHECK:     %[[VAL_5:.*]] = arith.constant 250 : index
   !CHECK:     %[[VAL_6:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
   !CHECK:     %[[VAL_7:.*]] = hlfir.designate %[[VAL_1]]#0{"arr"}   shape %[[VAL_6]] : (!fir.ref<[[MY_TYPE2]]>, !fir.shape<1>) -> !fir.ref<!fir.array<250xf32>>
   !CHECK:     %[[VAL_8:.*]] = arith.constant 1 : index
   !CHECK:     %[[VAL_9:.*]] = arith.constant 0 : index
   !CHECK:     %[[VAL_10:.*]] = arith.subi %[[VAL_5]], %[[VAL_8]] : index
   !CHECK:     %[[VAL_11:.*]] = omp.map.bounds lower_bound(%[[VAL_9]] : index) upper_bound(%[[VAL_10]] : index) extent(%[[VAL_5]] : index) stride(%[[VAL_8]] : index) start_idx(%[[VAL_8]] : index)
   !CHECK:     %[[VAL_12:.*]] = omp.map.info var_ptr(%[[VAL_7]] : !fir.ref<!fir.array<250xf32>>, !fir.array<250xf32>) map_clauses(tofrom) capture(ByRef) bounds(%[[VAL_11]]) -> !fir.ref<!fir.array<250xf32>> {name = "v%[[VAL_13:.*]]"}
   !CHECK:     %[[VAL_14:.*]] = omp.map.info var_ptr(%[[VAL_1]]#1 : !fir.ref<[[MY_TYPE2]]>, [[MY_TYPE2]]) map_clauses(tofrom) capture(ByRef) members(%[[VAL_3]], %[[VAL_12]] : [0], [1] : !fir.ref<[[MY_TYPE]]>, !fir.ref<!fir.array<250xf32>>) -> !fir.ref<[[MY_TYPE2]]> {name = "v", partial_map = true}
   !CHECK:     omp.declare_mapper.info map_entries(%[[VAL_14]], %[[VAL_3]], %[[VAL_12]] : !fir.ref<[[MY_TYPE2]]>, !fir.ref<[[MY_TYPE]]>, !fir.ref<!fir.array<250xf32>>)
   !CHECK:  }

   !CHECK:  omp.declare_mapper @[[MY_TYPE_MAPPER]] : [[MY_TYPE]] {
   !CHECK:   ^bb0(%[[VAL_0:.*]]: !fir.ref<[[MY_TYPE]]>):
   !CHECK:     %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFdeclare_mapper_3Evar"} : (!fir.ref<[[MY_TYPE]]>) -> (!fir.ref<[[MY_TYPE]]>, !fir.ref<[[MY_TYPE]]>)
   !CHECK:     %[[VAL_2:.*]] = hlfir.designate %[[VAL_1]]#0{"values"}   {fortran_attrs = #fir.var_attrs<allocatable>} : (!fir.ref<[[MY_TYPE]]>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
   !CHECK:     %[[VAL_3:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
   !CHECK:     %[[VAL_4:.*]] = fir.box_addr %[[VAL_3]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
   !CHECK:     %[[VAL_5:.*]] = arith.constant 0 : index
   !CHECK:     %[[VAL_6:.*]]:3 = fir.box_dims %[[VAL_3]], %[[VAL_5]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
   !CHECK:     %[[VAL_7:.*]] = arith.constant 0 : index
   !CHECK:     %[[VAL_8:.*]] = arith.constant 1 : index
   !CHECK:     %[[VAL_9:.*]] = arith.constant 1 : index
   !CHECK:     %[[VAL_10:.*]] = arith.subi %[[VAL_9]], %[[VAL_6]]#0 : index
   !CHECK:     %[[VAL_11:.*]] = hlfir.designate %[[VAL_1]]#0{"num_vals"}   : (!fir.ref<[[MY_TYPE]]>) -> !fir.ref<i32>
   !CHECK:     %[[VAL_12:.*]] = fir.load %[[VAL_11]] : !fir.ref<i32>
   !CHECK:     %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (i32) -> i64
   !CHECK:     %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i64) -> index
   !CHECK:     %[[VAL_15:.*]] = arith.subi %[[VAL_14]], %[[VAL_6]]#0 : index
   !CHECK:     %[[VAL_16:.*]] = omp.map.bounds lower_bound(%[[VAL_10]] : index) upper_bound(%[[VAL_15]] : index) extent(%[[VAL_6]]#1 : index) stride(%[[VAL_8]] : index) start_idx(%[[VAL_6]]#0 : index)
   !CHECK:     %[[VAL_18:.*]] = fir.coordinate_of %[[VAL_1]]#0, values : (!fir.ref<[[MY_TYPE]]>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
   !CHECK:     %[[VAL_19:.*]] = fir.box_offset %[[VAL_18]] base_addr : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
   !CHECK:     %[[VAL_20:.*]] = omp.map.info var_ptr(%[[VAL_18]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, i32) map_clauses(tofrom) capture(ByRef) var_ptr_ptr(%[[VAL_19]] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) bounds(%[[VAL_16]]) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>> {name = ""}
   !CHECK:     %[[VAL_21:.*]] = omp.map.info var_ptr(%[[VAL_18]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(to) capture(ByRef) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {name = "var%[[VAL_22:.*]](1:var%[[VAL_23:.*]])"}
   !CHECK:     %[[VAL_24:.*]] = omp.map.info var_ptr(%[[VAL_1]]#1 : !fir.ref<[[MY_TYPE]]>, [[MY_TYPE]]) map_clauses(tofrom) capture(ByRef) members(%[[VAL_21]], %[[VAL_20]] : [1], [1, 0] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) -> !fir.ref<[[MY_TYPE]]> {name = "var"}
   !CHECK:     omp.declare_mapper.info map_entries(%[[VAL_24]], %[[VAL_21]], %[[VAL_20]] : !fir.ref<[[MY_TYPE]]>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>)
   !CHECK:  }
   !$omp declare mapper (my_mapper : my_type :: var) map (var, var%values (1:var%num_vals))
   !$omp declare mapper (my_mapper2 : my_type2 :: v) map (mapper(my_mapper) : v%my_type_var) map (tofrom : v%arr)
end subroutine declare_mapper_3

!--- omp-declare-mapper-4.f90
subroutine declare_mapper_4
   type my_type
      integer              :: num
   end type

   !CHECK: omp.declare_mapper @[[MY_TYPE_MAPPER:_QQFdeclare_mapper_4my_type.omp.default.mapper]] : [[MY_TYPE:!fir\.type<_QFdeclare_mapper_4Tmy_type\{num:i32\}>]]
   !$omp declare mapper (my_type :: var) map (var%num)

   type(my_type) :: a
   integer :: b
   !CHECK: %{{.*}} = omp.map.info var_ptr(%{{.*}}#1 : !fir.ref<[[MY_TYPE]]>, [[MY_TYPE]]) map_clauses(tofrom) capture(ByRef) mapper(@[[MY_TYPE_MAPPER]]) -> !fir.ref<[[MY_TYPE]]> {name = "a"}
   !CHECK: %{{.*}} = omp.map.info var_ptr(%{{.*}}#1 : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32> {name = "b"}
   !$omp target map(a, b)
   a%num = 10
   b = 20
   !$omp end target

   !CHECK: %{{.*}} = omp.map.info var_ptr(%{{.*}} : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) mapper(@[[MY_TYPE_MAPPER]]) -> !fir.ref<i32> {name = "a%{{.*}}"}
   !$omp target map(a%num)
   a%num = 30
   !$omp end target

   !CHECK: %{{.*}} = omp.map.info var_ptr(%{{.*}}#1 : !fir.ref<[[MY_TYPE]]>, [[MY_TYPE]]) map_clauses(implicit, tofrom) capture(ByRef) mapper(@[[MY_TYPE_MAPPER]]) -> !fir.ref<[[MY_TYPE]]> {name = "a"}
   !$omp target
   a%num = 40
   !$omp end target
end subroutine declare_mapper_4

!--- omp-declare-mapper-5.f90
program declare_mapper_5
   implicit none

   type :: mytype
      integer :: x, y
   end type

   !CHECK: omp.declare_mapper @[[INNER_MAPPER_NAMED:_QQFFuse_innermy_mapper]] : [[MY_TYPE:!fir\.type<_QFTmytype\{x:i32,y:i32\}>]]
   !CHECK: omp.declare_mapper @[[INNER_MAPPER_DEFAULT:_QQFFuse_innermytype.omp.default.mapper]] : [[MY_TYPE]]
   !CHECK: omp.declare_mapper @[[OUTER_MAPPER_NAMED:_QQFmy_mapper]] : [[MY_TYPE]]
   !CHECK: omp.declare_mapper @[[OUTER_MAPPER_DEFAULT:_QQFmytype.omp.default.mapper]] : [[MY_TYPE]]
   !$omp declare mapper(mytype :: var) map(tofrom: var%x)
   !$omp declare mapper(my_mapper : mytype :: var) map(tofrom: var%y)

   type(mytype) :: a

   !CHECK: %{{.*}} = omp.map.info var_ptr(%{{.*}}#1 : !fir.ref<[[MY_TYPE]]>, [[MY_TYPE]]) map_clauses(implicit, tofrom) capture(ByRef) mapper(@[[OUTER_MAPPER_DEFAULT]]) -> !fir.ref<[[MY_TYPE]]> {name = "a"}
   !$omp target
   a%x = 10
   !$omp end target

   !CHECK: %{{.*}} = omp.map.info var_ptr(%{{.*}}#1 : !fir.ref<[[MY_TYPE]]>, [[MY_TYPE]]) map_clauses(tofrom) capture(ByRef) mapper(@[[OUTER_MAPPER_DEFAULT]]) -> !fir.ref<[[MY_TYPE]]> {name = "a"}
   !$omp target map(a)
   a%x = 10
   !$omp end target

   !CHECK: %{{.*}} = omp.map.info var_ptr(%{{.*}}#1 : !fir.ref<[[MY_TYPE]]>, [[MY_TYPE]]) map_clauses(tofrom) capture(ByRef) mapper(@[[OUTER_MAPPER_DEFAULT]]) -> !fir.ref<[[MY_TYPE]]> {name = "a"}
   !$omp target map(mapper(default) : a)
   a%x = 10
   !$omp end target

   !CHECK: %{{.*}} = omp.map.info var_ptr(%{{.*}}#1 : !fir.ref<[[MY_TYPE]]>, [[MY_TYPE]]) map_clauses(tofrom) capture(ByRef) mapper(@[[OUTER_MAPPER_NAMED]]) -> !fir.ref<[[MY_TYPE]]> {name = "a"}
   !$omp target map(mapper(my_mapper) : a)
   a%y = 10
   !$omp end target

contains
   subroutine use_outer()
      type(mytype) :: a

      !CHECK: %{{.*}} = omp.map.info var_ptr(%{{.*}}#1 : !fir.ref<[[MY_TYPE]]>, [[MY_TYPE]]) map_clauses(implicit, tofrom) capture(ByRef) mapper(@[[OUTER_MAPPER_DEFAULT]]) -> !fir.ref<[[MY_TYPE]]> {name = "a"}
      !$omp target
      a%x = 10
      !$omp end target

      !CHECK: %{{.*}} = omp.map.info var_ptr(%{{.*}}#1 : !fir.ref<[[MY_TYPE]]>, [[MY_TYPE]]) map_clauses(tofrom) capture(ByRef) mapper(@[[OUTER_MAPPER_DEFAULT]]) -> !fir.ref<[[MY_TYPE]]> {name = "a"}
      !$omp target map(a)
      a%x = 10
      !$omp end target

      !CHECK: %{{.*}} = omp.map.info var_ptr(%{{.*}}#1 : !fir.ref<[[MY_TYPE]]>, [[MY_TYPE]]) map_clauses(tofrom) capture(ByRef) mapper(@[[OUTER_MAPPER_DEFAULT]]) -> !fir.ref<[[MY_TYPE]]> {name = "a"}
      !$omp target map(mapper(default) : a)
      a%x = 10
      !$omp end target

      !CHECK: %{{.*}} = omp.map.info var_ptr(%{{.*}}#1 : !fir.ref<[[MY_TYPE]]>, [[MY_TYPE]]) map_clauses(tofrom) capture(ByRef) mapper(@[[OUTER_MAPPER_NAMED]]) -> !fir.ref<[[MY_TYPE]]> {name = "a"}
      !$omp target map(mapper(my_mapper) : a)
      a%y = 10
      !$omp end target
   end subroutine

   subroutine use_inner()
      !$omp declare mapper(mytype :: var) map(tofrom: var%x)
      !$omp declare mapper(my_mapper : mytype :: var) map(tofrom: var%y)

      type(mytype) :: a

      !CHECK: %{{.*}} = omp.map.info var_ptr(%{{.*}}#1 : !fir.ref<[[MY_TYPE]]>, [[MY_TYPE]]) map_clauses(implicit, tofrom) capture(ByRef) mapper(@[[INNER_MAPPER_DEFAULT]]) -> !fir.ref<[[MY_TYPE]]> {name = "a"}
      !$omp target
      a%x = 10
      !$omp end target

      !CHECK: %{{.*}} = omp.map.info var_ptr(%{{.*}}#1 : !fir.ref<[[MY_TYPE]]>, [[MY_TYPE]]) map_clauses(tofrom) capture(ByRef) mapper(@[[INNER_MAPPER_DEFAULT]]) -> !fir.ref<[[MY_TYPE]]> {name = "a"}
      !$omp target map(a)
      a%x = 10
      !$omp end target

      !CHECK: %{{.*}} = omp.map.info var_ptr(%{{.*}}#1 : !fir.ref<[[MY_TYPE]]>, [[MY_TYPE]]) map_clauses(tofrom) capture(ByRef) mapper(@[[INNER_MAPPER_DEFAULT]]) -> !fir.ref<[[MY_TYPE]]> {name = "a"}
      !$omp target map(mapper(default) : a)
      a%x = 10
      !$omp end target

      !CHECK: %{{.*}} = omp.map.info var_ptr(%{{.*}}#1 : !fir.ref<[[MY_TYPE]]>, [[MY_TYPE]]) map_clauses(tofrom) capture(ByRef) mapper(@[[INNER_MAPPER_NAMED]]) -> !fir.ref<[[MY_TYPE]]> {name = "a"}
      !$omp target map(mapper(my_mapper) : a)
      a%y = 10
      !$omp end target
   end subroutine
end program declare_mapper_5

!--- omp-declare-mapper-6.f90
subroutine declare_mapper_nested_parent
  type :: inner_t
    real, allocatable :: deep_arr(:)
  end type inner_t

  type, abstract :: base_t
    real, allocatable :: base_arr(:)
    type(inner_t) :: inner
  end type base_t

  type, extends(base_t) :: real_t
    real, allocatable :: real_arr(:)
  end type real_t

  !$omp declare mapper (custommapper : real_t :: t) map(tofrom: t%base_arr, t%real_arr)
  ! CHECK: omp.declare_mapper @{{.*custommapper}}
  ! CHECK-DAG: omp.map.info {{.*}} {name = "t%base_t%base_arr"}
  ! CHECK-DAG: omp.map.info {{.*}} {name = "t%real_arr"}
  ! CHECK: omp.declare_mapper.info

  type(real_t) :: r

  allocate(r%base_arr(10))
  allocate(r%inner%deep_arr(10))
  allocate(r%real_arr(10))
  r%base_arr = 1.0
  r%inner%deep_arr = 4.0
  r%real_arr = 0.0

  ! Check implicit maps for deep nested allocatable payloads not covered by mapper
  ! CHECK-DAG: omp.map.info {{.*}} {name = "r.deep_arr.implicit_map"}
  ! CHECK: omp.target
  !$omp target map(mapper(custommapper), tofrom: r)
    r%real_arr = r%base_arr(1) + r%inner%deep_arr(1)
  !$omp end target
end subroutine declare_mapper_nested_parent
