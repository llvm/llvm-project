! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s

! Check more advanced equivalence cases

! Several set of local and global equivalences in the same scope
! CHECK-LABEL: func.func @_QPtest_eq_sets
subroutine test_eq_sets
  DIMENSION Al(4), Bl(4)
  EQUIVALENCE (Al(1), Bl(2))
  ! CHECK-DAG: %[[albl:.*]] = fir.alloca !fir.array<20xi8>
  ! CHECK-DAG: %[[alAddr:.*]] = fir.coordinate_of %[[albl]], %c4{{.*}} : (!fir.ref<!fir.array<20xi8>>, index) -> !fir.ref<i8>
  ! CHECK-DAG: %[[al:.*]] = fir.convert %[[alAddr]] : (!fir.ref<i8>) -> !fir.ptr<!fir.array<4xf32>>
  ! CHECK-DAG: %[[al_decl:.*]]:2 = hlfir.declare %[[al]](%{{.*}}) storage(%[[albl]][4]) {uniq_name = "_QFtest_eq_setsEal"}
  ! CHECK-DAG: %[[blAddr:.*]] = fir.coordinate_of %[[albl]], %c0{{.*}} : (!fir.ref<!fir.array<20xi8>>, index) -> !fir.ref<i8>
  ! CHECK-DAG: %[[bl:.*]] = fir.convert %[[blAddr]] : (!fir.ref<i8>) -> !fir.ptr<!fir.array<4xf32>>
  ! CHECK-DAG: %[[bl_decl:.*]]:2 = hlfir.declare %[[bl]](%{{.*}}) storage(%[[albl]][0]) {uniq_name = "_QFtest_eq_setsEbl"}


  DIMENSION Il(2), Xl(2)
  EQUIVALENCE (Il(2), Xl(1))
  ! CHECK-DAG: %[[ilxl:.*]] = fir.alloca !fir.array<12xi8>
  ! CHECK-DAG: %[[igAddr:.*]] = fir.coordinate_of %{{.*}}, %c0{{.*}} : (!fir.ref<!fir.array<8xi8>>, index) -> !fir.ref<i8>
  ! CHECK-DAG: %[[ig:.*]] = fir.convert %[[igAddr]] : (!fir.ref<i8>) -> !fir.ptr<!fir.array<2xi32>>
  ! CHECK-DAG: %[[ig_decl:.*]]:2 = hlfir.declare %[[ig]](%{{.*}}) storage(%{{.*}}[0]) {uniq_name = "_QFtest_eq_setsEig"}
  ! CHECK-DAG: %[[ilAddr:.*]] = fir.coordinate_of %[[ilxl]], %c0{{.*}} : (!fir.ref<!fir.array<12xi8>>, index) -> !fir.ref<i8>
  ! CHECK-DAG: %[[il:.*]] = fir.convert %[[ilAddr]] : (!fir.ref<i8>) -> !fir.ptr<!fir.array<2xi32>>
  ! CHECK-DAG: %[[il_decl:.*]]:2 = hlfir.declare %[[il]](%{{.*}}) storage(%[[ilxl]][0]) {uniq_name = "_QFtest_eq_setsEil"}
  ! CHECK-DAG: %[[xgAddr:.*]] = fir.coordinate_of %{{.*}}, %c0{{.*}} : (!fir.ref<!fir.array<8xi8>>, index) -> !fir.ref<i8>
  ! CHECK-DAG: %[[xg:.*]] = fir.convert %[[xgAddr]] : (!fir.ref<i8>) -> !fir.ptr<!fir.array<2xf32>>
  ! CHECK-DAG: %[[xg_decl:.*]]:2 = hlfir.declare %[[xg]](%{{.*}}) storage(%{{.*}}[0]) {uniq_name = "_QFtest_eq_setsExg"}
  ! CHECK-DAG: %[[xlAddr:.*]] = fir.coordinate_of %[[ilxl]], %c4{{.*}} : (!fir.ref<!fir.array<12xi8>>, index) -> !fir.ref<i8>
  ! CHECK-DAG: %[[xl:.*]] = fir.convert %[[xlAddr]] : (!fir.ref<i8>) -> !fir.ptr<!fir.array<2xf32>>
  ! CHECK-DAG: %[[xl_decl:.*]]:2 = hlfir.declare %[[xl]](%{{.*}}) storage(%[[ilxl]][4]) {uniq_name = "_QFtest_eq_setsExl"}

  DIMENSION Ag(2), Bg(2)
  SAVE Ag, Bg
  EQUIVALENCE (Ag(1), Bg(2))
  ! CHECK-DAG: %[[agbg:.*]] = fir.address_of(@_QFtest_eq_setsEag) : !fir.ref<!fir.array<12xi8>>
  ! CHECK-DAG: %[[agAddr:.*]] = fir.coordinate_of %[[agbg]], %c4{{.*}} : (!fir.ref<!fir.array<12xi8>>, index) -> !fir.ref<i8>
  ! CHECK-DAG: %[[ag:.*]] = fir.convert %[[agAddr]] : (!fir.ref<i8>) -> !fir.ptr<!fir.array<2xf32>>
  ! CHECK-DAG: %[[ag_decl:.*]]:2 = hlfir.declare %[[ag]](%{{.*}}) storage(%[[agbg]][4]) {uniq_name = "_QFtest_eq_setsEag"}
  ! CHECK-DAG: %[[bgAddr:.*]] = fir.coordinate_of %[[agbg]], %c0{{.*}} : (!fir.ref<!fir.array<12xi8>>, index) -> !fir.ref<i8>
  ! CHECK-DAG: %[[bg:.*]] = fir.convert %[[bgAddr]] : (!fir.ref<i8>) -> !fir.ptr<!fir.array<2xf32>>
  ! CHECK-DAG: %[[bg_decl:.*]]:2 = hlfir.declare %[[bg]](%{{.*}}) storage(%[[agbg]][0]) {uniq_name = "_QFtest_eq_setsEbg"}

  DIMENSION Ig(2), Xg(2)
  SAVE Ig, Xg
  EQUIVALENCE (Ig(1), Xg(1))
  ! CHECK-DAG: %[[igxg:.*]] = fir.address_of(@_QFtest_eq_setsEig) : !fir.ref<!fir.array<8xi8>>

  ! CHECK: %[[alCast:.*]] = fir.convert %[[al_decl]]#0 : (!fir.ptr<!fir.array<4xf32>>) -> !fir.ref<!fir.array<4xf32>>
  ! CHECK: %[[blCast:.*]] = fir.convert %[[bl_decl]]#0 : (!fir.ptr<!fir.array<4xf32>>) -> !fir.ref<!fir.array<4xf32>>
  ! CHECK: %[[ilCast:.*]] = fir.convert %[[il_decl]]#0 : (!fir.ptr<!fir.array<2xi32>>) -> !fir.ref<!fir.array<2xi32>>
  ! CHECK: %[[xlCast:.*]] = fir.convert %[[xl_decl]]#0 : (!fir.ptr<!fir.array<2xf32>>) -> !fir.ref<!fir.array<2xf32>>
  ! CHECK: %[[agCast:.*]] = fir.convert %[[ag_decl]]#0 : (!fir.ptr<!fir.array<2xf32>>) -> !fir.ref<!fir.array<2xf32>>
  ! CHECK: %[[bgCast:.*]] = fir.convert %[[bg_decl]]#0 : (!fir.ptr<!fir.array<2xf32>>) -> !fir.ref<!fir.array<2xf32>>
  ! CHECK: %[[xgCast:.*]] = fir.convert %[[xg_decl]]#0 : (!fir.ptr<!fir.array<2xf32>>) -> !fir.ref<!fir.array<2xf32>>
  ! CHECK: %[[igCast:.*]] = fir.convert %[[ig_decl]]#0 : (!fir.ptr<!fir.array<2xi32>>) -> !fir.ref<!fir.array<2xi32>>

  call fooc(Al, Bl, Il, Xl, Ag, Bg, Xg, Ig)
  ! CHECK: fir.call @_QPfooc(%[[alCast]], %[[blCast]], %[[ilCast]], %[[xlCast]], %[[agCast]], %[[bgCast]], %[[xgCast]], %[[igCast]])

end subroutine


! Mixing global equivalence and entry
! CHECK-LABEL: func.func @_QPeq_and_entry_foo()
subroutine eq_and_entry_foo
  SAVE x, i
  DIMENSION :: x(2)
  EQUIVALENCE (x(2), i)
  call foo1(x, i)
  ! CHECK: %[[xi:.*]] = fir.address_of(@_QFeq_and_entry_fooEi) : !fir.ref<!fir.array<8xi8>>

  ! CHECK-DAG: %[[iOffset:.*]] = arith.constant 4 : index
  ! CHECK-DAG: %[[iAddr:.*]] = fir.coordinate_of %[[xi]], %[[iOffset]] : (!fir.ref<!fir.array<8xi8>>, index) -> !fir.ref<i8>
  ! CHECK-DAG: %[[i:.*]] = fir.convert %[[iAddr]] : (!fir.ref<i8>) -> !fir.ptr<i32>
  ! CHECK-DAG: %[[i_decl:.*]]:2 = hlfir.declare %[[i]] storage(%[[xi]][4]) {uniq_name = "_QFeq_and_entry_fooEi"}

  ! CHECK-DAG: %[[xOffset:.*]] = arith.constant 0 : index
  ! CHECK-DAG: %[[xAddr:.*]] = fir.coordinate_of %[[xi]], %[[xOffset]] : (!fir.ref<!fir.array<8xi8>>, index) -> !fir.ref<i8>
  ! CHECK-DAG: %[[x:.*]] = fir.convert %[[xAddr]] : (!fir.ref<i8>) -> !fir.ptr<!fir.array<2xf32>>
  ! CHECK-DAG: %[[x_decl:.*]]:2 = hlfir.declare %[[x]](%{{.*}}) storage(%[[xi]][0]) {uniq_name = "_QFeq_and_entry_fooEx"}

  call foo2(x, i)
  ! CHECK: %[[xCast:.*]] = fir.convert %[[x_decl]]#0 : (!fir.ptr<!fir.array<2xf32>>) -> !fir.ref<!fir.array<2xf32>>
  ! CHECK: %[[iCast:.*]] = fir.convert %[[i_decl]]#0 : (!fir.ptr<i32>) -> !fir.ref<i32>
  ! CHECK: fir.call @_QPfoo1(%[[xCast]], %[[iCast]]) {{.*}}: (!fir.ref<!fir.array<2xf32>>, !fir.ref<i32>) -> ()
  entry eq_and_entry_bar
  call foo2(x, i)
  ! CHECK: %[[xCast2:.*]] = fir.convert %[[x_decl]]#0 : (!fir.ptr<!fir.array<2xf32>>) -> !fir.ref<!fir.array<2xf32>>
  ! CHECK: %[[iCast2:.*]] = fir.convert %[[i_decl]]#0 : (!fir.ptr<i32>) -> !fir.ref<i32>
  ! CHECK: fir.call @_QPfoo2(%[[xCast2]], %[[iCast2]]) {{.*}}: (!fir.ref<!fir.array<2xf32>>, !fir.ref<i32>) -> ()
end

! CHECK-LABEL: func.func @_QPeq_and_entry_bar()
  ! CHECK: %[[xi:.*]] = fir.address_of(@_QFeq_and_entry_fooEi) : !fir.ref<!fir.array<8xi8>>

  ! CHECK-DAG: %[[iOffset:.*]] = arith.constant 4 : index
  ! CHECK-DAG: %[[iAddr:.*]] = fir.coordinate_of %[[xi]], %[[iOffset]] : (!fir.ref<!fir.array<8xi8>>, index) -> !fir.ref<i8>
  ! CHECK-DAG: %[[i:.*]] = fir.convert %[[iAddr]] : (!fir.ref<i8>) -> !fir.ptr<i32>
  ! CHECK-DAG: %[[i_decl:.*]]:2 = hlfir.declare %[[i]] storage(%[[xi]][4]) {uniq_name = "_QFeq_and_entry_fooEi"}

  ! CHECK-DAG: %[[xOffset:.*]] = arith.constant 0 : index
  ! CHECK-DAG: %[[xAddr:.*]] = fir.coordinate_of %[[xi]], %[[xOffset]] : (!fir.ref<!fir.array<8xi8>>, index) -> !fir.ref<i8>
  ! CHECK-DAG: %[[x:.*]] = fir.convert %[[xAddr]] : (!fir.ref<i8>) -> !fir.ptr<!fir.array<2xf32>>
  ! CHECK-DAG: %[[x_decl:.*]]:2 = hlfir.declare %[[x]](%{{.*}}) storage(%[[xi]][0]) {uniq_name = "_QFeq_and_entry_fooEx"}

  ! CHECK-NOT: fir.call @_QPfoo1
  ! CHECK: %[[xCast:.*]] = fir.convert %[[x_decl]]#0 : (!fir.ptr<!fir.array<2xf32>>) -> !fir.ref<!fir.array<2xf32>>
  ! CHECK: %[[iCast:.*]] = fir.convert %[[i_decl]]#0 : (!fir.ptr<i32>) -> !fir.ref<i32>
  ! CHECK: fir.call @_QPfoo2(%[[xCast]], %[[iCast]]) {{.*}}: (!fir.ref<!fir.array<2xf32>>, !fir.ref<i32>) -> ()


! Check that cases where equivalenced local variables and common blocks will
! share the same offset use the correct stores
! CHECK-LABEL: func.func @_QPeq_and_comm_same_offset()
subroutine eq_and_comm_same_offset
  real common_arr1(133),common_arr2(133)
  common /my_common_block/ common_arr1,common_arr2
  real arr1(133),arr2(133)
  real arr3(133,133),arr4(133,133)
  equivalence(arr1,common_arr1),(arr2,common_arr2)
  equivalence(arr3,arr4)

  ! CHECK: %[[arr4Store:.*]] = fir.alloca !fir.array<70756xi8> {uniq_name = "_QFeq_and_comm_same_offsetEarr3"}
  ! CHECK: %[[mcbAddr:.*]] = fir.address_of(@my_common_block_) : !fir.ref<!fir.array<1064xi8>>
  ! CHECK: %[[c0:.*]] = arith.constant 0 : index
  ! CHECK: %[[mcbCoor:.*]] = fir.coordinate_of %[[mcbAddr]], %[[c0]] : (!fir.ref<!fir.array<1064xi8>>, index) -> !fir.ref<i8>
  ! CHECK: %[[mcbCoorCast:.*]] = fir.convert %[[mcbCoor]] : (!fir.ref<i8>) -> !fir.ptr<!fir.array<133xf32>>
  ! CHECK: %[[c133:.*]] = arith.constant 133 : index
  ! CHECK: %[[shape1:.*]] = fir.shape %[[c133]] : (index) -> !fir.shape<1>
  ! CHECK: %[[arr1_decl:.*]]:2 = hlfir.declare %[[mcbCoorCast]](%[[shape1]]) storage(%[[mcbAddr]][0]) {uniq_name = "_QFeq_and_comm_same_offsetEarr1"}

  ! CHECK: %[[c1:.*]] = arith.constant 0 : index
  ! CHECK: %[[arr4Addr:.*]] = fir.coordinate_of %[[arr4Store]], %[[c1]] : (!fir.ref<!fir.array<70756xi8>>, index) -> !fir.ref<i8>
  ! CHECK: %[[arr4Cast:.*]] = fir.convert %[[arr4Addr]] : (!fir.ref<i8>) -> !fir.ptr<!fir.array<133x133xf32>>
  ! CHECK: %[[arr3_decl:.*]]:2 = hlfir.declare %[[arr4Cast]](%{{.*}}) storage(%[[arr4Store]][0]) {uniq_name = "_QFeq_and_comm_same_offsetEarr3"}

  ! CHECK: %[[c0_new:.*]] = arith.constant 0 : index
  ! CHECK: %[[arr4Addr_new:.*]] = fir.coordinate_of %[[arr4Store]], %[[c0_new]] : (!fir.ref<!fir.array<70756xi8>>, index) -> !fir.ref<i8>
  ! CHECK: %[[arr4Cast_new:.*]] = fir.convert %[[arr4Addr_new]] : (!fir.ref<i8>) -> !fir.ptr<!fir.array<133x133xf32>>
  ! CHECK: %[[arr4_decl:.*]]:2 = hlfir.declare %[[arr4Cast_new]](%{{.*}}) storage(%[[arr4Store]][0]) {uniq_name = "_QFeq_and_comm_same_offsetEarr4"}

  arr1(1) = 1
  ! CHECK:%[[mcbFinalAddr:.*]] = hlfir.designate %[[arr1_decl]]#0 (%c1{{.*}})
  ! CHECK:hlfir.assign %{{.*}} to %[[mcbFinalAddr]]

  arr4(1,1) = 2
  ! CHECK: %[[arr4FinalAddr:.*]] = hlfir.designate %[[arr4_decl]]#0 (%c1{{.*}}, %c1{{.*}})
  ! CHECK: hlfir.assign %{{.*}} to %[[arr4FinalAddr]]
end subroutine
