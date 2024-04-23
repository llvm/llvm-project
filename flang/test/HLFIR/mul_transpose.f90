! RUN: bbc -emit-hlfir %s -o - | FileCheck --check-prefix CHECK-BASE --check-prefix CHECK-ALL %s
! RUN: bbc -emit-hlfir %s -o - | fir-opt --canonicalize | FileCheck --check-prefix CHECK-CANONICAL --check-prefix CHECK-ALL %s
! RUN: bbc -emit-hlfir %s -o - | fir-opt --lower-hlfir-intrinsics | FileCheck --check-prefix CHECK-LOWERING --check-prefix CHECK-ALL %s
! RUN: bbc -emit-hlfir %s -o - | fir-opt --canonicalize | fir-opt --lower-hlfir-intrinsics | FileCheck --check-prefix CHECK-LOWERING-OPT --check-prefix CHECK-ALL %s
! RUN: bbc -emit-hlfir %s -o - | fir-opt --lower-hlfir-intrinsics | fir-opt --bufferize-hlfir | FileCheck --check-prefix CHECK-BUFFERING --check-prefix CHECK-ALL %s

! Test passing a hlfir.expr from one intrinsic to another
subroutine mul_transpose(a, b, res)
  real a(2,1), b(2,2), res(1,2)
  res = MATMUL(TRANSPOSE(a), b)
endsubroutine

! CHECK-ALL-LABEL:    func.func @_QPmul_transpose
! CHECK-ALL:              %[[A_ARG:.*]]: !fir.ref<!fir.array<2x1xf32>> {fir.bindc_name = "a"}
! CHECK-ALL:              %[[B_ARG:.*]]: !fir.ref<!fir.array<2x2xf32>> {fir.bindc_name = "b"}
! CHECK-ALL:              %[[RES_ARG:.*]]: !fir.ref<!fir.array<1x2xf32>> {fir.bindc_name = "res"}
! CHECK-ALL-DAG:        %[[A_DECL:.*]]:2 = hlfir.declare %[[A_ARG]]
! CHECK-ALL-DAG:        %[[B_DECL:.*]]:2 = hlfir.declare %[[B_ARG]]
! CHECK-ALL-DAG:        %[[RES_DECL:.*]]:2 = hlfir.declare %[[RES_ARG]]

! CHECK-BASE:           %[[TRANSPOSE_RES:.*]] = hlfir.transpose %[[A_DECL]]#0 : (!fir.ref<!fir.array<2x1xf32>>) -> !hlfir.expr<1x2xf32>
! CHECK-BASE-NEXT:      %[[MATMUL_RES:.*]] = hlfir.matmul %[[TRANSPOSE_RES]] %[[B_DECL]]#0 {{.*}}: (!hlfir.expr<1x2xf32>, !fir.ref<!fir.array<2x2xf32>>) -> !hlfir.expr<1x2xf32>
! CHECK-BASE-NEXT:      hlfir.assign %[[MATMUL_RES]] to %[[RES_DECL]]#0 : !hlfir.expr<1x2xf32>, !fir.ref<!fir.array<1x2xf32>>
! CHECK-BASE-NEXT:      hlfir.destroy %[[MATMUL_RES]]
! CHECK-BASE-NEXT:      hlfir.destroy %[[TRANSPOSE_RES]]

! CHECK-CANONICAL-NEXT: %[[CHAIN_RES:.*]] = hlfir.matmul_transpose %[[A_DECL]]#0 %[[B_DECL]]#0 : (!fir.ref<!fir.array<2x1xf32>>, !fir.ref<!fir.array<2x2xf32>>) -> !hlfir.expr<1x2xf32>
! CHECK-CANONICAL-NEXT: hlfir.assign %[[CHAIN_RES]] to %[[RES_DECL]]#0 : !hlfir.expr<1x2xf32>, !fir.ref<!fir.array<1x2xf32>>
! CHECK-CANONICAL-NEXT: hlfir.destroy %[[CHAIN_RES]]

! CHECK-LOWERING:       %[[A_BOX:.*]] = fir.embox %[[A_DECL]]#1(%{{.*}})
! CHECK-LOWERING:       %[[TRANSPOSE_CONV_RES:.*]] = fir.convert %[[TRANSPOSE_RES_BOX:.*]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK-LOWERING:       %[[A_BOX_CONV:.*]] = fir.convert %[[A_BOX]] : (!fir.box<!fir.array<2x1xf32>>) -> !fir.box<none>
! CHECK-LOWERING:       fir.call @_FortranATranspose(%[[TRANSPOSE_CONV_RES]], %[[A_BOX_CONV]], %[[LOC_STR1:.*]], %[[LOC_N1:.*]])
! CHECK-LOWERING:       %[[TRANSPOSE_RES_LD:.*]] = fir.load %[[TRANSPOSE_RES_BOX:.*]]
! CHECK-LOWERING:       %[[TRANSPOSE_RES_ADDR:.*]] = fir.box_addr %[[TRANSPOSE_RES_LD]]
! CHECK-LOWERING:       %[[TRANSPOSE_RES_VAR:.*]]:2 = hlfir.declare %[[TRANSPOSE_RES_ADDR]]({{.*}}) {uniq_name = ".tmp.intrinsic_result"}
! CHECK-LOWERING:       %[[TRANSPOSE_EXPR:.*]] = hlfir.as_expr %[[TRANSPOSE_RES_VAR]]#0 move {{.*}} : (!fir.box<!fir.array<?x?xf32>>, i1) -> !hlfir.expr<?x?xf32>
! CHECK-LOWERING:       %[[TRANSPOSE_ASSOC:.*]]:3 = hlfir.associate %[[TRANSPOSE_EXPR]]({{.*}}) {adapt.valuebyref}
! CHECK-LOWERING:           (!hlfir.expr<?x?xf32>, !fir.shape<2>) -> (!fir.ref<!fir.array<1x2xf32>>, !fir.ref<!fir.array<1x2xf32>>, i1)

! CHECK-LOWERING:       %[[LHS_BOX:.*]] = fir.embox %[[TRANSPOSE_ASSOC]]#1
! CHECK-LOWERING:       %[[B_BOX:.*]] = fir.embox %[[B_DECL]]#1(%{{.*}})
! CHECK-LOWERING:       %[[MUL_CONV_RES:.*]] = fir.convert %[[MUL_RES_BOX:.*]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK-LOWERING:       %[[LHS_CONV:.*]] = fir.convert %[[LHS_BOX]] : (!fir.box<!fir.array<1x2xf32>>) -> !fir.box<none>
! CHECK-LOWERING:       %[[B_BOX_CONV:.*]] = fir.convert %[[B_BOX]] : (!fir.box<!fir.array<2x2xf32>>) -> !fir.box<none>
! CHECK-LOWERING:       fir.call @_FortranAMatmul(%[[MUL_CONV_RES]], %[[LHS_CONV]], %[[B_BOX_CONV]], %[[LOC_STR2:.*]], %[[LOC_N2:.*]])
! CHECK-LOWERING:       %[[MUL_RES_LD:.*]] = fir.load %[[MUL_RES_BOX:.*]]
! CHECK-LOWERING:       %[[MUL_RES_ADDR:.*]] = fir.box_addr %[[MUL_RES_LD]]
! CHECK-LOWERING:       %[[MUL_RES_VAR:.*]]:2 = hlfir.declare %[[MUL_RES_ADDR]]({{.*}}) {uniq_name = ".tmp.intrinsic_result"}
! CHECK-LOWERING:       %[[MUL_EXPR:.*]] = hlfir.as_expr %[[MUL_RES_VAR]]#0 move {{.*}} : (!fir.box<!fir.array<?x?xf32>>, i1) -> !hlfir.expr<?x?xf32>

! CHECK-LOWERING:       hlfir.end_associate %[[TRANSPOSE_ASSOC]]#1, %[[TRANSPOSE_ASSOC]]#2 : !fir.ref<!fir.array<1x2xf32>>, i1
! CHECK-LOWERING-NEXT:  hlfir.assign %[[MUL_EXPR]] to %[[RES_DECL]]#0 : !hlfir.expr<?x?xf32>, !fir.ref<!fir.array<1x2xf32>>
! CHECK-LOWERING-NEXT:  hlfir.destroy %[[MUL_EXPR]]
! CHECK-LOWERING-NEXT:  hlfir.destroy %[[TRANSPOSE_EXPR]]

! CHECK-LOWERING-OPT:   %[[LHS_BOX:.*]] = fir.embox %[[A_DECL]]#1(%{{.*}})
! CHECK-LOWERING-OPT:   %[[B_BOX:.*]] = fir.embox %[[B_DECL]]#1(%{{.*}})
! CHECK-LOWERING-OPT:   %[[MUL_CONV_RES:.*]] = fir.convert %[[MUL_RES_BOX:.*]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK-LOWERING-OPT:   %[[LHS_CONV:.*]] = fir.convert %[[LHS_BOX]] : (!fir.box<!fir.array<2x1xf32>>) -> !fir.box<none>
! CHECK-LOWERING-OPT:   %[[B_BOX_CONV:.*]] = fir.convert %[[B_BOX]] : (!fir.box<!fir.array<2x2xf32>>) -> !fir.box<none>
! CHECK-LOWERING-OPT:   fir.call @_FortranAMatmulTranspose(%[[MUL_CONV_RES]], %[[LHS_CONV]], %[[B_BOX_CONV]], %[[LOC_STR2:.*]], %[[LOC_N2:.*]])
! CHECK-LOWERING-OPT:   %[[MUL_RES_LD:.*]] = fir.load %[[MUL_RES_BOX:.*]]
! CHECK-LOWERING-OPT:   %[[MUL_RES_ADDR:.*]] = fir.box_addr %[[MUL_RES_LD]]
! CHECK-LOWERING-OPT:   %[[MUL_RES_VAR:.*]]:2 = hlfir.declare %[[MUL_RES_ADDR]]({{.*}}) {uniq_name = ".tmp.intrinsic_result"}
! CHECK-LOWERING-OPT:   %[[MUL_EXPR:.*]] = hlfir.as_expr %[[MUL_RES_VAR]]#0 move {{.*}} : (!fir.box<!fir.array<?x?xf32>>, i1) -> !hlfir.expr<?x?xf32>
! CHECK-LOWERING-OPT:   hlfir.assign %[[MUL_EXPR]] to %[[RES_DECL]]#0 : !hlfir.expr<?x?xf32>, !fir.ref<!fir.array<1x2xf32>>
! CHECK-LOWERING-OPT:   hlfir.destroy %[[MUL_EXPR]]

! [argument handling unchanged]
! CHECK-BUFFERING:      fir.call @_FortranATranspose(
! CHECK-BUFFERING:      %[[TRANSPOSE_RES_LD:.*]] = fir.load %[[TRANSPOSE_RES_BOX:.*]]
! CHECK-BUFFERING:      %[[TRANSPOSE_RES_ADDR:.*]] = fir.box_addr %[[TRANSPOSE_RES_LD]]
! CHECK-BUFFERING:      %[[TRANSPOSE_RES_VAR:.*]]:2 = hlfir.declare %[[TRANSPOSE_RES_ADDR]]({{.*}}) {uniq_name = ".tmp.intrinsic_result"}
! CHECK-BUFFERING:      %[[TUPLE0:.*]] = fir.undefined tuple<!fir.box<!fir.array<?x?xf32>>, i1>
! CHECK-BUFFERING:      %[[TUPLE1:.*]] = fir.insert_value %[[TUPLE0]], {{.*}}, [1 : index]
! CHECK-BUFFERING:      %[[TUPLE2:.*]] = fir.insert_value %[[TUPLE1]], %[[TRANSPOSE_RES_VAR]]#0, [0 : index]

! CHECK-BUFFERING:      %[[TRANSPOSE_RES_REF:.*]] = fir.convert %[[TRANSPOSE_RES_VAR]]#1 : (!fir.heap<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<1x2xf32>>
! CHECK-BUFFERING:      %[[TRANSPOSE_RES_BOX:.*]] = fir.embox %[[TRANSPOSE_RES_REF]]({{.*}})
! CHECK-BUFFERING:      %[[LHS_CONV:.*]] = fir.convert %[[TRANSPOSE_RES_BOX]] : (!fir.box<!fir.array<1x2xf32>>) -> !fir.box<none>
! [argument handling unchanged]
! CHECK-BUFFERING:      fir.call @_FortranAMatmul(
! CHECK-BUFFERING:      %[[MUL_RES_LD:.*]] = fir.load %[[MUL_RES_BOX:.*]]
! CHECK-BUFFERING:      %[[MUL_RES_ADDR:.*]] = fir.box_addr %[[MUL_RES_LD]]
! CHECK-BUFFERING:      %[[MUL_RES_VAR:.*]]:2 = hlfir.declare %[[MUL_RES_ADDR]]({{.*}}) {uniq_name = ".tmp.intrinsic_result"}
! CHECK-BUFFERING:      %[[TUPLE3:.*]] = fir.undefined tuple<!fir.box<!fir.array<?x?xf32>>, i1>
! CHECK-BUFFERING:      %[[TUPLE4:.*]] = fir.insert_value %[[TUPLE3]], {{.*}}, [1 : index]
! CHECK-BUFFERING:      %[[TUPLE5:.*]] = fir.insert_value %[[TUPLE4]], %[[MUL_RES_VAR]]#0, [0 : index]

! CHECK-BUFFERING:      %[[TRANSPOSE_RES_HEAP:.*]] = fir.convert %[[TRANSPOSE_RES_REF]] : (!fir.ref<!fir.array<1x2xf32>>) -> !fir.heap<!fir.array<1x2xf32>>
! CHECK-BUFFERING-NEXT: fir.freemem %[[TRANSPOSE_RES_HEAP]]
! CHECK-BUFFERING-NEXT: hlfir.assign %[[MUL_RES_VAR]]#0 to %[[RES_DECL]]#0 : !fir.box<!fir.array<?x?xf32>>, !fir.ref<!fir.array<1x2xf32>>
! CHECK-BUFFERING-NEXT: %[[MUL_RES_HEAP:.*]] = fir.box_addr %[[MUL_RES_VAR]]#0 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.heap<!fir.array<?x?xf32>>
! CHECK-BUFFERING-NEXT: fir.freemem %[[MUL_RES_HEAP]]

! CHECK-ALL-NEXT:       return
! CHECK-ALL-NEXT:     }
