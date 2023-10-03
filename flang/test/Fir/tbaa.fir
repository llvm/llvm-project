// RUN: fir-opt %s --split-input-file --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu apply-tbaa=true" | FileCheck %s
// RUN: fir-opt %s --split-input-file --fir-to-llvm-ir="target=aarch64-unknown-linux-gnu apply-tbaa=true" | FileCheck %s

module {
  func.func @tbaa(%arg0: !fir.class<!fir.array<?xnone>> {fir.bindc_name = "a"}) {
    %c9_i8 = arith.constant 9 : i8
    %c0_i64 = arith.constant 0 : i64
    %c10_i32 = arith.constant 10 : i32
    %0 = fir.coordinate_of %arg0, %c0_i64 : (!fir.class<!fir.array<?xnone>>, i64) -> !fir.ref<none>
    %1 = fir.embox %0 source_box %arg0 : (!fir.ref<none>, !fir.class<!fir.array<?xnone>>) -> !fir.class<none>
    %2 = fir.box_typecode %1 : (!fir.class<none>) -> i8
    %3 = arith.cmpi eq, %2, %c9_i8 : i8
    cf.cond_br %3, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %4 = fir.box_addr %1 : (!fir.class<none>) -> !fir.ref<i32>
    fir.store %c10_i32 to %4 : !fir.ref<i32>
    cf.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    return
  }
}

// CHECK-DAG:     #[[ROOT:.*]] = #llvm.tbaa_root<id = "Flang Type TBAA Root">
// CHECK-DAG:     #[[ANYACC:.*]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[ROOT]], 0>}>
// CHECK-DAG:     #[[ANYDACC:.*]] = #llvm.tbaa_type_desc<id = "any data access", members = {<#[[ANYACC]], 0>}>
// CHECK-DAG:     #[[BOXMEM:.*]] = #llvm.tbaa_type_desc<id = "descriptor member", members = {<#[[ANYACC]], 0>}>
// CHECK-DAG:     #[[$BOXT:.*]] = #llvm.tbaa_tag<base_type = #[[BOXMEM]], access_type = #[[BOXMEM]], offset = 0>
// CHECK-DAG:     #[[$DATAT:.*]] = #llvm.tbaa_tag<base_type = #[[ANYDACC]], access_type = #[[ANYDACC]], offset = 0>

// CHECK-LABEL:   llvm.func @tbaa(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>> {fir.bindc_name = "a"}) {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_2:.*]] = llvm.alloca %[[VAL_1]] x !llvm.struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, ptr<i8>, array<1 x i64>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr<struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, ptr<i8>, array<1 x i64>)>>
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.constant(9 : i8) : i8
// CHECK:           %[[VAL_4:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:           %[[VAL_5:.*]] = llvm.mlir.constant(10 : i32) : i32
// CHECK:           %[[VAL_6:.*]] = llvm.getelementptr %[[VAL_0]][0, 0] : (!llvm.ptr<struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>>) -> !llvm.ptr<ptr<struct<()>>>
// CHECK:           %[[VAL_7:.*]] = llvm.load %[[VAL_6]] {tbaa = [#[[$BOXT]]]} : !llvm.ptr<ptr<struct<()>>>
// CHECK:           %[[VAL_8:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:           %[[VAL_9:.*]] = llvm.getelementptr %[[VAL_0]][0, 7, 0, 2] : (!llvm.ptr<struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>>) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_10:.*]] = llvm.load %[[VAL_9]] {tbaa = [#[[$BOXT]]]} : !llvm.ptr<i64>
// CHECK:           %[[VAL_11:.*]] = llvm.mul %[[VAL_4]], %[[VAL_10]]  : i64
// CHECK:           %[[VAL_12:.*]] = llvm.add %[[VAL_11]], %[[VAL_8]]  : i64
// CHECK:           %[[VAL_13:.*]] = llvm.bitcast %[[VAL_7]] : !llvm.ptr<struct<()>> to !llvm.ptr<i8>
// CHECK:           %[[VAL_14:.*]] = llvm.getelementptr %[[VAL_13]]{{\[}}%[[VAL_12]]] : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_15:.*]] = llvm.bitcast %[[VAL_14]] : !llvm.ptr<i8> to !llvm.ptr<struct<()>>
// CHECK:           %[[VAL_16:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:           %[[VAL_17:.*]] = llvm.mlir.constant(-1 : i32) : i32
// CHECK:           %[[VAL_18:.*]] = llvm.getelementptr %[[VAL_0]][0, 8] : (!llvm.ptr<struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>>) -> !llvm.ptr<ptr<i8>>
// CHECK:           %[[VAL_19:.*]] = llvm.load %[[VAL_18]] {tbaa = [#[[$BOXT]]]} : !llvm.ptr<ptr<i8>>
// CHECK:           %[[VAL_20:.*]] = llvm.getelementptr %[[VAL_0]][0, 1] : (!llvm.ptr<struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>>) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_21:.*]] = llvm.load %[[VAL_20]] {tbaa = [#[[$BOXT]]]} : !llvm.ptr<i64>
// CHECK:           %[[VAL_22:.*]] = llvm.getelementptr %[[VAL_0]][0, 4] : (!llvm.ptr<struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>>) -> !llvm.ptr<i32>
// CHECK:           %[[VAL_23:.*]] = llvm.load %[[VAL_22]] {tbaa = [#[[$BOXT]]]} : !llvm.ptr<i32>
// CHECK:           %[[VAL_24:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, ptr<i8>, array<1 x i64>)>
// CHECK:           %[[VAL_25:.*]] = llvm.insertvalue %[[VAL_21]], %[[VAL_24]][1] : !llvm.struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, ptr<i8>, array<1 x i64>)>
// CHECK:           %[[VAL_26:.*]] = llvm.mlir.constant(20180515 : i32) : i32
// CHECK:           %[[VAL_27:.*]] = llvm.insertvalue %[[VAL_26]], %[[VAL_25]][2] : !llvm.struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, ptr<i8>, array<1 x i64>)>
// CHECK:           %[[VAL_28:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_29:.*]] = llvm.trunc %[[VAL_28]] : i32 to i8
// CHECK:           %[[VAL_30:.*]] = llvm.insertvalue %[[VAL_29]], %[[VAL_27]][3] : !llvm.struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, ptr<i8>, array<1 x i64>)>
// CHECK:           %[[VAL_31:.*]] = llvm.trunc %[[VAL_23]] : i32 to i8
// CHECK:           %[[VAL_32:.*]] = llvm.insertvalue %[[VAL_31]], %[[VAL_30]][4] : !llvm.struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, ptr<i8>, array<1 x i64>)>
// CHECK:           %[[VAL_33:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_34:.*]] = llvm.trunc %[[VAL_33]] : i32 to i8
// CHECK:           %[[VAL_35:.*]] = llvm.insertvalue %[[VAL_34]], %[[VAL_32]][5] : !llvm.struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, ptr<i8>, array<1 x i64>)>
// CHECK:           %[[VAL_36:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_37:.*]] = llvm.trunc %[[VAL_36]] : i32 to i8
// CHECK:           %[[VAL_38:.*]] = llvm.insertvalue %[[VAL_37]], %[[VAL_35]][6] : !llvm.struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, ptr<i8>, array<1 x i64>)>
// CHECK:           %[[VAL_39:.*]] = llvm.bitcast %[[VAL_19]] : !llvm.ptr<i8> to !llvm.ptr<i8>
// CHECK:           %[[VAL_40:.*]] = llvm.insertvalue %[[VAL_39]], %[[VAL_38]][7] : !llvm.struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, ptr<i8>, array<1 x i64>)>
// CHECK:           %[[VAL_41:.*]] = llvm.bitcast %[[VAL_15]] : !llvm.ptr<struct<()>> to !llvm.ptr<struct<()>>
// CHECK:           %[[VAL_42:.*]] = llvm.insertvalue %[[VAL_41]], %[[VAL_40]][0] : !llvm.struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, ptr<i8>, array<1 x i64>)>
// CHECK:           llvm.store %[[VAL_42]], %[[VAL_2]] {tbaa = [#[[$BOXT]]]} : !llvm.ptr<struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, ptr<i8>, array<1 x i64>)>>
// CHECK:           %[[VAL_43:.*]] = llvm.getelementptr %[[VAL_2]][0, 4] : (!llvm.ptr<struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, ptr<i8>, array<1 x i64>)>>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_44:.*]] = llvm.load %[[VAL_43]] {tbaa = [#[[$BOXT]]]} : !llvm.ptr<i8>
// CHECK:           %[[VAL_45:.*]] = llvm.icmp "eq" %[[VAL_44]], %[[VAL_3]] : i8
// CHECK:           llvm.cond_br %[[VAL_45]], ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK:           %[[VAL_46:.*]] = llvm.getelementptr %[[VAL_2]][0, 0] : (!llvm.ptr<struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, ptr<i8>, array<1 x i64>)>>) -> !llvm.ptr<ptr<i32>>
// CHECK:           %[[VAL_47:.*]] = llvm.load %[[VAL_46]] {tbaa = [#[[$BOXT]]]} : !llvm.ptr<ptr<i32>>
// CHECK:           llvm.store %[[VAL_5]], %[[VAL_47]] {tbaa = [#[[$DATAT]]]} : !llvm.ptr<i32>
// CHECK:           llvm.br ^bb2
// CHECK:         ^bb2:
// CHECK:           llvm.return
// CHECK:         }

// -----

module {
  func.func @tbaa() {
    %c0 = arith.constant 0 : index
    %c8_i32 = arith.constant 8 : i32
    %c-1_i32 = arith.constant -1 : i32
    %0 = fir.address_of(@_QFEx) : !fir.ref<!fir.class<!fir.ptr<!fir.array<?xnone>>>>
    %1 = fir.address_of(@_QQcl.2E2F64756D6D792E66393000) : !fir.ref<!fir.char<1,12>>
    %2 = fir.convert %1 : (!fir.ref<!fir.char<1,12>>) -> !fir.ref<i8>
    %3 = fir.call @_FortranAioBeginExternalListOutput(%c-1_i32, %2, %c8_i32) fastmath<contract> : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
    %4 = fir.load %0 : !fir.ref<!fir.class<!fir.ptr<!fir.array<?xnone>>>>
    %5:3 = fir.box_dims %4, %c0 : (!fir.class<!fir.ptr<!fir.array<?xnone>>>, index) -> (index, index, index)
    %6 = fircg.ext_rebox %4 origin %5#0 : (!fir.class<!fir.ptr<!fir.array<?xnone>>>, index) -> !fir.class<!fir.array<?xnone>>
    %7 = fir.convert %6 : (!fir.class<!fir.array<?xnone>>) -> !fir.box<none>
    %8 = fir.call @_FortranAioOutputDescriptor(%3, %7) fastmath<contract> : (!fir.ref<i8>, !fir.box<none>) -> i1
    %9 = fir.call @_FortranAioEndIoStatement(%3) fastmath<contract> : (!fir.ref<i8>) -> i32
    return
  }
  func.func private @_FortranAioBeginExternalListOutput(i32, !fir.ref<i8>, i32) -> !fir.ref<i8> attributes {fir.io, fir.runtime}
  func.func private @_FortranAioOutputDescriptor(!fir.ref<i8>, !fir.box<none>) -> i1 attributes {fir.io, fir.runtime}
  func.func private @_FortranAioEndIoStatement(!fir.ref<i8>) -> i32 attributes {fir.io, fir.runtime}
  fir.global linkonce @_QQcl.2E2F64756D6D792E66393000 constant : !fir.char<1,12> {
    %0 = fir.string_lit "./dummy.f90\00"(12) : !fir.char<1,12>
    fir.has_value %0 : !fir.char<1,12>
  }
  fir.global internal @_QFEx : !fir.class<!fir.ptr<!fir.array<?xnone>>> {
    %c0 = arith.constant 0 : index
    %0 = fir.zero_bits !fir.ptr<!fir.array<?xnone>>
    %1 = fircg.ext_embox %0(%c0) : (!fir.ptr<!fir.array<?xnone>>, index) -> !fir.class<!fir.ptr<!fir.array<?xnone>>>
    fir.has_value %1 : !fir.class<!fir.ptr<!fir.array<?xnone>>>
  }
}

// CHECK-DAG:     #[[ROOT:.*]] = #llvm.tbaa_root<id = "Flang Type TBAA Root">
// CHECK-DAG:     #[[ANYACC:.*]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[ROOT]], 0>}>
// CHECK-DAG:     #[[BOXMEM:.*]] = #llvm.tbaa_type_desc<id = "descriptor member", members = {<#[[ANYACC]], 0>}>
// CHECK-DAG:     #[[$BOXT:.*]] = #llvm.tbaa_tag<base_type = #[[BOXMEM]], access_type = #[[BOXMEM]], offset = 0>

// CHECK-LABEL:   llvm.func @tbaa() {
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr<struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>>
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_3:.*]] = llvm.alloca %[[VAL_2]] x !llvm.struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr<struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>>
// CHECK:           %[[VAL_4:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           %[[VAL_5:.*]] = llvm.mlir.constant(8 : i32) : i32
// CHECK:           %[[VAL_6:.*]] = llvm.mlir.constant(-1 : i32) : i32
// CHECK:           %[[VAL_7:.*]] = llvm.mlir.addressof @_QFEx : !llvm.ptr<struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>>
// CHECK:           %[[VAL_8:.*]] = llvm.mlir.addressof @_QQcl.2E2F64756D6D792E66393000 : !llvm.ptr<array<12 x i8>>
// CHECK:           %[[VAL_9:.*]] = llvm.bitcast %[[VAL_8]] : !llvm.ptr<array<12 x i8>> to !llvm.ptr<i8>
// CHECK:           %[[VAL_10:.*]] = llvm.call @_FortranAioBeginExternalListOutput(%[[VAL_6]], %[[VAL_9]], %[[VAL_5]]) {fastmathFlags = #llvm.fastmath<contract>} : (i32, !llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_11:.*]] = llvm.load %[[VAL_7]] {tbaa = [#[[$BOXT]]]} : !llvm.ptr<struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>>
// CHECK:           llvm.store %[[VAL_11]], %[[VAL_3]] {tbaa = [#[[$BOXT]]]} : !llvm.ptr<struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>>
// CHECK:           %[[VAL_12:.*]] = llvm.getelementptr %[[VAL_3]][0, 7, %[[VAL_4]], 0] : (!llvm.ptr<struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>>, i64) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_13:.*]] = llvm.load %[[VAL_12]] {tbaa = [#[[$BOXT]]]} : !llvm.ptr<i64>
// CHECK:           %[[VAL_14:.*]] = llvm.getelementptr %[[VAL_3]][0, 7, %[[VAL_4]], 1] : (!llvm.ptr<struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>>, i64) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_15:.*]] = llvm.load %[[VAL_14]] {tbaa = [#[[$BOXT]]]} : !llvm.ptr<i64>
// CHECK:           %[[VAL_16:.*]] = llvm.getelementptr %[[VAL_3]][0, 7, %[[VAL_4]], 2] : (!llvm.ptr<struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>>, i64) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_17:.*]] = llvm.load %[[VAL_16]] {tbaa = [#[[$BOXT]]]} : !llvm.ptr<i64>
// CHECK:           %[[VAL_18:.*]] = llvm.getelementptr %[[VAL_3]][0, 8] : (!llvm.ptr<struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>>) -> !llvm.ptr<ptr<i8>>
// CHECK:           %[[VAL_19:.*]] = llvm.load %[[VAL_18]] {tbaa = [#[[$BOXT]]]} : !llvm.ptr<ptr<i8>>
// CHECK:           %[[VAL_20:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:           %[[VAL_21:.*]] = llvm.mlir.constant(-1 : i32) : i32
// CHECK:           %[[VAL_22:.*]] = llvm.getelementptr %[[VAL_3]][0, 1] : (!llvm.ptr<struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>>) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_23:.*]] = llvm.load %[[VAL_22]] {tbaa = [#[[$BOXT]]]} : !llvm.ptr<i64>
// CHECK:           %[[VAL_24:.*]] = llvm.getelementptr %[[VAL_3]][0, 4] : (!llvm.ptr<struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>>) -> !llvm.ptr<i32>
// CHECK:           %[[VAL_25:.*]] = llvm.load %[[VAL_24]] {tbaa = [#[[$BOXT]]]} : !llvm.ptr<i32>
// CHECK:           %[[VAL_26:.*]] = llvm.getelementptr %[[VAL_3]][0, 8] : (!llvm.ptr<struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>>) -> !llvm.ptr<ptr<i8>>
// CHECK:           %[[VAL_27:.*]] = llvm.load %[[VAL_26]] {tbaa = [#[[$BOXT]]]} : !llvm.ptr<ptr<i8>>
// CHECK:           %[[VAL_28:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>
// CHECK:           %[[VAL_29:.*]] = llvm.insertvalue %[[VAL_23]], %[[VAL_28]][1] : !llvm.struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>
// CHECK:           %[[VAL_30:.*]] = llvm.mlir.constant(20180515 : i32) : i32
// CHECK:           %[[VAL_31:.*]] = llvm.insertvalue %[[VAL_30]], %[[VAL_29]][2] : !llvm.struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>
// CHECK:           %[[VAL_32:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_33:.*]] = llvm.trunc %[[VAL_32]] : i32 to i8
// CHECK:           %[[VAL_34:.*]] = llvm.insertvalue %[[VAL_33]], %[[VAL_31]][3] : !llvm.struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>
// CHECK:           %[[VAL_35:.*]] = llvm.trunc %[[VAL_25]] : i32 to i8
// CHECK:           %[[VAL_36:.*]] = llvm.insertvalue %[[VAL_35]], %[[VAL_34]][4] : !llvm.struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>
// CHECK:           %[[VAL_37:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_38:.*]] = llvm.trunc %[[VAL_37]] : i32 to i8
// CHECK:           %[[VAL_39:.*]] = llvm.insertvalue %[[VAL_38]], %[[VAL_36]][5] : !llvm.struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>
// CHECK:           %[[VAL_40:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_41:.*]] = llvm.trunc %[[VAL_40]] : i32 to i8
// CHECK:           %[[VAL_42:.*]] = llvm.insertvalue %[[VAL_41]], %[[VAL_39]][6] : !llvm.struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>
// CHECK:           %[[VAL_43:.*]] = llvm.bitcast %[[VAL_27]] : !llvm.ptr<i8> to !llvm.ptr<i8>
// CHECK:           %[[VAL_44:.*]] = llvm.insertvalue %[[VAL_43]], %[[VAL_42]][8] : !llvm.struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>
// CHECK:           %[[VAL_45:.*]] = llvm.getelementptr %[[VAL_3]][0, 7, 0, 0] : (!llvm.ptr<struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>>) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_46:.*]] = llvm.load %[[VAL_45]] {tbaa = [#[[$BOXT]]]} : !llvm.ptr<i64>
// CHECK:           %[[VAL_47:.*]] = llvm.getelementptr %[[VAL_3]][0, 7, 0, 1] : (!llvm.ptr<struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>>) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_48:.*]] = llvm.load %[[VAL_47]] {tbaa = [#[[$BOXT]]]} : !llvm.ptr<i64>
// CHECK:           %[[VAL_49:.*]] = llvm.getelementptr %[[VAL_3]][0, 7, 0, 2] : (!llvm.ptr<struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>>) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_50:.*]] = llvm.load %[[VAL_49]] {tbaa = [#[[$BOXT]]]} : !llvm.ptr<i64>
// CHECK:           %[[VAL_51:.*]] = llvm.getelementptr %[[VAL_3]][0, 0] : (!llvm.ptr<struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>>) -> !llvm.ptr<ptr<struct<()>>>
// CHECK:           %[[VAL_52:.*]] = llvm.load %[[VAL_51]] {tbaa = [#[[$BOXT]]]} : !llvm.ptr<ptr<struct<()>>>
// CHECK:           %[[VAL_53:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:           %[[VAL_54:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           %[[VAL_55:.*]] = llvm.icmp "eq" %[[VAL_48]], %[[VAL_53]] : i64
// CHECK:           %[[VAL_56:.*]] = llvm.select %[[VAL_55]], %[[VAL_54]], %[[VAL_13]] : i1, i64
// CHECK:           %[[VAL_57:.*]] = llvm.insertvalue %[[VAL_56]], %[[VAL_44]][7, 0, 0] : !llvm.struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>
// CHECK:           %[[VAL_58:.*]] = llvm.insertvalue %[[VAL_48]], %[[VAL_57]][7, 0, 1] : !llvm.struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>
// CHECK:           %[[VAL_59:.*]] = llvm.insertvalue %[[VAL_50]], %[[VAL_58]][7, 0, 2] : !llvm.struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>
// CHECK:           %[[VAL_60:.*]] = llvm.bitcast %[[VAL_52]] : !llvm.ptr<struct<()>> to !llvm.ptr<struct<()>>
// CHECK:           %[[VAL_61:.*]] = llvm.insertvalue %[[VAL_60]], %[[VAL_59]][0] : !llvm.struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>
// CHECK:           llvm.store %[[VAL_61]], %[[VAL_1]] {tbaa = [#[[$BOXT]]]} : !llvm.ptr<struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>>
// CHECK:           %[[VAL_62:.*]] = llvm.bitcast %[[VAL_1]] : !llvm.ptr<struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>> to !llvm.ptr<struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, ptr<i8>, array<1 x i64>)>>
// CHECK:           %[[VAL_63:.*]] = llvm.call @_FortranAioOutputDescriptor(%[[VAL_10]], %[[VAL_62]]) {fastmathFlags = #llvm.fastmath<contract>} : (!llvm.ptr<i8>, !llvm.ptr<struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, ptr<i8>, array<1 x i64>)>>) -> i1
// CHECK:           %[[VAL_64:.*]] = llvm.call @_FortranAioEndIoStatement(%[[VAL_10]]) {fastmathFlags = #llvm.fastmath<contract>} : (!llvm.ptr<i8>) -> i32
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @_FortranAioBeginExternalListOutput(i32, !llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {fir.io, fir.runtime, sym_visibility = "private"}
// CHECK:         llvm.func @_FortranAioOutputDescriptor(!llvm.ptr<i8>, !llvm.ptr<struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, ptr<i8>, array<1 x i64>)>>) -> i1 attributes {fir.io, fir.runtime, sym_visibility = "private"}
// CHECK:         llvm.func @_FortranAioEndIoStatement(!llvm.ptr<i8>) -> i32 attributes {fir.io, fir.runtime, sym_visibility = "private"}

// CHECK-LABEL:   llvm.mlir.global linkonce constant @_QQcl.2E2F64756D6D792E66393000() comdat(@__llvm_comdat::@_QQcl.2E2F64756D6D792E66393000) {addr_space = 0 : i32} : !llvm.array<12 x i8> {
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant("./dummy.f90\00") : !llvm.array<12 x i8>
// CHECK:           llvm.return %[[VAL_0]] : !llvm.array<12 x i8>
// CHECK:         }

// CHECK-LABEL:   llvm.mlir.global internal @_QFEx() {addr_space = 0 : i32} : !llvm.struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)> {
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.zero : !llvm.ptr<struct<()>>
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.constant(-1 : i32) : i32
// CHECK:           %[[VAL_4:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>
// CHECK:           %[[VAL_5:.*]] = llvm.insertvalue %[[VAL_2]], %[[VAL_4]][1] : !llvm.struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>
// CHECK:           %[[VAL_6:.*]] = llvm.mlir.constant(20180515 : i32) : i32
// CHECK:           %[[VAL_7:.*]] = llvm.insertvalue %[[VAL_6]], %[[VAL_5]][2] : !llvm.struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>
// CHECK:           %[[VAL_8:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_9:.*]] = llvm.trunc %[[VAL_8]] : i32 to i8
// CHECK:           %[[VAL_10:.*]] = llvm.insertvalue %[[VAL_9]], %[[VAL_7]][3] : !llvm.struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>
// CHECK:           %[[VAL_11:.*]] = llvm.trunc %[[VAL_3]] : i32 to i8
// CHECK:           %[[VAL_12:.*]] = llvm.insertvalue %[[VAL_11]], %[[VAL_10]][4] : !llvm.struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>
// CHECK:           %[[VAL_13:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_14:.*]] = llvm.trunc %[[VAL_13]] : i32 to i8
// CHECK:           %[[VAL_15:.*]] = llvm.insertvalue %[[VAL_14]], %[[VAL_12]][5] : !llvm.struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>
// CHECK:           %[[VAL_16:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_17:.*]] = llvm.trunc %[[VAL_16]] : i32 to i8
// CHECK:           %[[VAL_18:.*]] = llvm.insertvalue %[[VAL_17]], %[[VAL_15]][6] : !llvm.struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>
// CHECK:           %[[VAL_19:.*]] = llvm.mlir.zero : !llvm.ptr<i8>
// CHECK:           %[[VAL_20:.*]] = llvm.bitcast %[[VAL_19]] : !llvm.ptr<i8> to !llvm.ptr<i8>
// CHECK:           %[[VAL_21:.*]] = llvm.insertvalue %[[VAL_20]], %[[VAL_18]][8] : !llvm.struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>
// CHECK:           %[[VAL_22:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:           %[[VAL_23:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           %[[VAL_24:.*]] = llvm.insertvalue %[[VAL_23]], %[[VAL_21]][7, 0, 0] : !llvm.struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>
// CHECK:           %[[VAL_25:.*]] = llvm.insertvalue %[[VAL_0]], %[[VAL_24]][7, 0, 1] : !llvm.struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>
// CHECK:           %[[VAL_26:.*]] = llvm.insertvalue %[[VAL_2]], %[[VAL_25]][7, 0, 2] : !llvm.struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>
// CHECK:           %[[VAL_27:.*]] = llvm.mul %[[VAL_2]], %[[VAL_0]]  : i64
// CHECK:           %[[VAL_28:.*]] = llvm.mul %[[VAL_23]], %[[VAL_0]]  : i64
// CHECK:           %[[VAL_29:.*]] = llvm.bitcast %[[VAL_1]] : !llvm.ptr<struct<()>> to !llvm.ptr<struct<()>>
// CHECK:           %[[VAL_30:.*]] = llvm.insertvalue %[[VAL_29]], %[[VAL_26]][0] : !llvm.struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>
// CHECK:           llvm.return %[[VAL_30]] : !llvm.struct<(ptr<struct<()>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>
// CHECK:         }

// -----

func.func @tbaa(%arg0: !fir.box<!fir.array<*:f64>>) -> i32 {
  %0 = fir.box_rank %arg0 : (!fir.box<!fir.array<*:f64>>) -> i32
  return %0 : i32
}

// CHECK-DAG:     #[[ROOT:.*]] = #llvm.tbaa_root<id = "Flang Type TBAA Root">
// CHECK-DAG:     #[[ANYACC:.*]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[ROOT]], 0>}>
// CHECK-DAG:     #[[BOXMEM:.*]] = #llvm.tbaa_type_desc<id = "descriptor member", members = {<#[[ANYACC]], 0>}>
// CHECK-DAG:     #[[$BOXT:.*]] = #llvm.tbaa_tag<base_type = #[[BOXMEM]], access_type = #[[BOXMEM]], offset = 0>

// CHECK-LABEL:   llvm.func @tbaa(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<struct<(ptr<f64>, i64, i32, i8, i8, i8, i8)>>) -> i32 {
// CHECK:           %[[VAL_1:.*]] = llvm.getelementptr %[[VAL_0]][0, 3] : (!llvm.ptr<struct<(ptr<f64>, i64, i32, i8, i8, i8, i8)>>) -> !llvm.ptr<i32>
// CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] {tbaa = [#[[$BOXT]]]} : !llvm.ptr<i32>
// CHECK:           llvm.return %[[VAL_2]] : i32
// CHECK:         }

// -----

func.func @tbaa(%arg0: !fir.box<!fir.array<*:f64>>) -> i1 {
  %0 = fir.box_isarray %arg0 : (!fir.box<!fir.array<*:f64>>) -> i1
  return %0 : i1
}

// CHECK-DAG:     #[[ROOT:.*]] = #llvm.tbaa_root<id = "Flang Type TBAA Root">
// CHECK-DAG:     #[[ANYACC:.*]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[ROOT]], 0>}>
// CHECK-DAG:     #[[BOXMEM:.*]] = #llvm.tbaa_type_desc<id = "descriptor member", members = {<#[[ANYACC]], 0>}>
// CHECK-DAG:     #[[$BOXT:.*]] = #llvm.tbaa_tag<base_type = #[[BOXMEM]], access_type = #[[BOXMEM]], offset = 0>

// CHECK-LABEL:   llvm.func @tbaa(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<struct<(ptr<f64>, i64, i32, i8, i8, i8, i8)>>) -> i1 {
// CHECK:           %[[VAL_1:.*]] = llvm.getelementptr %[[VAL_0]][0, 3] : (!llvm.ptr<struct<(ptr<f64>, i64, i32, i8, i8, i8, i8)>>) -> !llvm.ptr<i32>
// CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] {tbaa = [#[[$BOXT]]]} : !llvm.ptr<i32>
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_4:.*]] = llvm.icmp "ne" %[[VAL_2]], %[[VAL_3]] : i32
// CHECK:           llvm.return %[[VAL_4]] : i1
// CHECK:         }

// -----

func.func @tbaa(%arg0: !fir.box<f32>) -> i32 {
  %0 = fir.box_elesize %arg0 : (!fir.box<f32>) -> i32
  return %0 : i32
}

// CHECK-DAG:     #[[ROOT:.*]] = #llvm.tbaa_root<id = "Flang Type TBAA Root">
// CHECK-DAG:     #[[ANYACC:.*]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[ROOT]], 0>}>
// CHECK-DAG:     #[[BOXMEM:.*]] = #llvm.tbaa_type_desc<id = "descriptor member", members = {<#[[ANYACC]], 0>}>
// CHECK-DAG:     #[[$BOXT:.*]] = #llvm.tbaa_tag<base_type = #[[BOXMEM]], access_type = #[[BOXMEM]], offset = 0>

// CHECK-LABEL:   llvm.func @tbaa(
// CHECK-SAME:                               %[[VAL_0:.*]]: !llvm.ptr<struct<(ptr<f32>, i64, i32, i8, i8, i8, i8)>>) -> i32 {
// CHECK:           %[[VAL_1:.*]] = llvm.getelementptr %[[VAL_0]][0, 1] : (!llvm.ptr<struct<(ptr<f32>, i64, i32, i8, i8, i8, i8)>>) -> !llvm.ptr<i32>
// CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] {tbaa = [#[[$BOXT]]]} : !llvm.ptr<i32>
// CHECK:           llvm.return %[[VAL_2]] : i32
// CHECK:         }

// -----

func.func @tbaa(%arg0: !fir.box<!fir.array<*:f64>>) -> i1 {
  %0 = fir.box_isalloc %arg0 : (!fir.box<!fir.array<*:f64>>) -> i1
  return %0 : i1
}

// CHECK-DAG:     #[[ROOT:.*]] = #llvm.tbaa_root<id = "Flang Type TBAA Root">
// CHECK-DAG:     #[[ANYACC:.*]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[ROOT]], 0>}>
// CHECK-DAG:     #[[BOXMEM:.*]] = #llvm.tbaa_type_desc<id = "descriptor member", members = {<#[[ANYACC]], 0>}>
// CHECK-DAG:     #[[$BOXT:.*]] = #llvm.tbaa_tag<base_type = #[[BOXMEM]], access_type = #[[BOXMEM]], offset = 0>

// CHECK-LABEL:   llvm.func @tbaa(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<struct<(ptr<f64>, i64, i32, i8, i8, i8, i8)>>) -> i1 {
// CHECK:           %[[VAL_1:.*]] = llvm.getelementptr %[[VAL_0]][0, 5] : (!llvm.ptr<struct<(ptr<f64>, i64, i32, i8, i8, i8, i8)>>) -> !llvm.ptr<i32>
// CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] {tbaa = [#[[$BOXT]]]} : !llvm.ptr<i32>
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:           %[[VAL_4:.*]] = llvm.and %[[VAL_2]], %[[VAL_3]]  : i32
// CHECK:           %[[VAL_5:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_6:.*]] = llvm.icmp "ne" %[[VAL_4]], %[[VAL_5]] : i32
// CHECK:           llvm.return %[[VAL_6]] : i1
// CHECK:         }

// -----

func.func @tbaa(%arg0: !fir.box<!fir.array<?xi32>>) {
  %c0 = arith.constant 0 : i64
  %1 = fircg.ext_array_coor %arg0(%c0) <%c0> : (!fir.box<!fir.array<?xi32>>, i64, i64) -> !fir.ref<i32>
  return
}

// CHECK-DAG:     #[[ROOT:.*]] = #llvm.tbaa_root<id = "Flang Type TBAA Root">
// CHECK-DAG:     #[[ANYACC:.*]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[ROOT]], 0>}>
// CHECK-DAG:     #[[BOXMEM:.*]] = #llvm.tbaa_type_desc<id = "descriptor member", members = {<#[[ANYACC]], 0>}>
// CHECK-DAG:     #[[$BOXT:.*]] = #llvm.tbaa_tag<base_type = #[[BOXMEM]], access_type = #[[BOXMEM]], offset = 0>

// CHECK-LABEL:   llvm.func @tbaa(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<struct<(ptr<i32>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>>) {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:           %[[VAL_4:.*]] = llvm.sub %[[VAL_1]], %[[VAL_2]]  : i64
// CHECK:           %[[VAL_5:.*]] = llvm.mul %[[VAL_4]], %[[VAL_2]]  : i64
// CHECK:           %[[VAL_6:.*]] = llvm.getelementptr %[[VAL_0]][0, 7, 0, 2] : (!llvm.ptr<struct<(ptr<i32>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>>) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_7:.*]] = llvm.load %[[VAL_6]] {tbaa = [#[[$BOXT]]]} : !llvm.ptr<i64>
// CHECK:           %[[VAL_8:.*]] = llvm.mul %[[VAL_5]], %[[VAL_7]]  : i64
// CHECK:           %[[VAL_9:.*]] = llvm.add %[[VAL_8]], %[[VAL_3]]  : i64
// CHECK:           %[[VAL_10:.*]] = llvm.getelementptr %[[VAL_0]][0, 0] : (!llvm.ptr<struct<(ptr<i32>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>>) -> !llvm.ptr<ptr<i32>>
// CHECK:           %[[VAL_11:.*]] = llvm.load %[[VAL_10]] {tbaa = [#[[$BOXT]]]} : !llvm.ptr<ptr<i32>>
// CHECK:           %[[VAL_12:.*]] = llvm.bitcast %[[VAL_11]] : !llvm.ptr<i32> to !llvm.ptr<i8>
// CHECK:           %[[VAL_13:.*]] = llvm.getelementptr %[[VAL_12]]{{\[}}%[[VAL_9]]] : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_14:.*]] = llvm.bitcast %[[VAL_13]] : !llvm.ptr<i8> to !llvm.ptr<i32>
// CHECK:           llvm.return
// CHECK:         }

// -----

// Check that the scalar aggregate load/store with a descriptor member
// is mapped to any-access.
// CHECK-DAG:     #[[ROOT:.*]] = #llvm.tbaa_root<id = "Flang Type TBAA Root">
// CHECK-DAG:     #[[ANYACC:.*]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[ROOT]], 0>}>
// CHECK-DAG:     #[[$ANYT:.*]] = #llvm.tbaa_tag<base_type = #[[ANYACC]], access_type = #[[ANYACC]], offset = 0>

func.func @tbaa(%arg0: !fir.ref<!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>, %arg1: !fir.ref<!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>) {
  %0 = fir.load %arg0 : !fir.ref<!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>
  fir.store %0 to %arg1 : !fir.ref<!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>
  return
}
// CHECK-LABEL:   llvm.func @tbaa(
// CHECK: llvm.load{{.*}}{tbaa = [#[[$ANYT]]]}
// CHECK: llvm.store{{.*}}{tbaa = [#[[$ANYT]]]}

// -----

// Check that the array aggregate load/store with a descriptor member
// is mapped to any-access.
// CHECK-DAG:     #[[ROOT:.*]] = #llvm.tbaa_root<id = "Flang Type TBAA Root">
// CHECK-DAG:     #[[ANYACC:.*]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[ROOT]], 0>}>
// CHECK-DAG:     #[[$ANYT:.*]] = #llvm.tbaa_tag<base_type = #[[ANYACC]], access_type = #[[ANYACC]], offset = 0>

func.func @tbaa(%arg0: !fir.ref<!fir.array<2x!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>>, %arg1: !fir.ref<!fir.array<2x!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>>) {
  %0 = fir.load %arg0 : !fir.ref<!fir.array<2x!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>>
  fir.store %0 to %arg1 : !fir.ref<!fir.array<2x!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>>
  return
}
// CHECK-LABEL:   llvm.func @tbaa(
// CHECK: llvm.load{{.*}}{tbaa = [#[[$ANYT]]]}
// CHECK: llvm.store{{.*}}{tbaa = [#[[$ANYT]]]}
