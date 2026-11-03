// RUN: inter-opt %s | inter-opt | FileCheck %s --check-prefix=ROUNDTRIP
// RUN: inter-alias-dump %s | FileCheck %s --check-prefix=ALIAS

// ALIAS: xemachine.tuple_from_elements offset=0 destructive=0
// ALIAS-NEXT: xemachine.tuple_from_elements offset=32 destructive=0
// ALIAS-NEXT: xemachine.tuple_to_elements offset=0 destructive=0
// ALIAS-NEXT: xemachine.tuple_to_elements offset=32 destructive=0
// ALIAS-NEXT: xemachine.update_tuple offset=0 destructive=1
// ALIAS-NEXT: xemachine.update_tuple offset=0 destructive=1
// ALIAS-NEXT: xemachine.update_tuple offset=32 destructive=1

// ROUNDTRIP-LABEL: func.func @tuple_aliases
// ROUNDTRIP: [[TUPLE:%.*]] = xemachine.tuple_from_elements %arg0, %arg1
// ROUNDTRIP: [[PARTS:%.*]]:2 = xemachine.tuple_to_elements [[TUPLE]]
// ROUNDTRIP: xemachine.update_tuple [[TUPLE]], [[PARTS]]#0, [[PARTS]]#1 {offsets = [0, 32]}
func.func @tuple_aliases(%lo: !xemachine.reg<32, -1>,
                         %hi: !xemachine.reg<16, -1>)
    -> !xemachine.reg<48, -1> {
  %tuple = xemachine.tuple_from_elements %lo, %hi
      : (!xemachine.reg<32, -1>, !xemachine.reg<16, -1>)
        -> !xemachine.reg<48, -1>
  %parts:2 = xemachine.tuple_to_elements %tuple
      : (!xemachine.reg<48, -1>)
        -> (!xemachine.reg<32, -1>, !xemachine.reg<16, -1>)
  %updated = xemachine.update_tuple %tuple, %parts#0, %parts#1
      {offsets = [0, 32]}
      : (!xemachine.reg<48, -1>, !xemachine.reg<32, -1>,
         !xemachine.reg<16, -1>) -> !xemachine.reg<48, -1>
  return %updated : !xemachine.reg<48, -1>
}

// ROUNDTRIP-LABEL: func.func @fold_join_split
func.func @fold_join_split(%lo: !xemachine.reg<32, -1>,
                           %hi: !xemachine.reg<32, -1>)
    -> (!xemachine.reg<32, -1>, !xemachine.reg<32, -1>) {
  %tuple = xemachine.tuple_from_elements %lo, %hi
      : (!xemachine.reg<32, -1>, !xemachine.reg<32, -1>)
        -> !xemachine.reg<64, -1>
  %parts:2 = xemachine.tuple_to_elements %tuple
      : (!xemachine.reg<64, -1>)
        -> (!xemachine.reg<32, -1>, !xemachine.reg<32, -1>)
  return %parts#0, %parts#1
      : !xemachine.reg<32, -1>, !xemachine.reg<32, -1>
}

// ROUNDTRIP-LABEL: func.func @fold_split_join
func.func @fold_split_join(%tuple: !xemachine.reg<64, -1>)
    -> !xemachine.reg<64, -1> {
  %parts:2 = xemachine.tuple_to_elements %tuple
      : (!xemachine.reg<64, -1>)
        -> (!xemachine.reg<32, -1>, !xemachine.reg<32, -1>)
  %rebuilt = xemachine.tuple_from_elements %parts#0, %parts#1
      : (!xemachine.reg<32, -1>, !xemachine.reg<32, -1>)
        -> !xemachine.reg<64, -1>
  return %rebuilt : !xemachine.reg<64, -1>
}
