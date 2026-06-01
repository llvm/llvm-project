llvm.func @cond_br_demo(%arg0: i32, %arg1: i32, %cond: i1) -> i32 {
  llvm.cond_br %cond, ^true_bb, ^false_bb

^true_bb:
  llvm.return %arg0 : i32

^false_bb:
  llvm.return %arg1 : i32
}
