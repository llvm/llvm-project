llvm.func @diamond_cond_br(%x: i32, %y: i32, %z: i32) -> i32 {
  %zero = llvm.mlir.constant(0 : i32) : i32
  %cond = llvm.icmp "sgt" %x, %zero : i32

  llvm.cond_br %cond, ^then_bb, ^else_bb

^then_bb:
  llvm.br ^join(%y : i32)

^else_bb:
  llvm.br ^join(%z : i32)

^join(%result : i32):
  llvm.return %result : i32
}