  .weak weakFn
weakFn:
  .functype weakFn () -> (i32)
  i32.const 1
  end_function

  .globl  exportWeak1
exportWeak1:
  .functype exportWeak1 () -> (i32)
  i32.const weakFn
  end_function

  .section  .data.weakGlobal,"",@
  .weak weakGlobal
weakGlobal:
  .int32  1
  .size weakGlobal, 4
