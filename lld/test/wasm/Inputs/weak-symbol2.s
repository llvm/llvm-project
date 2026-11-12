  .weak weakFn
weakFn:
  .functype weakFn () -> (i32)
  i32.const 2
  end_function

  .globl  exportWeak2
exportWeak2:
  .functype exportWeak2 () -> (i32)
  i32.const weakFn
  end_function

  .section  .data.weakGlobal,"",@
  .weak weakGlobal
weakGlobal:
  .int32  2
  .size weakGlobal, 4
