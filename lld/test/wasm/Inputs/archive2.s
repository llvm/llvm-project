.functype foo () -> (i32)

.globl  bar
bar:
  .functype bar () -> (i32)
  call foo
  end_function

.globl archive2_symbol
archive2_symbol:
  .functype archive2_symbol () -> ()
  end_function
