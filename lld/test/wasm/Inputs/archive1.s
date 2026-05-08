.functype bar () -> (i32)

.globl foo
foo:
  .functype foo () -> (i32)
  call bar
  end_function
