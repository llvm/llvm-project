.functype bar () -> ()

.globl foo
foo:
  .functype foo () -> ()
  call bar
  end_function
