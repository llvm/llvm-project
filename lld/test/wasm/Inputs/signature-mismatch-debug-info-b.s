  .functype  foo (i32, i32) -> ()
  .functype  test1 () -> ()

  .section  .text.foo,"",@
  .weak  foo
  .type  foo,@function
foo:
  .functype  foo (i32, i32) -> ()
  end_function

  .section  .text.test1,"",@
  .globl  test1
  .type  test1,@function
test1:
  .functype  test1 () -> ()
  i32.const  4
  i32.const  5
  call  foo
  end_function

  .section  .debug_info,"",@
  .int32 foo
  .int32 test1
