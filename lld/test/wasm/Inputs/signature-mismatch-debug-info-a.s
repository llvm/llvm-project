  .functype  foo (i32) -> ()
  .functype  test0 () -> ()

  .section  .text.foo,"",@
  .weak  foo
  .type  foo,@function
foo:
  .functype  foo (i32) -> ()
  end_function

  .section  .text.test0,"",@
  .globl  test0
  .type  test0,@function
test0:
  .functype  test0 () -> ()
  i32.const  3
  call  foo
  end_function

  .section  .debug_info,"",@
  .int32 foo
  .int32 test0
