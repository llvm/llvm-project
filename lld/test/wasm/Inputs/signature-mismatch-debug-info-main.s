  .functype  test0 () -> ()
  .functype  test1 () -> ()
  .functype  main (i32, i32) -> (i32)

  .section  .text.main,"",@
  .globl  main
  .type  main,@function
main:
  .functype  main (i32, i32) -> (i32)
  call  test0
  call  test1
  i32.const  0
  end_function

  .section  .debug_info,"",@
  .int32 test0
  .int32 test1
