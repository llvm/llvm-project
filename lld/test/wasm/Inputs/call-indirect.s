  .globl  bar
bar:
  .functype bar () -> (i64)
  i64.const 1
  end_function

  .globl  call_bar_indirect
call_bar_indirect:
  .functype call_bar_indirect () -> ()
  i32.load  indirect_bar
  call_indirect () -> (i64)
  drop
  i32.load  indirect_foo
  call_indirect () -> (i32)
  drop
  end_function

  .section  .data.indirect_bar,"",@
indirect_bar:
  .int32  bar
  .size indirect_bar, 4

  .section  .data.indirect_foo,"",@
indirect_foo:
  .int32  foo
  .size indirect_foo, 4

  .functype foo () -> (i32)
