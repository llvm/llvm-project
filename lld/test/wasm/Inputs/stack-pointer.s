.globaltype __stack_pointer, i32

.globl  _start
_start:
  .functype _start () -> (i32)
  global.get __stack_pointer
  i32.const 16
  i32.sub
  drop
  i32.const 0
  end_function