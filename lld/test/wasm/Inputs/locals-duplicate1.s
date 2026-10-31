colliding_func1:
  .functype colliding_func1 () -> (i32)
  i32.const 2
  end_function

.globl colliding_func2
colliding_func2:
  .functype colliding_func2 () -> (i32)
  i32.const 2
  end_function

colliding_func3:
  .functype colliding_func3 () -> (i32)
  i32.const 2
  end_function

.globl get_global1A
get_global1A:
  .functype get_global1A () -> (i32)
  i32.const colliding_global1
  end_function

.globl get_global2A
get_global2A:
  .functype get_global2A () -> (i32)
  i32.const colliding_global2
  end_function

.globl get_global3A
get_global3A:
  .functype get_global3A () -> (i32)
  i32.const colliding_global3
  end_function

.globl get_func1A
get_func1A:
  .functype get_func1A () -> (i32)
  i32.const colliding_func1
  end_function

.globl get_func2A
get_func2A:
  .functype get_func2A () -> (i32)
  i32.const colliding_func2
  end_function

.globl get_func3A
get_func3A:
  .functype get_func3A () -> (i32)
  i32.const colliding_func3
  end_function

.section .data.colliding_global1,"",@
.p2align 2
colliding_global1:
  .int32 1
  .size colliding_global1, 4

.section .data.colliding_global2,"",@
.p2align 2
.globl colliding_global2
colliding_global2:
  .int32 1
  .size colliding_global2, 4

.section .data.colliding_global3,"",@
.p2align 2
colliding_global3:
  .int32 1
  .size colliding_global3, 4
