.globl colliding_func1
colliding_func1:
  .functype colliding_func1 () -> (i32)
  i32.const 2
  end_function

colliding_func2:
  .functype colliding_func2 () -> (i32)
  i32.const 2
  end_function

colliding_func3:
  .functype colliding_func3 () -> (i32)
  i32.const 2
  end_function

.globl get_global1B
get_global1B:
  .functype get_global1B () -> (i32)
  i32.const colliding_global1
  end_function

.globl get_global2B
get_global2B:
  .functype get_global2B () -> (i32)
  i32.const colliding_global2
  end_function

.globl get_global3B
get_global3B:
  .functype get_global3B () -> (i32)
  i32.const colliding_global3
  end_function

.globl get_func1B
get_func1B:
  .functype get_func1B () -> (i32)
  i32.const colliding_func1
  end_function

.globl get_func2B
get_func2B:
  .functype get_func2B () -> (i32)
  i32.const colliding_func2
  end_function

.globl get_func3B
get_func3B:
  .functype get_func3B () -> (i32)
  i32.const colliding_func3
  end_function

.section .data.colliding_global1,"",@
.p2align 2
.globl colliding_global1
colliding_global1:
  .int32 1
  .size colliding_global1, 4

.section .data.colliding_global2,"",@
.p2align 2
colliding_global2:
  .int32 1
  .size colliding_global2, 4

.section .data.colliding_global3,"",@
.p2align 2
colliding_global3:
  .int32 1
  .size colliding_global3, 4
