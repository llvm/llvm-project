.globl foo
foo:
   .functype foo (f32) -> (i32)
   i32.const 0
   end_function


.globl call_foo
call_foo:
   .functype call_foo () -> (i32)
   f32.const 0.0
   call foo
   end_function
