.functype ret32 (f32) -> (i32)

  .globl  call_ret32
call_ret32:
  .functype call_ret32 () -> (i32)
  f32.const 0x0p0
  call  ret32
  drop
  i32.const ret32_address
  end_function

  .section  .data.ret32_address,"",@
  .globl ret32_address
ret32_address:
  .int32  ret32
  .size ret32_address, 4
