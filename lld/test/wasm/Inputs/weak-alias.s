.globl direct_fn
direct_fn:
  .functype direct_fn () -> (i32)
  i32.const 0
  end_function

.globl call_direct
call_direct:
  .functype call_direct () -> (i32)
  call direct_fn
  end_function

.functype alias_fn () -> (i32)
.weak alias_fn
alias_fn = direct_fn

.globl call_alias
call_alias:
  .functype call_alias () -> (i32)
  call alias_fn
  end_function

.globl call_alias_ptr
call_alias_ptr:
  .functype call_alias_ptr () -> (i32)
  i32.const alias_fn
  call_indirect () -> (i32)
  end_function

.globl call_direct_ptr
call_direct_ptr:
  .functype call_direct_ptr () -> (i32)
  i32.const direct_fn
  call_indirect () -> (i32)
  end_function
