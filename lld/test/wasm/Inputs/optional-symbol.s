# __dso_handle is an linker-generated symbol that is included only when needed.

  .globl  get_optional
get_optional:
  .functype get_optional () -> (i32)
  i32.const __dso_handle
  end_function
