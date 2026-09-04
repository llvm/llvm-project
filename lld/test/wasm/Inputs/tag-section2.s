.tagtype __cpp_exception i32

.globl bar
bar:
  .functype bar (i32) -> ()
  local.get 0
  throw __cpp_exception
  end_function
