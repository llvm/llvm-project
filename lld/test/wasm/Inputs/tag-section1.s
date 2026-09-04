.tagtype __cpp_exception i32
.globl foo

foo:
  .functype foo (i32) -> ()
  local.get 0
  throw __cpp_exception
  end_function
