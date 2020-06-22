  .globl  call_foo
call_foo:
  .functype call_foo () -> ()
  call  foo
  end_function

  .functype foo () -> ()
  .import_module  foo, baz
