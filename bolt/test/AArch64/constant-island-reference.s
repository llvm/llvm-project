// Test BOLT won't ignore function having reference to constant island
// that is defined in another function.

// RUN: %clang %cxxflags -fuse-ld=lld %s -o %t.so -Wl,-q -no-pie
// RUN: llvm-bolt %t.so -o %t.bolt.so --strict

  .text
  .type foo,@function
foo:
  adrp x9, :got:_global_func_ptr
  ldr x9, [x9, #:got_lo12:_global_func_ptr]
  blr x9
  .size foo, .-foo

  .type bar,@function
bar:
  nop
_global_func_ptr:
  .quad 0
  .size bar, .-bar
