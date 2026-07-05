! RUN: %flang -E %s 2>&1 | FileCheck %s
! CHECK: n = "foobar" // "baz"
! # stringizing works in FLM
#define STR(x) # x
#define MAC(a,b) STR(a ## b)
program main
  character(6) n
  n = MAC(foo, bar) // STR( baz )
end
