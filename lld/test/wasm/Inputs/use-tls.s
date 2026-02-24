  .globl  get_tls1
get_tls1:
  .functype get_tls1 () -> (i32)
  i32.const tls1@TLSREL
  end_function

.section  .custom_section.target_features,"",@
  .int8 3
  .int8 43
  .int8 7
  .ascii  "atomics"
  .int8 43
  .int8 11
  .ascii  "bulk-memory"
  .int8 43
  .int8 15
  .ascii "mutable-globals"
