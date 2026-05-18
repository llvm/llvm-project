.functype bar () -> ()

.globl foo
foo:
  .functype foo () -> ()
  call bar
  end_function

# Static constructor inserted here to ensure the object file is not
# being processed as "live".  Live object files have their static constructors
# preserved even if no symbol within is used.
.section .init_array,"",@
  .p2align 2
  .int32 foo
