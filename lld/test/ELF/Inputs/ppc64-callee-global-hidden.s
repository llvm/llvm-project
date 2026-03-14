func_extern:
  blr

.hidden callee3_stother0_hidden
.globl callee3_stother0_hidden
callee3_stother0_hidden:
  blr

.hidden callee4_stother1_hidden
.globl callee4_stother1_hidden
callee4_stother1_hidden:
  .localentry callee4_stother1_hidden, 1
  ## nop is not needed after bl for R_PPC64_REL24_NOTOC
  bl func_extern@notoc
  blr
