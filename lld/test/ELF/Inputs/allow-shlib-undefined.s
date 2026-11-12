.globl _shared
.weak x2
_shared:
  callq x1@PLT

       callq x2@PLT
