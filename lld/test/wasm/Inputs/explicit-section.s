.globl var1
.globl var2

.section mysection,"",@
.p2align 2

var1:
    .int32 42
    .size var1, 4

var2:
    .int32 43
    .size var2, 4
