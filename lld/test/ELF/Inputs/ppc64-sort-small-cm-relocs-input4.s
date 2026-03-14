    .text
    .global getRodata
    .type getRodata,@function
getRodata:
.Lgep:
    addis 2, 12, .TOC.-.Lgep@ha
    addi 2, 2, .TOC.-.Lgep@l
.Llep:
    .localentry     getRodata, .Llep-.Lgep
    lwa 3, .LC0@toc(2)
    blr

    .section        .rodata,"a",@progbits
    .quad _start

    .section        .toc,"aw",@progbits
.LC0:
    .tc .rodata[TC], .rodata
