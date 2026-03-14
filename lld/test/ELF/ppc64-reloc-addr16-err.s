# REQUIRES: ppc
# RUN: llvm-mc -filetype=obj -triple=ppc64le --defsym HI=1 %s -o %thi.o
# RUN: ld.lld %thi.o --defsym=a=0x7fffffff -o /dev/null
# RUN: not ld.lld %thi.o --defsym=a=0x80000000 -o /dev/null
# RUN: ld.lld %thi.o --defsym=a=0xffffffff80000000 -o /dev/null
# RUN: not ld.lld %thi.o --defsym=a=0xffffffff7fffffff -o /dev/null

# RUN: llvm-mc -filetype=obj -triple=ppc64le --defsym HA=1 %s -o %tha.o
# RUN: ld.lld %tha.o --defsym=a=0x7fff7fff -o /dev/null
# RUN: not ld.lld %tha.o --defsym=a=0x7fff8000 -o /dev/null
# RUN: ld.lld %tha.o --defsym=a=0xffffffff7fff8000 -o /dev/null
# RUN: not ld.lld %tha.o --defsym=a=0xffffffff7fff7fff -o /dev/null

.ifdef HI
lis 4, a@h  # R_PPC64_ADDR16_HI
.endif

.ifdef HA
lis 4, a@ha  # R_PPC64_ADDR16_HA
.endif

lis 4, a@high  # R_PPC64_ADDR16_HIGH
