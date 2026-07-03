# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=i686-windows %s -o %t.obj
# RUN: llvm-lib -machine:x86 -out:%t-func.lib %t.obj

# -export:func creates a weak alias to a lazy symbol. Make sure we can handle that when processing -export:func2=func.
# RUN: lld-link -dll -noentry -machine:x86 -out:%t.dll %t-func.lib -export:func -export:func2=func

        .text
        .def    @feat.00;
        .scl    3;
        .type   0;
        .endef
        .globl  @feat.00
.set @feat.00, 1
        .globl _func@0
_func@0:
        retl
