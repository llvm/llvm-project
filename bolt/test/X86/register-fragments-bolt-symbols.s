# Test the heuristics for matching BOLT-added split functions.

# RUN: llvm-mc --filetype=obj --triple x86_64-unknown-unknown %S/cdsplit-symbol-names.s -o %t.main.o
# RUN: llvm-mc --filetype=obj --triple x86_64-unknown-unknown %s -o %t.chain.o
# RUN: link_fdata %S/cdsplit-symbol-names.s %t.main.o %t.fdata
# RUN: sed -i 's|chain|chain/2|g' %t.fdata
# RUN: llvm-strip --strip-unneeded %t.main.o
# RUN: llvm-objcopy --localize-symbol=chain %t.main.o
# RUN: %clang %cflags %t.chain.o %t.main.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --split-functions --split-strategy=randomN \
# RUN:   --reorder-blocks=ext-tsp --enable-bat --bolt-seed=7 --data=%t.fdata
# RUN: llvm-objdump --syms %t.bolt | FileCheck %s --check-prefix=CHECK-SYMS

# RUN: link_fdata %s %t.bolt %t.preagg PREAGG
# PREAGG: B X:0 #chain.cold.0# 1 0
# RUN: perf2bolt %t.bolt -p %t.preagg --pa -o %t.bat.fdata -w %t.bat.yaml -v=1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-REGISTER

# CHECK-SYMS: l df *ABS*          [[#]] chain.s
# CHECK-SYMS: l  F .bolt.org.text [[#]] chain
# CHECK-SYMS: l  F .text.cold     [[#]] chain.cold.0
# CHECK-SYMS: l  F .text          [[#]] chain
# CHECK-SYMS: l df *ABS*          [[#]] bolt-pseudo.o

# CHECK-REGISTER: BOLT-INFO: marking chain.cold.0/1(*2) as a fragment of chain/2(*2)

.file "chain.s"
        .text
        .type   chain, @function
chain:
        ret
        .size   chain, .-chain
