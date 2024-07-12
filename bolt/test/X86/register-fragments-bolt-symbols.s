## Test the heuristics for matching BOLT-added split functions.

# RUN: llvm-mc --filetype=obj --triple x86_64-unknown-unknown %S/cdsplit-symbol-names.s -o %t.main.o
# RUN: llvm-mc --filetype=obj --triple x86_64-unknown-unknown %s -o %t.chain.o
# RUN: link_fdata %S/cdsplit-symbol-names.s %t.main.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.main.o

## Check warm fragment name matching (produced by cdsplit)
# RUN: %clang %cflags %t.main.o -o %t.warm.exe -Wl,-q
# RUN: llvm-bolt %t.warm.exe -o %t.warm.bolt --split-functions --split-strategy=cdsplit \
# RUN:   --call-scale=2 --data=%t.fdata --reorder-blocks=ext-tsp --enable-bat
# RUN: link_fdata %s %t.warm.bolt %t.preagg.warm PREAGGWARM
# PREAGGWARM: B X:0 #chain.warm# 1 0
# RUN: perf2bolt %t.warm.bolt -p %t.preagg.warm --pa -o %t.warm.fdata -w %t.warm.yaml \
# RUN:   -v=1 | FileCheck %s --check-prefix=CHECK-BOLT-WARM
# RUN: FileCheck %s --input-file %t.warm.fdata --check-prefix=CHECK-FDATA-WARM
# RUN: FileCheck %s --input-file %t.warm.yaml --check-prefix=CHECK-YAML-WARM

# CHECK-BOLT-WARM: marking chain.warm/1(*2) as a fragment of chain
# CHECK-FDATA-WARM: chain
# CHECK-YAML-WARM: chain

# RUN: sed -i 's|chain|chain/2|g' %t.fdata
# RUN: llvm-objcopy --localize-symbol=chain %t.main.o
# RUN: %clang %cflags %t.chain.o %t.main.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --split-functions --split-strategy=randomN \
# RUN:   --reorder-blocks=ext-tsp --enable-bat --bolt-seed=7 --data=%t.fdata
# RUN: llvm-objdump --syms %t.bolt | FileCheck %s --check-prefix=CHECK-SYMS

# RUN: link_fdata %s %t.bolt %t.preagg PREAGG
# PREAGG: B X:0 #chain.cold.0# 1 0
# RUN: perf2bolt %t.bolt -p %t.preagg --pa -o %t.bat.fdata -w %t.bat.yaml -v=1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-REGISTER
# RUN: FileCheck --input-file %t.bat.fdata --check-prefix=CHECK-FDATA %s
# RUN: FileCheck --input-file %t.bat.yaml --check-prefix=CHECK-YAML %s

# RUN: link_fdata --no-redefine %s %t.bolt %t.preagg2 PREAGG2
# PREAGG2: B X:0 #chain# 1 0
# RUN: perf2bolt %t.bolt -p %t.preagg2 --pa -o %t.bat2.fdata -w %t.bat2.yaml
# RUN: FileCheck %s --input-file %t.bat2.yaml --check-prefix=CHECK-YAML2

# CHECK-SYMS: l df *ABS*          [[#]] chain.s
# CHECK-SYMS: l  F .bolt.org.text [[#]] chain
# CHECK-SYMS: l  F .text.cold     [[#]] chain.cold.0
# CHECK-SYMS: l  F .text          [[#]] chain
# CHECK-SYMS: l df *ABS*          [[#]] bolt-pseudo.o

# CHECK-REGISTER: BOLT-INFO: marking chain.cold.0/1(*2) as a fragment of chain/2(*2)

# CHECK-FDATA: 0 [unknown] 0 1 chain/chain.s/2 10 0 1
# CHECK-YAML: - name: 'chain/chain.s/2'
# CHECK-YAML2: - name: 'chain/chain.s/1'
## non-BAT function has non-zero insns:
# CHECK-YAML2: insns: 1

.file "chain.s"
        .text
        .type   chain, @function
chain:
        ret
        .size   chain, .-chain
