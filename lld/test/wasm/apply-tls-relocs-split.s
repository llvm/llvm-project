## __wasm_apply_tls_relocs is split across helper functions in the same way as
## __wasm_apply_data_relocs when its body would exceed the maximum function
## body size.

# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s

## Each of the five relocations below turns into 13 bytes of code; 28 bytes
## hold two per function (1 + 2 * 13 + 1) but not three.

# RUN: wasm-ld -pie -no-gc-sections --shared-memory --no-entry -o %t.wasm %t.o
# RUN: obj2yaml %t.wasm | FileCheck %s --check-prefix=ONE

# RUN: wasm-ld -pie -no-gc-sections --shared-memory --no-entry --max-function-body-size=28 -o %t.28.wasm %t.o
# RUN: obj2yaml %t.28.wasm | FileCheck %s --check-prefix=SPLIT

.section .tdata,"T",@
.globl tls_sym
.p2align 2
tls_sym:
  .int32 tls_sym
  .int32 tls_sym
  .int32 tls_sym
  .int32 tls_sym
  .int32 tls_sym
.size tls_sym, 20

.section .custom_section.target_features,"",@
  .int8 2
  .int8 43
  .int8 7
  .ascii "atomics"
  .int8 43
  .int8 11
  .ascii "bulk-memory"

# ONE:          FunctionNames:
# ONE:              Name:            __wasm_apply_tls_relocs
# ONE-NOT:          __wasm_apply_tls_relocs_

## The helpers are not exported, and __wasm_apply_tls_relocs only calls them.
# SPLIT:        - Type:            EXPORT
# SPLIT-NOT:        __wasm_apply_tls_relocs
# SPLIT:        - Type:            CODE
# SPLIT:            - Index:           3
# SPLIT-NEXT:         Locals:          []
# SPLIT-NEXT:         Body:            1004100510060B
# SPLIT-NEXT:       - Index:           4
# SPLIT-NEXT:         Locals:          []
# SPLIT-NEXT:         Body:            410023036A230341006A360200410423036A230341006A3602000B
# SPLIT-NEXT:       - Index:           5
# SPLIT-NEXT:         Locals:          []
# SPLIT-NEXT:         Body:            410823036A230341006A360200410C23036A230341006A3602000B
# SPLIT-NEXT:       - Index:           6
# SPLIT-NEXT:         Locals:          []
# SPLIT-NEXT:         Body:            411023036A230341006A3602000B
# SPLIT:            FunctionNames:
# SPLIT:                Name:            __wasm_apply_tls_relocs
# SPLIT-NEXT:         - Index:           4
# SPLIT-NEXT:           Name:            __wasm_apply_tls_relocs_0
# SPLIT-NEXT:         - Index:           5
# SPLIT-NEXT:           Name:            __wasm_apply_tls_relocs_1
# SPLIT-NEXT:         - Index:           6
# SPLIT-NEXT:           Name:            __wasm_apply_tls_relocs_2
# SPLIT-NEXT:       GlobalNames:
