# Verify that --gc-sections preserves functions reachable only through the
# thread-context accessors __wasm_{get,set}_tls_base. These accessors are
# invoked by synthetic init functions (e.g. __wasm_init_tls) via `call`
# instructions that carry no relocations, so the linker marks the accessors
# live explicitly. Their own relocations must still be followed during GC,
# otherwise functions they call (here modeling the cooperative-threading
# context.set builtin) are incorrectly collected and the call is mis-resolved.

# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: wasm-ld --cooperative-threading --gc-sections -o %t.wasm %t.o
# RUN: obj2yaml %t.wasm | FileCheck %s

    .functype set_helper (i32) -> ()
    .import_module set_helper, "env"
    .import_name set_helper, "set_helper"

.globl __wasm_get_tls_base
__wasm_get_tls_base:
    .functype __wasm_get_tls_base () -> (i32)
    i32.const 0
    end_function

# Reachable only via the synthetic __wasm_init_tls. Its call to `set_helper`
# must keep `set_helper` live.
.globl __wasm_set_tls_base
__wasm_set_tls_base:
    .functype __wasm_set_tls_base (i32) -> ()
    local.get 0
    call set_helper
    end_function

.globl _start
_start:
    .functype _start () -> (i32)
    call __wasm_get_tls_base
    i32.const tls1@TLSREL
    i32.add
    i32.load 0
    end_function

.section  .tdata.tls1,"",@
.globl  tls1
tls1:
    .int32  1
    .size tls1, 4

.section  .custom_section.target_features,"",@
    .int8 2
    .int8 43
    .int8 11
    .ascii  "bulk-memory"
    .int8 43
    .int8 7
    .ascii  "atomics"

# The imported helper called by __wasm_set_tls_base must survive GC.
# CHECK: Field:           set_helper
