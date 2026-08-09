.functype foo () -> ()
.import_name foo, foo_import

.globl _start
_start:
    .functype _start () -> ()
    i32.const __heap_base@GOT
    drop
    call foo
    end_function
