.functype foo () -> ()
.import_name foo, foo_import

.globaltype __heap_base, i32, immutable
__heap_base:

.globl _start
_start:
    .functype _start () -> ()
    call foo
    end_function
