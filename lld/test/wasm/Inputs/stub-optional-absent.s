.functype foo () -> ()
.import_name foo, foo_import

.globl _start
_start:
    .functype _start () -> ()
    call foo
    end_function
