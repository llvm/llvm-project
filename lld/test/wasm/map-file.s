# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %s -o %t1.o
# RUN: wasm-ld %t1.o -o %t -M | FileCheck --match-full-lines --strict-whitespace %s
# RUN: wasm-ld %t1.o -o %t -print-map | FileCheck --match-full-lines --strict-whitespace %s
# RUN: wasm-ld %t1.o -o %t -Map=%t.map
# RUN: FileCheck --match-full-lines --strict-whitespace %s < %t.map

.globaltype wasm_global, i32, immutable
wasm_global:

bar:
    .functype bar () -> ()
    i32.const   somedata
    end_function

write_global:
    .functype write_global (i32) -> ()
    local.get 0
    global.set wasm_global
    end_function

    .globl _start
_start:
    .functype _start () -> ()
    call bar
    call write_global
    end_function

.section .data.somedata,"",@
somedata:
    .int32 123
.size somedata, 4

.section .debug_info,"",@
    .int32 bar

#      CHECK:    Addr      Off     Size Out     In      Symbol
# CHECK-NEXT:       -        8        a TYPE
# CHECK-NEXT:       -       12        6 FUNCTION
# CHECK-NEXT:       -       18        7 TABLE
# CHECK-NEXT:       -       1f        5 MEMORY
# CHECK-NEXT:       -       24        f GLOBAL
# CHECK-NEXT:       0        0        0         __stack_pointer
# CHECK-NEXT:       1        0        0         wasm_global
# CHECK-NEXT:       -       33       15 EXPORT
# CHECK-NEXT:       -       48       26 CODE
# CHECK-NEXT:       -       49        9         {{.*}}{{/|\\}}map-file.s.tmp1.o:(bar)
# CHECK-NEXT:       -       49        9                 bar
# CHECK-NEXT:       -       52        b         {{.*}}{{/|\\}}map-file.s.tmp1.o:(write_global)
# CHECK-NEXT:       -       52        b                 write_global
# CHECK-NEXT:       -       5d        f         {{.*}}{{/|\\}}map-file.s.tmp1.o:(_start)
# CHECK-NEXT:       -       5d        f                 _start
# CHECK-NEXT:       -       6e        d DATA
# CHECK-NEXT:     400       6f        4 .data
# CHECK-NEXT:     400       75        4         {{.*}}{{/|\\}}map-file.s.tmp1.o:(.data.somedata)
# CHECK-NEXT:     400       75        4                 somedata
# CHECK-NEXT:       -       7b       12 CUSTOM(.debug_info)
# CHECK-NEXT:       -       8d       50 CUSTOM(name)

# RUN: not wasm-ld %t1.o -o /dev/null -Map=/ 2>&1 \
# RUN:  | FileCheck -check-prefix=FAIL %s
# FAIL: wasm-ld: error: cannot open map file /
