# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %s -o %t1.o
# RUN: wasm-ld %t1.o -o %t -M | FileCheck --match-full-lines --strict-whitespace %s
# RUN: wasm-ld %t1.o -o %t -print-map | FileCheck --match-full-lines --strict-whitespace %s
# RUN: wasm-ld %t1.o -o %t -Map=%t.map
# RUN: FileCheck --match-full-lines --strict-whitespace %s < %t.map

bar:
    .functype bar () -> ()
    i32.const   somedata
    end_function

    .globl _start
_start:
    .functype _start () -> ()
    call bar
    end_function

.section .data.somedata,"",@
somedata:
    .int32 123
.size somedata, 4

.section .debug_info,"",@
    .int32 bar

#      CHECK:    Addr      Off     Size Out     In      Symbol
# CHECK-NEXT:       -        8        6 TYPE
# CHECK-NEXT:       -        e        5 FUNCTION
# CHECK-NEXT:       -       13        7 TABLE
# CHECK-NEXT:       -       1a        5 MEMORY
# CHECK-NEXT:       -       1f        a GLOBAL
# CHECK-NEXT:       -       29       15 EXPORT
# CHECK-NEXT:       -       3e       15 CODE
# CHECK-NEXT:       -       3f        9         {{.*}}{{/|\\}}map-file.s.tmp1.o:(bar)
# CHECK-NEXT:       -       3f        9                 bar
# CHECK-NEXT:       -       48        9         {{.*}}{{/|\\}}map-file.s.tmp1.o:(_start)
# CHECK-NEXT:       -       48        9                 _start
# CHECK-NEXT:       -       53        d DATA
# CHECK-NEXT:     400       54        4 .data
# CHECK-NEXT:     400       5a        4         {{.*}}{{/|\\}}map-file.s.tmp1.o:(.data.somedata)
# CHECK-NEXT:     400       5a        4                 somedata
# CHECK-NEXT:       -       60       12 CUSTOM(.debug_info)
# CHECK-NEXT:       -       72       2b CUSTOM(name)

# RUN: not wasm-ld %t1.o -o /dev/null -Map=/ 2>&1 \
# RUN:  | FileCheck -check-prefix=FAIL %s
# FAIL: wasm-ld: error: cannot open map file /
