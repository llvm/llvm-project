# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %s -o %t1.o
# RUN: wasm-ld %t1.o -o %t -M | FileCheck --match-full-lines --strict-whitespace %s
# RUN: wasm-ld %t1.o -o %t -print-map | FileCheck --match-full-lines --strict-whitespace %s
# RUN: wasm-ld %t1.o -o %t -Map=%t.map
# RUN: FileCheck --match-full-lines --strict-whitespace %s < %t.map

.globaltype wasm_global, i32, immutable
wasm_global:

bar:
    .functype bar () -> (i32)
    i32.const   somedata
    i32.const   somezeroes
    drop
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
  .int32 bar
.size somedata, 8

.section .bss.somezeroes,"",@
somezeroes:
    .int32 0
.size somezeroes, 4

.section .debug_info,"",@
    .int32 bar


#      CHECK:    Addr      Off     Size Out     In      Symbol
# CHECK-NEXT:       -        8        e TYPE
# CHECK-NEXT:       -       16        6 FUNCTION
# CHECK-NEXT:       -       1c        7 TABLE
# CHECK-NEXT:       -       23        5 MEMORY
# CHECK-NEXT:       -       28        f GLOBAL
# CHECK-NEXT:       0        0        0         __stack_pointer
# CHECK-NEXT:       1        0        0         wasm_global
# CHECK-NEXT:       -       37       15 EXPORT
# CHECK-NEXT:       -       4c        9 ELEM
# CHECK-NEXT:       -       55       2d CODE
# CHECK-NEXT:       -       56       10         {{.*}}{{/|\\}}map-file.s.tmp1.o:(bar)
# CHECK-NEXT:       -       56       10                 bar
# CHECK-NEXT:       -       66        b         {{.*}}{{/|\\}}map-file.s.tmp1.o:(write_global)
# CHECK-NEXT:       -       66        b                 write_global
# CHECK-NEXT:       -       71        f         {{.*}}{{/|\\}}map-file.s.tmp1.o:(_start)
# CHECK-NEXT:       -       71        f                 _start
# CHECK-NEXT:       -       82       11 DATA
# CHECK-NEXT:     400       83        8 .data
# CHECK-NEXT:     400       89        8         {{.*}}{{/|\\}}map-file.s.tmp1.o:(.data.somedata)
# CHECK-NEXT:     400       89        8                 somedata
# CHECK-NEXT:     408       82        4 .bss
# CHECK-NEXT:     408        0        4         {{.*}}{{/|\\}}map-file.s.tmp1.o:(.bss.somezeroes)
# CHECK-NEXT:     408        0        4                 somezeroes
# CHECK-NEXT:       -       93       12 CUSTOM(.debug_info)
# CHECK-NEXT:       -       a5       50 CUSTOM(name)

# RUN: not wasm-ld %t1.o -o /dev/null -Map=/ 2>&1 \
# RUN:  | FileCheck -check-prefix=FAIL %s
# FAIL: wasm-ld: error: cannot open map file /
