# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj --triple=wasm32-unknown-unknown -o %t/main.o %t/main.s
# RUN: llvm-mc -filetype=obj --triple=wasm32-unknown-unknown -o %t/liba_x.o %t/liba_x.s
# RUN: llvm-mc -filetype=obj --triple=wasm32-unknown-unknown -o %t/liba_y.o %t/liba_y.s
# RUN: rm -f %t/liba.a
# RUN: llvm-ar rcs %t/liba.a %t/liba_x.o %t/liba_y.o
# RUN: wasm-ld %t/main.o %t/liba.a --gc-sections -o %t/main.wasm --print-gc-sections | FileCheck %s --check-prefix=GC
# RUN: obj2yaml %t/main.wasm | FileCheck %s

# --gc-sections should remove non-retained and unused "weathers" section from live object liba_x.o
# GC: removing unused section {{.*}}/liba.a(liba_x.o):(weathers)
# Should not remove retained "greetings" sections from live objects main.o and liba_x.o
# GC-NOT: removing unused section %t/main.o:(greetings)
# GC-NOT: removing unused section %t/liba_x.o:(greetings)

# Note: All symbols are private so that they don't join the symbol table.

#--- main.s
  .functype grab_liba () -> ()
  .globl  _start
_start:
  .functype _start () -> ()
  call grab_liba
  end_function

  .section greetings,"R",@
  .asciz  "hello"
  .section weathers,"R",@
  .asciz  "cloudy"

#--- liba_x.s
  .globl  grab_liba
grab_liba:
  .functype grab_liba () -> ()
  end_function

  .section greetings,"R",@
  .asciz  "world"
  .section weathers,"",@
  .asciz  "rainy"

#--- liba_y.s
  .section        greetings,"R",@
  .asciz  "bye"


# "greetings" section
# CHECK: - Type:            DATA
# CHECK:   Segments:
# CHECK:     - SectionOffset:   7
# CHECK:       InitFlags:       0
# CHECK:       Offset:
# CHECK:         Opcode:          I32_CONST
# CHECK:         Value:           1024
# CHECK:       Content:         68656C6C6F00776F726C6400
# "weahters" section.
# CHECK: - SectionOffset:   25
# CHECK:   InitFlags:       0
# CHECK:   Offset:
# CHECK:     Opcode:          I32_CONST
# CHECK:     Value:           1036
# CHECK:   Content:         636C6F75647900
