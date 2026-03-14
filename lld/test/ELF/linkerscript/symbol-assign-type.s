# REQUIRES: x86
## Keep st_type for simple assignment (`alias = aliasee`). This property is
## desired on some targets, where symbol types can affect relocation processing
## (e.g. Thumb interworking). However, the st_size field should not be retained
## because some tools use st_size=0 as a heuristic to detect aliases. With any
## operation, it can be argued that the new symbol may not be of the same type,
## so reset st_type to STT_NOTYPE.

## NOTE: GNU ld retains st_type for many operations.

# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/main.s -o %t.o
# RUN: ld.lld -T %t/a.lds %t.o -o %t1
# RUN: llvm-readelf -s %t1 | FileCheck %s

# CHECK:      Size Type   Bind   Vis     Ndx Name
# CHECK:         1 FUNC   GLOBAL DEFAULT   1 _start
# CHECK:         0 FUNC   GLOBAL DEFAULT   1 retain1
# CHECK-NEXT:    0 FUNC   GLOBAL DEFAULT   1 retain2
# CHECK-NEXT:    0 NOTYPE GLOBAL DEFAULT   1 drop1
# CHECK-NEXT:    0 NOTYPE GLOBAL DEFAULT ABS drop2
# CHECK-NEXT:    0 NOTYPE GLOBAL DEFAULT ABS drop3

# RUN: ld.lld --defsym 'retain=_start' --defsym 'drop=_start+0' %t.o -o %t2
# RUN: llvm-readelf -s %t2 | FileCheck %s --check-prefix=DEFSYM

# DEFSYM:        0 FUNC   GLOBAL DEFAULT   1 retain
# DEFSYM-NEXT:   0 NOTYPE GLOBAL DEFAULT   1 drop

#--- a.lds
retain1 = _start;
retain2 = 1 ? _start : 0;

## Reset to STT_NOTYPE if any operation is performed,
## even if the operation is an identity function.
drop1 = _start + 0;
drop2 = 0 ? _start : 1;
drop3 = -_start;

#--- main.s
.globl _start
.type _start, @function
_start:
  ret
.size _start, 1
