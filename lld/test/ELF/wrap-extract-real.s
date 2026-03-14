# REQUIRES: x86
## --wrap=xxx should trigger archive extraction for symbol xxx for references to __real_xxx.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 _start.s -o _start.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 ref__real_foo.s -o ref__real_foo.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 wrap.s -o wrap.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 foo.s -o foo.o

## Test when the reference to __real_foo is not in __wrap_foo.
# RUN: ld.lld _start.o ref__real_foo.o --start-lib foo.o --end-lib --wrap foo -o %t_real.elf
# RUN: llvm-readelf --symbols %t_real.elf | FileCheck %s --check-prefix=REAL

# REAL:      Symbol table '.symtab' contains 4 entries:
# REAL-NEXT:        Value             Size Type    Bind   Vis       Ndx Name
# REAL-NEXT:        {{.*}}               0 NOTYPE  LOCAL  DEFAULT   UND
# REAL-NEXT:        {{.*}}               0 NOTYPE  GLOBAL DEFAULT [[#]] _start
# REAL-NEXT:        {{.*}}               0 NOTYPE  GLOBAL DEFAULT [[#]] foo
# REAL-NEXT:        {{.*}}               0 NOTYPE  GLOBAL DEFAULT   UND __wrap_foo

## Test when the reference to __real_foo is in __wrap_foo.
# RUN: ld.lld _start.o --start-lib wrap.o --end-lib --start-lib foo.o --end-lib --wrap foo -o %t_wrap_real.elf
# RUN: llvm-readelf --symbols %t_wrap_real.elf | FileCheck %s --check-prefix=WRAP_REAL

# WRAP_REAL:      Symbol table '.symtab' contains 4 entries:
# WRAP_REAL-NEXT:        Value             Size Type    Bind   Vis       Ndx Name
# WRAP_REAL-NEXT:        {{.*}}               0 NOTYPE  LOCAL  DEFAULT   UND
# WRAP_REAL-NEXT:        {{.*}}               0 NOTYPE  GLOBAL DEFAULT [[#]] _start
# WRAP_REAL-NEXT:        {{.*}}               0 NOTYPE  GLOBAL DEFAULT [[#]] __wrap_foo
# WRAP_REAL-NEXT:        {{.*}}               0 NOTYPE  GLOBAL DEFAULT [[#]] foo

#--- _start.s
.global _start; _start:; ret

#--- ref__real_foo.s
call __real_foo

#--- wrap.s
.global __wrap_foo; __wrap_foo:; call __real_foo

#--- foo.s
.global foo; foo:; ret
