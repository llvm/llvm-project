# REQUIRES: x86
## --wrap=xxx should trigger archive extraction for symbol xxx for references to __real_xxx.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-as _start.ll -o _start.o
# RUN: llvm-as _start_ref__real_foo.ll -o _start_ref__real_foo.o
# RUN: llvm-as wrap.ll -o wrap.o
# RUN: llvm-as foo.ll -o foo.o

## Test when the reference to __real_foo is not in __wrap_foo.
# RUN: ld.lld _start_ref__real_foo.o --start-lib foo.o --end-lib --wrap foo -o %t_real.elf
# RUN: llvm-readelf --symbols %t_real.elf | FileCheck %s --check-prefix=REAL

# REAL:      Symbol table '.symtab' contains 5 entries:
# REAL-NEXT:        Value              Size Type    Bind   Vis       Ndx Name
# REAL-NEXT:        {{.*}}           {{.*}} NOTYPE  LOCAL  DEFAULT   UND
# REAL-NEXT:        {{.*}}           {{.*}} FILE    LOCAL  DEFAULT   ABS ld-temp.o
# REAL-NEXT:        {{.*}}           {{.*}} FUNC    GLOBAL DEFAULT [[#]] _start
# REAL-NEXT:        {{.*}}           {{.*}} FUNC    WEAK   DEFAULT [[#]] foo
# REAL-NEXT:        {{.*}}           {{.*}} NOTYPE  GLOBAL DEFAULT   UND __wrap_foo

## Test when the reference to __real_foo is in __wrap_foo.
# RUN: ld.lld _start.o --start-lib wrap.o --end-lib --start-lib foo.o --end-lib --wrap foo -o %t_wrap_real.elf
# RUN: llvm-readelf --symbols %t_wrap_real.elf | FileCheck %s --check-prefix=WRAP_REAL

# WRAP_REAL:      Symbol table '.symtab' contains 5 entries:
# WRAP_REAL-NEXT:    Value              Size Type    Bind   Vis       Ndx Name
# WRAP_REAL-NEXT:    {{.*}}           {{.*}} NOTYPE  LOCAL  DEFAULT   UND
# WRAP_REAL-NEXT:    {{.*}}           {{.*}} FILE    LOCAL  DEFAULT   ABS ld-temp.o
# WRAP_REAL-NEXT:    {{.*}}           {{.*}} FUNC    GLOBAL DEFAULT [[#]] _start
# WRAP_REAL-NEXT:    {{.*}}           {{.*}} FUNC    GLOBAL DEFAULT [[#]] __wrap_foo
# WRAP_REAL-NEXT:    {{.*}}           {{.*}} FUNC    WEAK   DEFAULT [[#]] foo

#--- _start.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-elf"
define void @_start() {
  ret void
}

#--- _start_ref__real_foo.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-elf"
define void @_start() {
  call void @__real_foo()
  ret void
}

declare void @__real_foo()

#--- wrap.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-elf"
define void @__wrap_foo() {
  call void @__real_foo()
  ret void
}

declare void @__real_foo()

#--- foo.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-elf"
define void @foo() {
  ret void
}
