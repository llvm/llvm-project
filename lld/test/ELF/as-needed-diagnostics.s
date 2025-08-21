# REQUIRES: x86
## Test --as-needed-diagnostics and --as-needed-diagnostics-exclude.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 b.s -o b.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 c.s -o c.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 d.s -o d.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 main.s -o main.o
# RUN: ld.lld -shared a.o -soname=liba.so -o liba.so
# RUN: ld.lld -shared b.o liba.so -soname=libb.so -o libb.so
# RUN: ld.lld -shared c.o -soname=libc.so -o libc.so
# RUN: ld.lld -shared d.o -soname=libd-soname.so -o libd.so

## main.o only references libb.so, so liba.so and libc.so are dropped by
## --as-needed. liba.so is a DT_NEEDED entry of libb.so, so it is reported as a
## possible transitive dependency.
# RUN: ld.lld --as-needed --as-needed-diagnostics main.o libb.so liba.so libc.so \
# RUN:   -o main 2>&1 | FileCheck %s --check-prefix=DIAG --implicit-check-not=warning:

# DIAG:      warning: liba.so: DT_NEEDED entry dropped because no symbol from it is used
# DIAG-NEXT: >>> the following linked files share this dependency:
# DIAG-NEXT: >>> libb.so
# DIAG:      warning: libc.so: DT_NEEDED entry dropped because no symbol from it is used

## Nothing is reported without --as-needed-diagnostics.
# RUN: ld.lld --as-needed main.o libb.so liba.so libc.so -o main 2>&1 | count 0

## --as-needed-diagnostics-exclude= suppresses the report for one library.
# RUN: ld.lld --as-needed --as-needed-diagnostics --as-needed-diagnostics-exclude=liba.so \
# RUN:   main.o libb.so liba.so libc.so -o main 2>&1 | \
# RUN:   FileCheck %s --check-prefix=EXCLUDE --implicit-check-not=warning:

# EXCLUDE: warning: libc.so: DT_NEEDED entry dropped because no symbol from it is used

## The option accepts glob patterns and may be specified multiple times.
# RUN: ld.lld --as-needed --as-needed-diagnostics '--as-needed-diagnostics-exclude=lib?.so' \
# RUN:   main.o libb.so liba.so libc.so -o main 2>&1 | count 0
# RUN: ld.lld --as-needed --as-needed-diagnostics --as-needed-diagnostics-exclude=liba.so \
# RUN:   --as-needed-diagnostics-exclude=libc.so main.o libb.so liba.so libc.so -o main 2>&1 | count 0

## If DT_SONAME differs from the input path, the report names both. Either name
## may be excluded.
# RUN: ld.lld --as-needed --as-needed-diagnostics --as-needed-diagnostics-exclude=liba.so \
# RUN:   main.o libb.so liba.so libd.so -o main 2>&1 | \
# RUN:   FileCheck %s --check-prefix=SONAME --implicit-check-not=warning:

# SONAME: warning: libd.so: DT_NEEDED entry libd-soname.so dropped because no symbol from it is used

# RUN: ld.lld --as-needed --as-needed-diagnostics --as-needed-diagnostics-exclude=liba.so \
# RUN:   --as-needed-diagnostics-exclude=libd-soname.so main.o libb.so liba.so libd.so \
# RUN:   -o main 2>&1 | count 0
# RUN: ld.lld --as-needed --as-needed-diagnostics --as-needed-diagnostics-exclude=liba.so \
# RUN:   --as-needed-diagnostics-exclude=libd.so main.o libb.so liba.so libd.so \
# RUN:   -o main 2>&1 | count 0

## Only libraries loaded while --as-needed is in effect are reported.
# RUN: ld.lld --as-needed --as-needed-diagnostics main.o libb.so liba.so --no-as-needed libc.so \
# RUN:   -o main 2>&1 | FileCheck %s --check-prefix=NOASNEEDED --implicit-check-not=warning:

# NOASNEEDED: warning: liba.so: DT_NEEDED entry dropped because no symbol from it is used

## The report is also emitted when producing a shared object.
# RUN: ld.lld -shared --as-needed --as-needed-diagnostics b.o liba.so libc.so \
# RUN:   -o libshared.so 2>&1 | FileCheck %s --check-prefix=SHARED --implicit-check-not=warning:

# SHARED: warning: libc.so: DT_NEEDED entry dropped because no symbol from it is used

## --fatal-warnings promotes the report to an error.
# RUN: not ld.lld --as-needed --as-needed-diagnostics --fatal-warnings main.o libb.so liba.so \
# RUN:   -o main 2>&1 | FileCheck %s --check-prefix=FATAL

# FATAL: error: liba.so: DT_NEEDED entry dropped because no symbol from it is used

## Both options are no-ops without --as-needed.
# RUN: ld.lld --as-needed-diagnostics main.o libb.so liba.so libc.so -o main 2>&1 | \
# RUN:   FileCheck %s --check-prefix=REDUNDANT --implicit-check-not=warning:
# RUN: ld.lld --as-needed-diagnostics-exclude=liba.so main.o libb.so liba.so libc.so \
# RUN:   -o main 2>&1 | FileCheck %s --check-prefix=REDUNDANT --implicit-check-not=warning:

# REDUNDANT: warning: --as-needed-diagnostics and --as-needed-diagnostics-exclude have no effect without --as-needed

## --as-needed-diagnostics-exclude= alone does nothing without --as-needed-diagnostics.
# RUN: ld.lld --as-needed --as-needed-diagnostics-exclude=liba.so main.o libb.so liba.so \
# RUN:   libc.so -o main 2>&1 | FileCheck %s --check-prefix=NO-DIAG --implicit-check-not=warning:

# NO-DIAG: warning: --as-needed-diagnostics-exclude has no effect without --as-needed-diagnostics

## An invalid glob pattern is an error.
# RUN: not ld.lld --as-needed --as-needed-diagnostics '--as-needed-diagnostics-exclude=[' \
# RUN:   main.o libb.so liba.so libc.so -o main 2>&1 | FileCheck %s --check-prefix=BAD-GLOB

# BAD-GLOB: error: --as-needed-diagnostics-exclude: invalid glob pattern, unmatched '[': [

#--- a.s
.globl foo
foo:
  ret

#--- b.s
.globl bar
bar:
  jmp foo@PLT

#--- c.s
.globl baz
baz:
  ret

#--- d.s
.globl qux
qux:
  ret

#--- main.s
.globl _start
_start:
  call bar@PLT
