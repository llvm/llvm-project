# REQUIRES: x86

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o

## Test that --print-relax-stats produces JSON with per-type relaxation counts.
# RUN: ld.lld a.o -o a --print-relax-stats=stats.json
# RUN: FileCheck %s --input-file=stats.json

# CHECK:      "relaxations": {
# CHECK-NEXT:   "R_X86_64_REX_GOTPCRELX": {
# CHECK-NEXT:     "total": 1,
# CHECK-NEXT:     "relaxed": 1,
# CHECK:        }
# CHECK:      }

## --no-relax disables GOT optimization. Verify total > 0 but relaxed == 0.
# RUN: ld.lld --no-relax a.o -o a2 --print-relax-stats=stats-norelax.json
# RUN: FileCheck %s --check-prefix=NORELAX --input-file=stats-norelax.json
# NORELAX:      "relaxations": {
# NORELAX-NEXT:   "R_X86_64_REX_GOTPCRELX": {
# NORELAX-NEXT:     "total": 1,
# NORELAX-NEXT:     "relaxed": 0,
# NORELAX:        }
# NORELAX:      }

## - means stdout.
# RUN: ld.lld a.o -o a3 --print-relax-stats=- | FileCheck %s

## Error opening file.
# RUN: not ld.lld a.o -o /dev/null --print-relax-stats=/ 2>&1 | FileCheck --check-prefix=ERR %s
# ERR: error: --print-relax-stats=: cannot open /: {{.*}}

#--- a.s
.globl _start, foo
.hidden foo

_start:
  movq foo@GOTPCREL(%rip), %rax

foo:
  ret
