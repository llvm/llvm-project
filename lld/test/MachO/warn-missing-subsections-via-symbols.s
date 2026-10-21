# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %t/no_subsections.s -o %t/no_subsections.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %t/with_subsections.s -o %t/with_subsections.o

## Default: no warning (even under -fatal_warnings)
# RUN: %lld -dylib %t/no_subsections.o -o /dev/null 2>&1 | count 0

## Warn when flag is passed
# RUN: %no-fatal-warnings-lld -dylib %t/no_subsections.o --warn-missing-subsections-via-symbols -o /dev/null 2>&1 \
# RUN:     | FileCheck %s --check-prefix=WARN

## Overridden by --no-warn-missing-subsections-via-symbols
# RUN: %lld -dylib %t/no_subsections.o --warn-missing-subsections-via-symbols \
# RUN:     --no-warn-missing-subsections-via-symbols -o /dev/null 2>&1 | count 0

## Object with .subsections_via_symbols should not trigger a warning even when flag is passed
# RUN: %lld -dylib %t/with_subsections.o --warn-missing-subsections-via-symbols -o /dev/null 2>&1 | count 0

# WARN: warning: {{.*}}no_subsections.o: missing MH_SUBSECTIONS_VIA_SYMBOLS

#--- no_subsections.s
.text
.globl _foo
_foo:
  ret

#--- with_subsections.s
.text
.globl _bar
_bar:
  ret
.subsections_via_symbols
