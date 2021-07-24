# REQUIRES: x86

## FIXME: Add tests for segment$start$foo, segment$end$foo once implemented.

# RUN: rm -rf %t; split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/main.s -o %t/main.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/foo.s -o %t/foo.o

# RUN: %lld -lSystem %t/main.o %t/foo.o -o %t.out \
# RUN:    -rename_section __FOO __bar __BAZ __quux \
# RUN:    -rename_section __WHAT __ever __FOO __bar \
# RUN:    -u 'section$start$__UFLAG_SEG$__uflag_sect' \
# RUN:    -U 'section$start$__DYNAMIC$__lookup' \
# RUN:    -U 'section$start$__DYNAMIC$__unref' \
# RUN:    -e 'section$start$__TEXT$__text'
# RUN: llvm-objdump --macho --syms --section-headers %t.out > %t-dump.txt
# RUN: llvm-objdump --macho -d --no-symbolic-operands --no-show-raw-insn %t.out >> %t-dump.txt
# RUN: llvm-objdump --macho --function-starts %t.out >> %t-dump.txt
# RUN: FileCheck %s < %t-dump.txt

## Setting the entry point to the start of the __text section should
## set it to _main, since that's the first function in that section.
# RUN: llvm-objdump --macho --syms --all-headers %t.out \
# RUN:   | FileCheck --check-prefix=MAINENTRY %s
# MAINENTRY:      [[#%x, MAINADDR:]] g     F __TEXT,__text _main
# MAINENTRY:      LC_MAIN
# MAINENTRY-NEXT: cmdsize
# MAINENTRY-NEXT: entryoff [[#%d, MAINADDR - 0x100000000]]

## Nothing should change if we reorder two functions in the text segment.
## (Reorder some section$start/end symbols too for good measure.)
# RUN: %lld -lSystem %t/main.o %t/foo.o -o %t.ordered.out \
# RUN:    -order_file %t/order.txt \
# RUN:    -rename_section __FOO __bar __BAZ __quux \
# RUN:    -rename_section __WHAT __ever __FOO __bar \
# RUN:    -u 'section$start$__UFLAG_SEG$__uflag_sect' \
# RUN:    -U 'section$start$__DYNAMIC$__lookup' \
# RUN:    -U 'section$start$__DYNAMIC$__unref' \
# RUN:    -e 'section$start$__TEXT$__text'
# RUN: llvm-objdump --macho --syms --section-headers %t.ordered.out > %t-ordered-dump.txt
# RUN: llvm-objdump --macho -d --no-symbolic-operands --no-show-raw-insn %t.ordered.out >> %t-ordered-dump.txt
# RUN: llvm-objdump --macho --function-starts %t.out >> %t-ordered-dump.txt
# RUN: FileCheck %s < %t-ordered-dump.txt

## `-undefined dynamic_lookup` also shouldn't change anything.
# RUN: %lld -lSystem %t/main.o %t/foo.o -o %t.dl.out -undefined dynamic_lookup \
# RUN:    -rename_section __FOO __bar __BAZ __quux \
# RUN:    -rename_section __WHAT __ever __FOO __bar \
# RUN:    -u 'section$start$__UFLAG_SEG$__uflag_sect' \
# RUN:    -U 'section$start$__DYNAMIC$__lookup' \
# RUN:    -U 'section$start$__DYNAMIC$__unref' \
# RUN:    -e 'section$start$__TEXT$__text'
# RUN: llvm-objdump --macho --syms --section-headers %t.dl.out > %t-dump.dl.txt
# RUN: llvm-objdump --macho -d --no-symbolic-operands --no-show-raw-insn %t.dl.out >> %t-dump.dl.txt
# RUN: llvm-objdump --macho --function-starts %t.out >> %t-dump.dl.txt
# RUN: FileCheck %s < %t-dump.dl.txt

## ...except that the entry point is now _otherfun instead of _main since
## _otherfun is now at the start of the __text section.
# RUN: llvm-objdump --macho --syms --all-headers %t.ordered.out \
# RUN:   | FileCheck --check-prefix=OTHERENTRY %s
# OTHERENTRY:      [[#%x, OTHERADDR:]] g     F __TEXT,__text _otherfun
# OTHERENTRY:      LC_MAIN
# OTHERENTRY-NEXT: cmdsize
# OTHERENTRY-NEXT: entryoff [[#%d, OTHERADDR - 0x100000000]]


## Test that the link succeeds with dead-stripping enabled too.
# RUN: %lld -dead_strip -lSystem %t/main.o -o %t/stripped.out

## (Fun fact: `-e 'section$start$__TEXT$__text -dead_strip` strips
## everything in the text section because markLive runs well before
## section$start symbols are replaced, so the entry point is just
## an undefined symbol that keeps nothing alive, and then later it
## sets the entry point to the start of the now-empty text section
## and the output program crashes when running. This matches ld64's
## behavior.)

# CHECK-LABEL: Sections:
# CHECK-NEXT:  Idx Name           Size     VMA              Type
# CHECK:       0 __text           {{[0-9a-f]*}} [[#%x, TEXTSTART:]] TEXT
# CHECK:       1 __aftertext      {{[0-9a-f]*}} [[#%x, TEXTEND:]]
# CHECK:       2 __cstring        {{[0-9a-f]*}} [[#%x, CSTRINGSTART:]] DATA
# CHECK:       3 __aftercstring   {{[0-9a-f]*}} [[#%x, CSTRINGEND:]]
# CHECK:       4 __data           00000008      [[#%x, DATASTART:]] DATA
# CHECK:       5 __llvm_orderfile 00000000      [[#%x, LLVMORDERFILESTART:]] DATA
# CHECK:       6 __mybss          00008000      [[#%x, MYBSSSTART:]] BSS
# CHECK:       7 __quux           0000002a      [[#%x, QUUXSTART:]]
# CHECK:       8 __bar            00000059      [[#%x, BARSTART:]]
# CHECK:       9 __uflag_sect     00000000
# CHECK:       10 __lookup        00000000
# CHECK-NOT:   symbol
# CHECK-NOT:   __unref

# CHECK-LABEL: SYMBOL TABLE:
# CHECK-NOT: section$start$__TEXT$__text
# CHECK-NOT: section$end$__TEXT$__text
# CHECK-NOT: section$start$__TEXT$__cstring
# CHECK-NOT: section$end$__TEXT$__cstring
# CHECK-NOT: section$start$__DATA$__data
# CHECK-NOT: section$end$__DATA$__data
# CHECK-NOT: section$start$__DATA$__llvm_orderfile
# CHECK-NOT: section$end$__DATA$__llvm_orderfile
# CHECK-NOT: section$start$__DYNAMIC$__lookup
# CHECK-NOT: section$start$__DYNAMIC$__unref
# CHECK: section$end$ACTUAL$symbol
# CHECK: section$start$ACTUAL$symbol

# CHECK-LABEL: _main:

## The CHECK-SAMEs work around FileCheck's
## "error: numeric variable 'PC2' defined earlier in the same CHECK directive"
## limitation.
## The 7s are the length of a leaq instruction.
## section$start$__TEXT$__text / section$end$__TEXT$__text

# CHECK:      [[#%x, PC1:]]:
# CHECK-SAME: leaq [[#%d, TEXTSTART - PC1 - 7]](%rip), %rax
# CHECK-NEXT: [[#%x, PC2:]]:
# CHECK-SAME: leaq [[#%d, TEXTEND - PC2 - 7]](%rip), %rbx

## section$start$__TEXT$__cstring / section$end$__TEXT$__cstring
# CHECK:      [[#%x, PC3:]]:
# CHECK-SAME: leaq [[#%d, CSTRINGSTART - PC3 - 7]](%rip), %rax
# CHECK-NEXT: [[#%x, PC4:]]:
# CHECK-SAME: leaq [[#%d, CSTRINGEND - PC4 - 7]](%rip), %rbx

## section$start$__DATA$__data / section$end$__DATA$__data
# CHECK:      [[#%x, PC5:]]:
# CHECK-SAME: leaq [[#%d, DATASTART - PC5 - 7]](%rip), %rax
# CHECK-NEXT: [[#%x, PC6:]]:
# CHECK-SAME: leaq [[#%d, DATASTART + 8 - PC6 - 7]](%rip), %rbx

## section$start$__MYBSS$__mybss / section$end$__MYBSS$__mybss
# CHECK:      [[#%x, PC7:]]:
# CHECK-SAME: leaq [[#%d, MYBSSSTART - PC7 - 7]](%rip), %rax
# CHECK-NEXT: [[#%x, PC8:]]:
# CHECK-SAME: leaq [[#%d, MYBSSSTART + 0x8000 - PC8 - 7]](%rip), %rbx

## section$start$__DATA$__llvm_orderfile / section$end$__DATA$__llvm_orderfile
## This section has size 0.
# CHECK:      [[#%x, PC9:]]:
# CHECK-SAME: leaq [[#%d, LLVMORDERFILESTART - PC9 - 7]](%rip), %rax
# CHECK-NEXT: [[#%x, PC10:]]:
# CHECK-SAME: leaq [[#%d, LLVMORDERFILESTART - PC10 - 7]](%rip), %rbx

## Section-rename tests.
## Input section __FOO/__bar is renamed to output section
## __BAZ/__quux by a -rename_section flag.
## section$start$__FOO$__bar ends up referring to the __BAZ/__quux section.
# CHECK:      [[#%x, PC11:]]:
# CHECK-SAME: leaq [[#%d, QUUXSTART - PC11 - 7]](%rip), %rax
# CHECK-NEXT: [[#%x, PC12:]]:
# CHECK-SAME: leaq [[#%d, QUUXSTART + 42 - PC12 - 7]](%rip), %rbx
## section$start$__BAZ$__quux also refers to the __BAZ/__quux section.
# CHECK:      [[#%x, PC13:]]:
# CHECK-SAME: leaq [[#%d, QUUXSTART - PC13 - 7]](%rip), %rax
# CHECK-NEXT: [[#%x, PC14:]]:
# CHECK-SAME: leaq [[#%d, QUUXSTART + 42 - PC14 - 7]](%rip), %rbx
## Input section __WHAT/__ever is renamed to output section
## __FOO/__bar by a -rename_section flag.
## section$start$__WHAT$__ever ends up referring to the __FOO/__bar section.
# CHECK:      [[#%x, PC15:]]:
# CHECK-SAME: leaq [[#%d, BARSTART - PC15 - 7]](%rip), %rax
# CHECK-NEXT: [[#%x, PC16:]]:
# CHECK-SAME: leaq [[#%d, BARSTART + 89 - PC16 - 7]](%rip), %rbx

## The function_starts section should not have an entry for the
## section$end$__TEXT$__text symbol.
# CHECK: [[#%.16x, TEXTSTART]]
# CHECK-NOT: [[#%.16x, TEXTEND]]

#--- order.txt
_otherfun
_main
section$end$__TEXT$__text
section$start$__TEXT$__text

#--- main.s
.zerofill __MYBSS,__mybss,_zero_foo,0x8000

.globl section$start$ACTUAL$symbol
.globl section$end$ACTUAL$symbol

## Renamed to __BAZ,__quux by -rename_section
.section __FOO,__bar
.space 42

## Renamed to __FOO,__bar by -rename_section
.section __WHAT,__ever
.space 89

.text
.globl _main
_main:
  # Basics: start/end of existing, normal sections.

  # For __TEXT/__text, these magic symbols shouldn't be
  # included in __function_starts
  movq section$start$__TEXT$__text@GOTPCREL(%rip), %rax
  movq section$end$__TEXT$__text@GOTPCREL(%rip), %rbx

  # __TEXT/__cstring are interesting because they're not ConcatInputSections.
  movq section$start$__TEXT$__cstring@GOTPCREL(%rip), %rax
  movq section$end$__TEXT$__cstring@GOTPCREL(%rip), %rbx

  # Vanilla __DATA/__data
  movq section$start$__DATA$__data@GOTPCREL(%rip), %rax
  movq section$end$__DATA$__data@GOTPCREL(%rip), %rbx

  # Vanilla zerofill.
  movq section$start$__MYBSS$__mybss@GOTPCREL(%rip), %rax
  movq section$end$__MYBSS$__mybss@GOTPCREL(%rip), %rbx

  # Referring to a non-existent section wills it into existence.
  # This is needed for e.g. __DATA/__llvm_orderfile in libclang_rt.profile.
  # This means `-u` can be used as a janky `-sectcreate`.
  movq section$start$__DATA$__llvm_orderfile@GOTPCREL(%rip), %rax
  movq section$end$__DATA$__llvm_orderfile@GOTPCREL(%rip), %rbx

  # Section-rename tests.
  movq section$start$__FOO$__bar@GOTPCREL(%rip), %rax
  movq section$end$__FOO$__bar@GOTPCREL(%rip), %rbx

  movq section$start$__BAZ$__quux@GOTPCREL(%rip), %rax
  movq section$end$__BAZ$__quux@GOTPCREL(%rip), %rbx

  movq section$start$__WHAT$__ever@GOTPCREL(%rip), %rax
  movq section$end$__WHAT$__ever@GOTPCREL(%rip), %rbx

  # If there are actual symbols with the magic names, the magic
  # names lose their magic and just refer to those symbols (and
  # no section is implicitly created for them).
  movq section$start$ACTUAL$symbol@GOTPCREL(%rip), %rax
  movq section$end$ACTUAL$symbol@GOTPCREL(%rip), %rbx

  # -U section$start is not exported as dynamic_lookup, it just
  # creates a section like -u.
  movq section$start$__DYNAMIC$__lookup@GOTPCREL(%rip), %rax
  movq section$end$__DYNAMIC$__lookup@GOTPCREL(%rip), %rbx

  ret

.globl _otherfun
_otherfun:
  ret

.section __TEXT,__aftertext
.fill 1

.cstring
.asciz "foo"
.asciz "barbaz"

.section __TEXT,__aftercstring
.fill 1

.data
.quad 0x1234

.subsections_via_symbols

#--- foo.s
.text
.globl section$start$ACTUAL$symbol
section$start$ACTUAL$symbol:
.fill 1

.globl section$end$ACTUAL$symbol
section$end$ACTUAL$symbol:
.fill 1

.subsections_via_symbols
