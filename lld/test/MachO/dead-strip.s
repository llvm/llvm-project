# REQUIRES: x86, llvm-64-bits

# RUN: rm -rf %t; split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos \
# RUN:     %t/basics.s -o %t/basics.o

## Check that .private_extern symbols are marked as local in the symbol table
## and aren't in the export trie.
# RUN: %lld -lSystem -dead_strip -map %t/map -u _ref_private_extern_u \
# RUN:     %t/basics.o -o %t/basics
# RUN: llvm-objdump --syms --section-headers %t/basics | \
# RUN:     FileCheck --check-prefix=EXEC --implicit-check-not _unref %s
# RUN: llvm-objdump --macho --section=__DATA,__ref_section \
# RUN:     --exports-trie --indirect-symbols %t/basics | \
# RUN:     FileCheck --check-prefix=EXECDATA --implicit-check-not _unref %s
# RUN: llvm-otool -l %t/basics | grep -q 'segname __PAGEZERO'
# EXEC-LABEL: Sections:
# EXEC-LABEL: Name
# EXEC-NEXT:  __text
# EXEC-NEXT:  __got
# EXEC-NEXT:  __ref_section
# EXEC-NEXT:  __common
# EXEC-LABEL: SYMBOL TABLE:
# EXEC-DAG:   l {{.*}} _ref_data
# EXEC-DAG:   l {{.*}} _ref_local
# EXEC-DAG:   l {{.*}} _ref_from_no_dead_strip_globl
# EXEC-DAG:   l {{.*}} _no_dead_strip_local
# EXEC-DAG:   l {{.*}} _ref_from_no_dead_strip_local
# EXEC-DAG:   l {{.*}} _ref_private_extern_u
# EXEC-DAG:   l {{.*}} _main
# EXEC-DAG:   l {{.*}} _ref_private_extern
# EXEC-DAG:   g {{.*}} _no_dead_strip_globl
# EXEC-DAG:   g {{.*}} _ref_com
# EXEC-DAG:   g {{.*}} __mh_execute_header
# EXECDATA-LABEL: Indirect symbols
# EXECDATA-NEXT:  name
# EXECDATA-NEXT:  _ref_com
# EXECDATA-LABEL: Contents of (__DATA,__ref_section) section
# EXECDATA-NEXT:   04 00 00 00 00 00 00 00 05 00 00 00 00 00 00 00
# EXECDATA-LABEL: Exports trie:
# EXECDATA-DAG:   _ref_com
# EXECDATA-DAG:   _no_dead_strip_globl
# EXECDATA-DAG:   __mh_execute_header

## Check that dead stripped symbols get listed properly.
# RUN: FileCheck --check-prefix=MAP %s < %t/map

# MAP: _main
# MAP-LABEL: Dead Stripped Symbols
# MAP: <<dead>> 0x00000000 [ 1] _unref_com
# MAP: <<dead>> 0x00000008 [ 1] _unref_data
# MAP: <<dead>> 0x00000006 [ 1] _unref_extern
# MAP: <<dead>> 0x00000001 [ 1] _unref_local
# MAP: <<dead>> 0x00000007 [ 1] _unref_private_extern
# MAP: <<dead>> 0x00000008 [ 1] l_unref_data

## Run dead stripping on code without any dead symbols.
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos \
# RUN:     %t/no-dead-symbols.s -o %t/no-dead-symbols.o
# RUN: %lld -lSystem -dead_strip -map %t/no-dead-symbols-map \
# RUN:     %t/no-dead-symbols.o -o %t/no-dead-symbols
## Mark the end of the file with a string.
# RUN: FileCheck --check-prefix=NODEADSYMBOLS %s < %t/no-dead-symbols-map

# NODEADSYMBOLS-LABEL: # Symbols:
# NODEADSYMBOLS-NEXT:  # Address Size File Name
# NODEADSYMBOLS-NEXT:  _main
# NODEADSYMBOLS-LABEL: # Dead Stripped Symbols:
# NODEADSYMBOLS-NEXT:  # Size File Name
# NODEADSYMBOLS-EMPTY:

# RUN: %lld -dylib -dead_strip -u _ref_private_extern_u %t/basics.o -o %t/basics.dylib
# RUN: llvm-objdump --syms %t/basics.dylib | \
# RUN:     FileCheck --check-prefix=DYLIB --implicit-check-not _unref %s
# RUN: %lld -bundle -dead_strip -u _ref_private_extern_u %t/basics.o -o %t/basics.dylib
# RUN: llvm-objdump --syms %t/basics.dylib | \
# RUN:     FileCheck --check-prefix=DYLIB --implicit-check-not _unref %s
# DYLIB-LABEL: SYMBOL TABLE:
# DYLIB-DAG:   l {{.*}} _ref_data
# DYLIB-DAG:   l {{.*}} _ref_local
# DYLIB-DAG:   l {{.*}} _ref_from_no_dead_strip_globl
# DYLIB-DAG:   l {{.*}} _no_dead_strip_local
# DYLIB-DAG:   l {{.*}} _ref_from_no_dead_strip_local
# DYLIB-DAG:   l {{.*}} _ref_private_extern_u
# DYLIB-DAG:   l {{.*}} _ref_private_extern
# DYLIB-DAG:   g {{.*}} _ref_com
# DYLIB-DAG:   g {{.*}} _unref_com
# DYLIB-DAG:   g {{.*}} _unref_extern
# DYLIB-DAG:   g {{.*}} _no_dead_strip_globl

## Extern symbols aren't stripped from executables with -export_dynamic
# RUN: %lld -lSystem -dead_strip -export_dynamic -u _ref_private_extern_u \
# RUN:     %t/basics.o -o %t/basics-export-dyn
# RUN: llvm-objdump --syms --section-headers %t/basics-export-dyn | \
# RUN:     FileCheck --check-prefix=EXECDYN %s
# EXECDYN-LABEL: Sections:
# EXECDYN-LABEL: Name
# EXECDYN-NEXT:  __text
# EXECDYN-NEXT:  __got
# EXECDYN-NEXT:  __ref_section
# EXECDYN-NEXT:  __common
# EXECDYN-LABEL: SYMBOL TABLE:
# EXECDYN-DAG:   l {{.*}} _ref_data
# EXECDYN-DAG:   l {{.*}} _ref_local
# EXECDYN-DAG:   l {{.*}} _ref_from_no_dead_strip_globl
# EXECDYN-DAG:   l {{.*}} _no_dead_strip_local
# EXECDYN-DAG:   l {{.*}} _ref_from_no_dead_strip_local
# EXECDYN-DAG:   l {{.*}} _ref_private_extern_u
# EXECDYN-DAG:   l {{.*}} _main
# EXECDYN-DAG:   l {{.*}} _ref_private_extern
# EXECDYN-DAG:   g {{.*}} _ref_com
# EXECDYN-DAG:   g {{.*}} _unref_com
# EXECDYN-DAG:   g {{.*}} _unref_extern
# EXECDYN-DAG:   g {{.*}} _no_dead_strip_globl
# EXECDYN-DAG:   g {{.*}} __mh_execute_header

## Absolute symbol handling.
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos \
# RUN:     %t/abs.s -o %t/abs.o
# RUN: %lld -lSystem -dead_strip %t/abs.o -o %t/abs
# RUN: llvm-objdump --macho --syms --exports-trie %t/abs | \
# RUN:     FileCheck --check-prefix=ABS %s
#ABS-LABEL: SYMBOL TABLE:
#ABS-NEXT:   g {{.*}} _main
#ABS-NEXT:   g *ABS* _abs1
#ABS-NEXT:   g {{.*}} __mh_execute_header
#ABS-LABEL: Exports trie:
#ABS-NEXT:   __mh_execute_header
#ABS-NEXT:   _main
#ABS-NEXT:   _abs1 [absolute]

## Check that symbols from -exported_symbol(s_list) are preserved.
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos \
# RUN:     %t/exported-symbol.s -o %t/exported-symbol.o
# RUN: %lld -lSystem -dead_strip -exported_symbol _my_exported_symbol \
# RUN:     %t/exported-symbol.o -o %t/exported-symbol
# RUN: llvm-objdump --syms %t/exported-symbol | \
# RUN:     FileCheck --check-prefix=EXPORTEDSYMBOL --implicit-check-not _unref %s
# EXPORTEDSYMBOL-LABEL: SYMBOL TABLE:
# EXPORTEDSYMBOL-NEXT:   l {{.*}} _main
# EXPORTEDSYMBOL-NEXT:   l {{.*}} __mh_execute_header
# EXPORTEDSYMBOL-NEXT:   g {{.*}} _my_exported_symbol

## Check that mod_init_funcs and mod_term_funcs are not stripped.
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos \
# RUN:     %t/mod-funcs.s -o %t/mod-funcs.o
# RUN: %lld -lSystem -dead_strip %t/mod-funcs.o -o %t/mod-funcs
# RUN: llvm-objdump --syms %t/mod-funcs | \
# RUN:     FileCheck --check-prefix=MODFUNCS --implicit-check-not _unref %s
# MODFUNCS-LABEL: SYMBOL TABLE:
# MODFUNCS-NEXT:   l {{.*}} _ref_from_init
# MODFUNCS-NEXT:   l {{.*}} _ref_init
# MODFUNCS-NEXT:   l {{.*}} _ref_from_term
# MODFUNCS-NEXT:   l {{.*}} _ref_term
# MODFUNCS-NEXT:   g {{.*}} _main
# MODFUNCS-NEXT:   g {{.*}} __mh_execute_header

## Check that DylibSymbols in dead subsections are stripped: They should
## not be in the import table and should have no import stubs.
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos \
# RUN:     %t/dylib.s -o %t/dylib.o
# RUN: %lld -dylib -dead_strip %t/dylib.o -o %t/dylib.dylib
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos \
# RUN:     %t/strip-dylib-ref.s -o %t/strip-dylib-ref.o
# RUN: %lld -lSystem -dead_strip %t/strip-dylib-ref.o %t/dylib.dylib \
# RUN:     -o %t/strip-dylib-ref -U _ref_undef_fun -U _unref_undef_fun
# RUN: llvm-objdump --syms --bind --lazy-bind --weak-bind %t/strip-dylib-ref | \
# RUN:     FileCheck --check-prefix=STRIPDYLIB --implicit-check-not _unref %s
# STRIPDYLIB:      SYMBOL TABLE:
# STRIPDYLIB-NEXT:  l {{.*}} __dyld_private
# STRIPDYLIB-NEXT:  g {{.*}} _main
# STRIPDYLIB-NEXT:  g {{.*}} __mh_execute_header
# STRIPDYLIB-NEXT:  *UND* dyld_stub_binder
# STRIPDYLIB-NEXT:  *UND* _ref_dylib_fun
# STRIPDYLIB-NEXT:  *UND* _ref_undef_fun
# STRIPDYLIB:      Bind table:
# STRIPDYLIB:      Lazy bind table:
# STRIPDYLIB:       __DATA   __la_symbol_ptr {{.*}} flat-namespace _ref_undef_fun
# STRIPDYLIB:       __DATA   __la_symbol_ptr {{.*}} dylib _ref_dylib_fun
# STRIPDYLIB:      Weak bind table:
## Stubs smoke check: There should be two stubs entries, not four, but we
## don't verify that they belong to _ref_undef_fun and _ref_dylib_fun.
# RUN: llvm-objdump -d --section=__stubs --section=__stub_helper \
# RUN:     %t/strip-dylib-ref |FileCheck --check-prefix=STUBS %s
# STUBS-LABEL: <__stubs>:
# STUBS-NEXT:  jmpq
# STUBS-NEXT:  jmpq
# STUBS-NOT:   jmpq
# STUBS-LABEL: <__stub_helper>:
# STUBS:  pushq $0
# STUBS:  jmp
# STUBS:  jmp
# STUBS-NOT:  jmp
## An undefined symbol referenced from a dead-stripped function shouldn't
## produce a diagnostic:
# RUN: %lld -lSystem -dead_strip %t/strip-dylib-ref.o %t/dylib.dylib \
# RUN:     -o %t/strip-dylib-ref -U _ref_undef_fun

## Check that referenced undefs are kept with -undefined dynamic_lookup.
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos \
# RUN:     %t/ref-undef.s -o %t/ref-undef.o
# RUN: %lld -lSystem -dead_strip %t/ref-undef.o \
# RUN:     -o %t/ref-undef -undefined dynamic_lookup
# RUN: llvm-objdump --syms --lazy-bind %t/ref-undef | \
# RUN:     FileCheck --check-prefix=STRIPDYNLOOKUP %s
# STRIPDYNLOOKUP: SYMBOL TABLE:
# STRIPDYNLOOKUP:   *UND* _ref_undef_fun
# STRIPDYNLOOKUP: Lazy bind table:
# STRIPDYNLOOKUP:   __DATA   __la_symbol_ptr {{.*}} flat-namespace _ref_undef_fun

## S_ATTR_LIVE_SUPPORT tests.
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos \
# RUN:     %t/live-support.s -o %t/live-support.o
# RUN: %lld -lSystem -dead_strip %t/live-support.o %t/dylib.dylib \
# RUN:     -U _ref_undef_fun -U _unref_undef_fun -o %t/live-support
# RUN: llvm-objdump --syms %t/live-support | \
# RUN:     FileCheck --check-prefix=LIVESUPP --implicit-check-not _unref %s
# LIVESUPP-LABEL: SYMBOL TABLE:
# LIVESUPP-NEXT:   l {{.*}} _ref_ls_fun_fw
# LIVESUPP-NEXT:   l {{.*}} _ref_ls_fun_bw
# LIVESUPP-NEXT:   l {{.*}} _ref_ls_dylib_fun
# LIVESUPP-NEXT:   l {{.*}} _ref_ls_undef_fun
# LIVESUPP-NEXT:   l {{.*}} __dyld_private
# LIVESUPP-NEXT:   g {{.*}} _main
# LIVESUPP-NEXT:   g {{.*}} _bar
# LIVESUPP-NEXT:   g {{.*}} _foo
# LIVESUPP-NEXT:   g {{.*}} __mh_execute_header
# LIVESUPP-NEXT:   *UND* dyld_stub_binder
# LIVESUPP-NEXT:   *UND* _ref_dylib_fun
# LIVESUPP-NEXT:   *UND* _ref_undef_fun

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos \
# RUN:     %t/live-support-iterations.s -o %t/live-support-iterations.o
# RUN: %lld -lSystem -dead_strip %t/live-support-iterations.o \
# RUN:     -o %t/live-support-iterations
# RUN: llvm-objdump --syms %t/live-support-iterations | \
# RUN:     FileCheck --check-prefix=LIVESUPP2 --implicit-check-not _unref %s
# LIVESUPP2-LABEL: SYMBOL TABLE:
# LIVESUPP2-NEXT:   l {{.*}} _bar
# LIVESUPP2-NEXT:   l {{.*}} _foo_refd
# LIVESUPP2-NEXT:   l {{.*}} _bar_refd
# LIVESUPP2-NEXT:   l {{.*}} _baz
# LIVESUPP2-NEXT:   l {{.*}} _baz_refd
# LIVESUPP2-NEXT:   l {{.*}} _foo
# LIVESUPP2-NEXT:   g {{.*}} _main
# LIVESUPP2-NEXT:   g {{.*}} __mh_execute_header

## Dead stripping should not remove the __TEXT,__unwind_info
## and __TEXT,__gcc_except_tab functions, but it should still
## remove the unreferenced function __Z5unref.
## The reference to ___gxx_personality_v0 should also not be
## stripped.
## (Need to use darwin19.0.0 to make -mc emit __LD,__compact_unwind.)
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin19.0.0 \
# RUN:     %t/unwind.s -o %t/unwind.o
# RUN: %lld -lc++ -lSystem -dead_strip %t/unwind.o -o %t/unwind
# RUN: llvm-objdump --syms %t/unwind | \
# RUN:     FileCheck --check-prefix=UNWIND --implicit-check-not unref %s
# RUN: llvm-otool -l %t/unwind | FileCheck --check-prefix=UNWINDSECT %s
# UNWINDSECT-DAG: sectname __unwind_info
# UNWINDSECT-DAG: sectname __gcc_except_tab
# UNWIND-LABEL: SYMBOL TABLE:
# UNWIND-NEXT:   l O __TEXT,__gcc_except_tab GCC_except_table1
# UNWIND-NEXT:   l O __DATA,__data __dyld_private
# UNWIND-NEXT:   g F __TEXT,__text _main
# UNWIND-NEXT:   g F __TEXT,__text __mh_execute_header
# UNWIND-NEXT:   *UND* dyld_stub_binder
# UNWIND-NEXT:   *UND* ___cxa_allocate_exception
# UNWIND-NEXT:   *UND* ___cxa_end_catch
# UNWIND-NEXT:   *UND* __ZTIi
# UNWIND-NEXT:   *UND* ___cxa_throw
# UNWIND-NEXT:   *UND* ___gxx_personality_v0
# UNWIND-NEXT:   *UND* ___cxa_begin_catch
# UNWIND-NOT:    GCC_except_table0

## If a dead stripped function has a strong ref to a dylib symbol but
## a live function only a weak ref, the dylib is still not a WEAK_DYLIB.
## This matches ld64.
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos \
# RUN:     %t/weak-ref.s -o %t/weak-ref.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos \
# RUN:     %t/strong-dead-ref.s -o %t/strong-dead-ref.o
# RUN: %lld -lSystem -dead_strip %t/weak-ref.o %t/strong-dead-ref.o \
# RUN:     %t/dylib.dylib -o %t/weak-ref
# RUN: llvm-otool -l %t/weak-ref | FileCheck -DDIR=%t --check-prefix=WEAK %s
# WEAK:          cmd LC_LOAD_DYLIB
# WEAK-NEXT: cmdsize
# WEAK-NEXT:    name /usr/lib/libSystem.dylib
# WEAK:          cmd LC_LOAD_DYLIB
# WEAK-NEXT: cmdsize
# WEAK-NEXT:    name [[DIR]]/dylib.dylib

## A strong symbol that would override a weak import does not emit the
## "this overrides a weak import" opcode if it is dead-stripped.
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos \
# RUN:     %t/weak-dylib.s -o %t/weak-dylib.o
# RUN: %lld -dylib -dead_strip %t/weak-dylib.o -o %t/weak-dylib.dylib
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos \
# RUN:     %t/dead-weak-override.s -o %t/dead-weak-override.o
# RUN: %lld -dead_strip %t/dead-weak-override.o %t/weak-dylib.dylib \
# RUN:     -o %t/dead-weak-override
# RUN: llvm-objdump --macho --weak-bind --private-header \
# RUN:     %t/dead-weak-override | FileCheck --check-prefix=DEADWEAK %s
# DEADWEAK-NOT: WEAK_DEFINES
# DEADWEAK:     Weak bind table:
# DEADWEAK:     segment  section            address     type       addend   symbol
# DEADWEAK-NOT: strong              _weak_in_dylib

## Stripped symbols should not be in the debug info stabs entries.
# RUN: llvm-mc -g -filetype=obj -triple=x86_64-apple-macos \
# RUN:     %t/debug.s -o %t/debug.o
# RUN: %lld -lSystem -dead_strip %t/debug.o -o %t/debug
# RUN: dsymutil -s %t/debug | FileCheck --check-prefix=EXECSTABS %s
# EXECSTABS-NOT: N_FUN {{.*}} '_unref'
# EXECSTABS:     N_FUN {{.*}} '_main'
# EXECSTABS-NOT: N_FUN {{.*}} '_unref'

# RUN: llvm-mc -g -filetype=obj -triple=x86_64-apple-macos \
# RUN:     %t/literals.s -o %t/literals.o
# RUN: %lld -dylib -dead_strip --deduplicate-literals %t/literals.o -o %t/literals
# RUN: llvm-objdump --macho --section="__TEXT,__cstring" --section="__DATA,str_ptrs" \
# RUN:   --section="__TEXT,__literals" %t/literals | FileCheck %s --check-prefix=LIT
# LIT:      Contents of (__TEXT,__cstring) section
# LIT-NEXT: foobar
# LIT-NEXT: Contents of (__DATA,str_ptrs) section
# LIT-NEXT: __TEXT:__cstring:bar
# LIT-NEXT: __TEXT:__cstring:bar
# LIT-NEXT: Contents of (__TEXT,__literals) section
# LIT-NEXT: ef be ad de {{$}}

## Duplicate symbols that will be dead stripped later should not fail when using
## the --dead-stripped-duplicates flag
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos \
# RUN:     %t/duplicate1.s -o %t/duplicate1.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos \
# RUN:     %t/duplicate2.s -o %t/duplicate2.o
# RUN: %lld -lSystem -dead_strip --dead-strip-duplicates -map %t/stripped-duplicate-map \
# RUN:     %t/duplicate1.o %t/duplicate2.o -o %t/duplicate
# RUN: llvm-objdump --syms %t/duplicate | FileCheck %s --check-prefix=DUP
# DUP-LABEL: SYMBOL TABLE:
# DUP-NEXT:   g F __TEXT,__text _main
# DUP-NEXT:   g F __TEXT,__text __mh_execute_header
# DUP-NEXT:   *UND* dyld_stub_binder

## Check that the duplicate dead stripped symbols get listed properly.
# RUN: FileCheck --check-prefix=DUPMAP %s < %t/stripped-duplicate-map
# DUPMAP: _main
# DUPMAP-LABEL: Dead Stripped Symbols
# DUPMAP: <<dead>> 0x00000001 [ 2] _foo

#--- duplicate1.s
.text
.globl _main, _foo
_foo:
  retq

_main:
  retq

.subsections_via_symbols

#--- duplicate2.s
.text
.globl _foo
_foo:
  retq

.subsections_via_symbols

#--- basics.s
.comm _ref_com, 1
.comm _unref_com, 1

.section __DATA,__unref_section
_unref_data:
  .quad 4

l_unref_data:
  .quad 5

## Referenced by no_dead_strip == S_ATTR_NO_DEAD_STRIP
.section __DATA,__ref_section,regular,no_dead_strip

## Referenced because in no_dead_strip section.
_ref_data:
  .quad 4

## This is a local symbol so it's not in the symbol table, but
## it is still in the section data.
l_ref_data:
  .quad 5

.text

# Exported symbols should not be stripped from dylibs
# or bundles, but they should be stripped from executables.
.globl _unref_extern
_unref_extern:
  callq _ref_local
  retq

# Unreferenced local symbols should be stripped.
_unref_local:
  retq

# Same for unreferenced private externs.
.globl _unref_private_extern
.private_extern _unref_private_extern
_unref_private_extern:
  # This shouldn't create an indirect symbol since it's
  # a reference from a dead function.
  movb _unref_com@GOTPCREL(%rip), %al
  retq

# Referenced local symbols should not be stripped.
_ref_local:
  callq _ref_private_extern
  retq

# Same for referenced private externs.
# This one is referenced by a relocation.
.globl _ref_private_extern
.private_extern _ref_private_extern
_ref_private_extern:
  retq

# This one is referenced by a -u flag.
.globl _ref_private_extern_u
.private_extern _ref_private_extern_u
_ref_private_extern_u:
  retq

# Entry point should not be stripped for executables, even if hidden.
# For shared libraries this is stripped since it's just a regular hidden
# symbol there.
.globl _main
.private_extern _main
_main:
  movb _ref_com@GOTPCREL(%rip), %al
  callq _ref_local
  retq

# Things marked no_dead_strip should not be stripped either.
# (clang emits this e.g. for `__attribute__((used))` globals.)
# Both for .globl symbols...
.globl _no_dead_strip_globl
.no_dead_strip _no_dead_strip_globl
_no_dead_strip_globl:
  callq _ref_from_no_dead_strip_globl
  retq
_ref_from_no_dead_strip_globl:
  retq

# ...and for locals.
.no_dead_strip _no_dead_strip_local
_no_dead_strip_local:
  callq _ref_from_no_dead_strip_local
  retq
_ref_from_no_dead_strip_local:
  retq

.subsections_via_symbols

#--- exported-symbol.s
.text

.globl _unref_symbol
_unref_symbol:
  retq

.globl _my_exported_symbol
_my_exported_symbol:
  retq

.globl _main
_main:
  retq

.subsections_via_symbols

#--- abs.s
.globl _abs1, _abs2, _abs3

.no_dead_strip _abs1
_abs1 = 1
_abs2 = 2
_abs3 = 3

.section __DATA,__foo,regular,no_dead_strip
# Absolute symbols are not in a section, so the no_dead_strip
# on the section above has no effect.
.globl _abs4
_abs4 = 4

.text
.globl _main
_main:
  # This is relaxed away, so there's no relocation here and
  # _abs3 isn't in the exported symbol table.
  mov _abs3, %rax
  retq

.subsections_via_symbols

#--- mod-funcs.s
## Roughly based on `clang -O2 -S` output for `struct A { A(); ~A(); }; A a;`
## for mod_init_funcs. mod_term_funcs then similar to that.
.section __TEXT,__StaticInit,regular,pure_instructions

__unref:
  retq

_ref_from_init:
  retq

_ref_init:
  callq _ref_from_init
  retq

_ref_from_term:
  retq

_ref_term:
  callq _ref_from_term
  retq

.globl _main
_main:
  retq

.section __DATA,__mod_init_func,mod_init_funcs
.quad _ref_init

.section __DATA,__mod_term_func,mod_term_funcs
.quad _ref_term

.subsections_via_symbols

#--- dylib.s
.text

.globl _ref_dylib_fun
_ref_dylib_fun:
  retq

.globl _unref_dylib_fun
_unref_dylib_fun:
  retq

.subsections_via_symbols

#--- strip-dylib-ref.s
.text

_unref:
  callq _ref_dylib_fun
  callq _unref_dylib_fun
  callq _ref_undef_fun
  callq _unref_undef_fun
  retq

.globl _main
_main:
  callq _ref_dylib_fun
  callq _ref_undef_fun
  retq

.subsections_via_symbols

#--- live-support.s
## In practice, live_support is used for instruction profiling
## data and asan. (Also for __eh_frame, but that needs special handling
## in the linker anyways.)
## This test isn't based on anything happening in real code though.
.section __TEXT,__ref_ls_fw,regular,live_support
_ref_ls_fun_fw:
  # This is called by _main and is kept alive by normal
  # forward liveness propagation, The live_support attribute
  # does nothing in this case.
  retq

.section __TEXT,__unref_ls_fw,regular,live_support
_unref_ls_fun_fw:
  retq

.section __TEXT,__ref_ls_bw,regular,live_support
_ref_ls_fun_bw:
  # This _calls_ something that's alive but isn't referenced itself. This is
  # kept alive only due to this being in a live_support section.
  callq _foo

  # _bar on the other hand is kept alive since it's called from here.
  callq _bar
  retq

## Kept alive by a live symbol form a dynamic library.
_ref_ls_dylib_fun:
  callq _ref_dylib_fun
  retq

## Kept alive by a live undefined symbol.
_ref_ls_undef_fun:
  callq _ref_undef_fun
  retq

## All symbols in this live_support section reference dead symbols
## and are hence dead themselves.
.section __TEXT,__unref_ls_bw,regular,live_support
_unref_ls_fun_bw:
  callq _unref
  retq

_unref_ls_dylib_fun_bw:
  callq _unref_dylib_fun
  retq

_unref_ls_undef_fun_bw:
  callq _unref_undef_fun
  retq

.text
.globl _unref
_unref:
  retq

.globl _bar
_bar:
  retq

.globl _foo
_foo:
  callq _ref_ls_fun_fw
  retq

.globl _main
_main:
  callq _ref_ls_fun_fw
  callq _foo
  callq _ref_dylib_fun
  callq _ref_undef_fun
  retq

.subsections_via_symbols

#--- live-support-iterations.s
.section __TEXT,_ls,regular,live_support

## This is a live_support subsection that only becomes
## live after _foo below is processed. This means the algorithm of
## 1. mark things reachable from gc roots live
## 2. go through live sections and mark the ones live pointing to
##    live symbols or sections
## needs more than one iteration, since _bar won't be live when step 2
## runs for the first time.
## (ld64 gets this wrong -- it has different output based on if _bar is
## before _foo or after it.)
_bar:
  callq _foo_refd
  callq _bar_refd
  retq

## Same here. This is maybe more interesting since it references a live_support
## symbol instead of a "normal" symbol.
_baz:
  callq _foo_refd
  callq _baz_refd
  retq

_foo:
  callq _main
  callq _foo_refd
  retq

## Test no_dead_strip on a symbol in a live_support section.
## ld64 ignores this, but that doesn't look intentional. So lld honors it.
.no_dead_strip
_quux:
  retq


.text
.globl _main
_main:
  movq $0, %rax
  retq

_foo_refd:
  retq

_bar_refd:
  retq

_baz_refd:
  retq

.subsections_via_symbols

#--- unwind.s
## This is the output of `clang -O2 -S throw.cc` where throw.cc
## looks like this:
##     int unref() {
##       try {
##         throw 0;
##       } catch (int i) {
##         return i + 1;
##       }
##     }
##     int main() {
##       try {
##         throw 0;
##       } catch (int i) {
##         return i;
##       }
##     }
.section __TEXT,__text,regular,pure_instructions
.globl __Z5unrefv                      ## -- Begin function _Z5unrefv
.p2align 4, 0x90
__Z5unrefv:                             ## @_Z5unrefv
Lfunc_begin0:
  .cfi_startproc
  .cfi_personality 155, ___gxx_personality_v0
  .cfi_lsda 16, Lexception0
## %bb.0:
  pushq  %rbp
  .cfi_def_cfa_offset 16
  .cfi_offset %rbp, -16
  movq  %rsp, %rbp
  .cfi_def_cfa_register %rbp
  subq  $16, %rsp
  movl  $4, %edi
  callq  ___cxa_allocate_exception
  movl  $0, (%rax)
Ltmp0:
  movq  __ZTIi@GOTPCREL(%rip), %rsi
  movq  %rax, %rdi
  xorl  %edx, %edx
  callq  ___cxa_throw
Ltmp1:
## %bb.1:
  ud2
LBB0_2:
Ltmp2:
  leaq  -4(%rbp), %rcx
  movq  %rax, %rdi
  movl  %edx, %esi
  movq  %rcx, %rdx
  callq  __Z5unrefv.cold.1
  movl  -4(%rbp), %eax
  addq  $16, %rsp
  popq  %rbp
  retq
Lfunc_end0:
  .cfi_endproc
  .section  __TEXT,__gcc_except_tab
  .p2align 2
GCC_except_table0:
Lexception0:
  .byte 255                             ## @LPStart Encoding = omit
  .byte 155                             ## @TType Encoding = indirect pcrel sdata4
  .uleb128 Lttbase0-Lttbaseref0
Lttbaseref0:
  .byte 1                               ## Call site Encoding = uleb128
  .uleb128 Lcst_end0-Lcst_begin0
Lcst_begin0:
  .uleb128 Lfunc_begin0-Lfunc_begin0    ## >> Call Site 1 <<
  .uleb128 Ltmp0-Lfunc_begin0           ##   Call between Lfunc_begin0 and Ltmp0
  .byte 0                               ##     has no landing pad
  .byte 0                               ##   On action: cleanup
  .uleb128 Ltmp0-Lfunc_begin0           ## >> Call Site 2 <<
  .uleb128 Ltmp1-Ltmp0                  ##   Call between Ltmp0 and Ltmp1
  .uleb128 Ltmp2-Lfunc_begin0           ##     jumps to Ltmp2
  .byte 1                               ##   On action: 1
  .uleb128 Ltmp1-Lfunc_begin0           ## >> Call Site 3 <<
  .uleb128 Lfunc_end0-Ltmp1             ##   Call between Ltmp1 and Lfunc_end0
  .byte 0                               ##     has no landing pad
  .byte 0                               ##   On action: cleanup
Lcst_end0:
  .byte 1                               ## >> Action Record 1 <<
                                        ##   Catch TypeInfo 1
  .byte 0                               ##   No further actions
  .p2align 2
                                        ## >> Catch TypeInfos <<
  .long  __ZTIi@GOTPCREL+4              ## TypeInfo 1
Lttbase0:
  .p2align 2
                                        ## -- End function
  .section  __TEXT,__text,regular,pure_instructions
  .globl  _main                         ## -- Begin function main
  .p2align 4, 0x90
_main:                                  ## @main
Lfunc_begin1:
  .cfi_startproc
  .cfi_personality 155, ___gxx_personality_v0
  .cfi_lsda 16, Lexception1
## %bb.0:
  pushq  %rbp
  .cfi_def_cfa_offset 16
  .cfi_offset %rbp, -16
  movq  %rsp, %rbp
  .cfi_def_cfa_register %rbp
  pushq  %rbx
  pushq  %rax
  .cfi_offset %rbx, -24
  movl  $4, %edi
  callq  ___cxa_allocate_exception
  movl  $0, (%rax)
Ltmp3:
  movq  __ZTIi@GOTPCREL(%rip), %rsi
  movq  %rax, %rdi
  xorl  %edx, %edx
  callq ___cxa_throw
Ltmp4:
## %bb.1:
  ud2
LBB1_2:
Ltmp5:
  movq  %rax, %rdi
  callq ___cxa_begin_catch
  movl  (%rax), %ebx
  callq ___cxa_end_catch
  movl  %ebx, %eax
  addq  $8, %rsp
  popq  %rbx
  popq  %rbp
  retq
Lfunc_end1:
  .cfi_endproc
  .section  __TEXT,__gcc_except_tab
  .p2align  2
GCC_except_table1:
Lexception1:
  .byte 255                             ## @LPStart Encoding = omit
  .byte 155                             ## @TType Encoding = indirect pcrel sdata4
  .uleb128 Lttbase1-Lttbaseref1
Lttbaseref1:
  .byte 1                               ## Call site Encoding = uleb128
  .uleb128 Lcst_end1-Lcst_begin1
Lcst_begin1:
  .uleb128 Lfunc_begin1-Lfunc_begin1    ## >> Call Site 1 <<
  .uleb128 Ltmp3-Lfunc_begin1           ##   Call between Lfunc_begin1 and Ltmp3
  .byte 0                               ##     has no landing pad
  .byte 0                               ##   On action: cleanup
  .uleb128 Ltmp3-Lfunc_begin1           ## >> Call Site 2 <<
  .uleb128 Ltmp4-Ltmp3                  ##   Call between Ltmp3 and Ltmp4
  .uleb128 Ltmp5-Lfunc_begin1           ##     jumps to Ltmp5
  .byte 1                               ##   On action: 1
  .uleb128 Ltmp4-Lfunc_begin1           ## >> Call Site 3 <<
  .uleb128 Lfunc_end1-Ltmp4             ##   Call between Ltmp4 and Lfunc_end1
  .byte 0                               ##     has no landing pad
  .byte 0                               ##   On action: cleanup
Lcst_end1:
  .byte 1                               ## >> Action Record 1 <<
                                        ##   Catch TypeInfo 1
  .byte 0                               ##   No further actions
  .p2align 2
                                        ## >> Catch TypeInfos <<
  .long __ZTIi@GOTPCREL+4               ## TypeInfo 1
Lttbase1:
  .p2align 2
                                        ## -- End function
  .section __TEXT,__text,regular,pure_instructions
  .p2align 4, 0x90                      ## -- Begin function _Z5unrefv.cold.1
__Z5unrefv.cold.1:                      ## @_Z5unrefv.cold.1
  .cfi_startproc
## %bb.0:
  pushq  %rbp
  .cfi_def_cfa_offset 16
  .cfi_offset %rbp, -16
  movq  %rsp, %rbp
  .cfi_def_cfa_register %rbp
  pushq  %rbx
  pushq  %rax
  .cfi_offset %rbx, -24
  movq  %rdx, %rbx
  callq  ___cxa_begin_catch
  movl  (%rax), %eax
  incl  %eax
  movl  %eax, (%rbx)
  addq  $8, %rsp
  popq  %rbx
  popq  %rbp
  jmp  ___cxa_end_catch                 ## TAILCALL
  .cfi_endproc
                                        ## -- End function
.subsections_via_symbols

#--- weak-ref.s
.text
.weak_reference _ref_dylib_fun
.globl _main
_main:
  callq _ref_dylib_fun
  retq

.subsections_via_symbols

#--- strong-dead-ref.s
.text
.globl _unref_dylib_fun
_unref:
  callq _unref_dylib_fun
  retq

.subsections_via_symbols

#--- weak-dylib.s
.text
.globl _weak_in_dylib
.weak_definition _weak_in_dylib
_weak_in_dylib:
  retq

.subsections_via_symbols

#--- dead-weak-override.s

## Overrides the _weak_in_dylib symbol in weak-dylib, but is dead stripped.
.text

#.no_dead_strip _weak_in_dylib
.globl _weak_in_dylib
_weak_in_dylib:
  retq

.globl _main
_main:
  retq

.subsections_via_symbols

#--- debug.s
.text
.globl _unref
_unref:
  retq

.globl _main
_main:
  retq

.subsections_via_symbols

#--- no-dead-symbols.s
.text
.globl _main
_main:
  retq

#--- literals.s
.cstring
_unref_foo:
  .ascii "foo"
_bar:
Lbar:
  .asciz "bar"
_unref_baz:
  .asciz "baz"

.literal4
.p2align 2
L._foo4:
  .long 0xdeadbeef
L._bar4:
  .long 0xdeadbeef
L._unref:
  .long 0xfeedface

.section __DATA,str_ptrs,literal_pointers
.globl _data
_data:
  .quad _bar
  .quad Lbar

## The output binary has these integer literals put into a section that isn't
## marked with a S_*BYTE_LITERALS flag, so we don't mark word_ptrs with the
## S_LITERAL_POINTERS flag in order not to confuse llvm-objdump.
.section __DATA,word_ptrs
.globl _more_data
_more_data:
  .quad L._foo4
  .quad L._bar4

.subsections_via_symbols

#--- ref-undef.s
.globl _main
_main:
  callq _ref_undef_fun
.subsections_via_symbols
