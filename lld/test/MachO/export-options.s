# REQUIRES: x86

# RUN: rm -rf %t; split-file %s %t
# RUN: echo "" | llvm-mc -filetype=obj -triple=x86_64-apple-macos -o %t/empty.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %t/default.s -o %t/default.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %t/lazydef.s -o %t/lazydef.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %t/too-many-warnings.s -o %t/too-many-warnings.o
# RUN: llvm-ar --format=darwin rcs %t/lazydef.a %t/lazydef.o

## Check that mixing exported and unexported symbol options yields an error
# RUN: not %lld -dylib %t/default.o -o /dev/null \
# RUN:         -exported_symbol a -unexported_symbol b 2>&1 | \
# RUN:     FileCheck --check-prefix=CONFLICT %s
#
# RUN: not %lld -dylib %t/default.o -o /dev/null \
# RUN:         -exported_symbols_list /dev/null -unexported_symbol b 2>&1 | \
# RUN:     FileCheck --check-prefix=CONFLICT %s
#
# RUN: not %lld -dylib %t/default.o -o /dev/null \
# RUN:         -no_exported_symbols -unexported_symbol b 2>&1 | \
# RUN:     FileCheck --check-prefix=CONFLICT %s
#
# RUN: not %lld -dylib %t/default.o -o /dev/null \
# RUN:         -no_exported_symbols -exported_symbol b 2>&1 | \
# RUN:     FileCheck --check-prefix=CONFLICT-NO-EXPORTS %s
#
# RUN: not %lld -dylib %t/default.o -o /dev/null \
# RUN:         -no_exported_symbols -exported_symbols_list %t/literals.txt 2>&1 | \
# RUN:     FileCheck --check-prefix=CONFLICT-NO-EXPORTS %s

# CONFLICT: error: cannot use both -exported_symbol* and -unexported_symbol* options

# CONFLICT-NO-EXPORTS: error: cannot use both -exported_symbol* and -no_exported_symbols options

## Check that an exported literal name with no symbol definition yields an error
## but that an exported glob-pattern with no matching symbol definition is OK
# RUN: not %lld -dylib %t/default.o -o /dev/null \
# RUN:         -exported_symbol absent_literal \
# RUN:         -exported_symbol absent_gl?b 2>&1 | \
# RUN:     FileCheck --check-prefix=UNDEF %s

# UNDEF: error: undefined symbol: absent_literal
# UNDEF-NEXT: >>> referenced by -exported_symbol(s_list)
# UNDEF-NOT: error: {{.*}} absent_gl{{.}}b

## Check that dynamic_lookup suppresses the error
# RUN: %lld -dylib %t/default.o -undefined dynamic_lookup -o %t/dyn-lookup \
# RUN:      -exported_symbol absent_literal
# RUN: llvm-objdump --macho --syms %t/dyn-lookup | FileCheck %s --check-prefix=DYN
# DYN: *UND* absent_literal

## Check that exported literal symbols are present in output's
## symbol table, even lazy symbols which would otherwise be omitted
# RUN: %lld -dylib %t/default.o %t/lazydef.a -o %t/lazydef \
# RUN:         -exported_symbol _keep_globl \
# RUN:         -exported_symbol _keep_lazy
# RUN: llvm-objdump --syms %t/lazydef | \
# RUN:     FileCheck --check-prefix=EXPORT %s

# EXPORT-DAG: g     F __TEXT,__text _keep_globl
# EXPORT-DAG: g     F __TEXT,__text _keep_lazy

## Check that exported symbol is global
# RUN: %no-fatal-warnings-lld -dylib %t/default.o -o %t/hidden-export \
# RUN:         -exported_symbol _private_extern 2>&1 | \
# RUN:     FileCheck --check-prefix=PRIVATE %s

# PRIVATE: warning: cannot export hidden symbol _private_extern

## Check that we still hide the other symbols despite the warning
# RUN: llvm-objdump --macho --exports-trie %t/hidden-export | \
# RUN:     FileCheck --check-prefix=EMPTY-TRIE %s
# EMPTY-TRIE:       Exports trie:
# EMPTY-TRIE-EMPTY:

## Check that the export trie is unaltered
# RUN: %lld -dylib %t/default.o -o %t/default
# RUN: llvm-objdump --macho --exports-trie %t/default | \
# RUN:     FileCheck --check-prefix=DEFAULT %s

# DEFAULT-LABEL: Exports trie:
# DEFAULT-DAG:   _hide_globl
# DEFAULT-DAG:   _keep_globl
# DEFAULT-NOT:   _private_extern

## Check that the export trie is shaped by an allow list and then
## by a deny list. Both lists are designed to yield the same result.

## Check the allow list
# RUN: %lld -dylib %t/default.o -o %t/allowed \
# RUN:     -exported_symbol _keep_globl
# RUN: llvm-objdump --macho --exports-trie %t/allowed | \
# RUN:     FileCheck --check-prefix=TRIE %s
# RUN: llvm-nm -m %t/allowed | \
# RUN:     FileCheck --check-prefix=NM %s

## Check the deny list
# RUN: %lld -dylib %t/default.o -o %t/denied \
# RUN:     -unexported_symbol _hide_globl
# RUN: llvm-objdump --macho --exports-trie %t/denied | \
# RUN:     FileCheck --check-prefix=TRIE %s
# RUN: llvm-nm -m %t/denied | \
# RUN:     FileCheck --check-prefix=NM %s

# TRIE-LABEL: Exports trie:
# TRIE-DAG:   _keep_globl
# TRIE-NOT:   _hide_globl
# TRIE-NOT:   _private_extern

# NM-DAG: external _keep_globl
# NM-DAG: non-external (was a private external) _hide_globl
# NM-DAG: non-external (was a private external) _private_extern

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos \
# RUN:     %t/symdefs.s -o %t/symdefs.o

## Check that only string-literal patterns match
## Check that comments and blank lines are stripped from symbol list
# RUN: %lld -dylib %t/symdefs.o -o %t/literal \
# RUN:     -exported_symbols_list %t/literals.txt
# RUN: llvm-objdump --macho --exports-trie %t/literal | \
# RUN:     FileCheck --check-prefix=LITERAL %s

# LITERAL-DAG: literal_only
# LITERAL-DAG: literal_also
# LITERAL-DAG: globby_also
# LITERAL-NOT: globby_only

## Check that only glob patterns match
## Check that comments and blank lines are stripped from symbol list
# RUN: %lld -dylib %t/symdefs.o -o %t/globby \
# RUN:     -exported_symbols_list %t/globbys.txt
# RUN: llvm-objdump --macho --exports-trie %t/globby | \
# RUN:     FileCheck --check-prefix=GLOBBY %s

# GLOBBY-DAG: literal_also
# GLOBBY-DAG: globby_only
# GLOBBY-DAG: globby_also
# GLOBBY-NOT: literal_only

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos \
# RUN:     %t/autohide.s -o %t/autohide.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos \
# RUN:     %t/autohide-private-extern.s -o %t/autohide-private-extern.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos \
# RUN:     %t/glob-private-extern.s -o %t/glob-private-extern.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos \
# RUN:     %t/weak-private-extern.s -o %t/weak-private-extern.o
## Test that we can export the autohide symbol but not when it's also
## private-extern
# RUN: %lld -dylib -exported_symbol "_foo" %t/autohide.o -o %t/exp-autohide.dylib
# RUN: llvm-nm -g %t/exp-autohide.dylib | FileCheck %s --check-prefix=EXP-AUTOHIDE

# RUN: not %lld -dylib -exported_symbol "_foo" %t/autohide-private-extern.o \
# RUN:   -o /dev/null  2>&1 | FileCheck %s --check-prefix=AUTOHIDE-PRIVATE

# RUN: not %lld -dylib -exported_symbol "_foo" %t/autohide.o \
# RUN:   %t/glob-private-extern.o -o /dev/null 2>&1 | \
# RUN:   FileCheck %s --check-prefix=AUTOHIDE-PRIVATE

# RUN: not %lld -dylib -exported_symbol "_foo" %t/autohide.o \
# RUN:   %t/weak-private-extern.o -o /dev/null 2>&1 | \
# RUN:   FileCheck %s --check-prefix=AUTOHIDE-PRIVATE

## Test that exported hidden symbols are still treated as a liveness root.
## This previously used to crash when enabling -dead_strip since it's unconventional
## to add treat private extern symbols as a liveness root.
# RUN: %no-fatal-warnings-lld -dylib -exported_symbol "_foo" %t/autohide-private-extern.o \
# RUN:   -dead_strip -o %t/exported-hidden
# RUN: llvm-nm -m %t/exported-hidden | FileCheck %s --check-prefix=AUTOHIDE-PRIVATE-DEAD-STRIP

# RUN: %no-fatal-warnings-lld -dylib -exported_symbol "_foo" %t/autohide.o \
# RUN:   -dead_strip %t/glob-private-extern.o -o %t/exported-hidden
# RUN: llvm-nm -m %t/exported-hidden | FileCheck %s --check-prefix=AUTOHIDE-PRIVATE-DEAD-STRIP

# RUN: %no-fatal-warnings-lld -dylib -exported_symbol "_foo" %t/autohide.o \
# RUN:   -dead_strip %t/weak-private-extern.o -o %t/exported-hidden
# RUN: llvm-nm -m %t/exported-hidden | FileCheck %s --check-prefix=AUTOHIDE-PRIVATE-DEAD-STRIP

# EXP-AUTOHIDE: T _foo
# AUTOHIDE-PRIVATE: error: cannot export hidden symbol _foo
# AUTOHIDE-PRIVATE-DEAD-STRIP: (__TEXT,__text) non-external (was a private external) _foo

## Test not exporting any symbols
# RUN: %lld -dylib %t/symdefs.o -o %t/noexports -exported_symbols_list /dev/null
# RUN: llvm-objdump --macho --exports-trie %t/noexports | FileCheck --check-prefix=NOEXPORTS %s
# RUN: %lld -dylib %t/symdefs.o -o %t/noexports -no_exported_symbols
# RUN: llvm-objdump --macho --exports-trie %t/noexports | FileCheck --check-prefix=NOEXPORTS %s

# NOEXPORTS-NOT: globby_also
# NOEXPORTS-NOT: globby_only
# NOEXPORTS-NOT: literal_also
# NOEXPORTS-NOT: literal_only

# RUN: %lld -dylib %t/default.o -o %t/libdefault.dylib
# RUN: %lld -dylib %t/empty.o %t/libdefault.dylib -exported_symbol _keep_globl \
# RUN:   -exported_symbol _undef -exported_symbol _tlv \
# RUN:   -undefined dynamic_lookup -o %t/reexport-dylib
# RUN: llvm-objdump --macho --exports-trie %t/reexport-dylib

# REEXPORT:      Exports trie:
# REEXPORT-NEXT: [re-export] _tlv [per-thread] (from libdefault)
# REEXPORT-NEXT: [re-export] _keep_globl (from libdefault)
# REEXPORT-NEXT: [re-export] _undef (from unknown)

## -unexported_symbol will not make us re-export symbols in dylibs.
# RUN: %lld -dylib %t/default.o -o %t/libdefault.dylib
# RUN: %lld -dylib %t/empty.o %t/libdefault.dylib -unexported_symbol _tlv \
# RUN:   -o %t/unexport-dylib
# RUN: llvm-objdump --macho --exports-trie %t/unexport-dylib | FileCheck %s \
# RUN:   --check-prefix=EMPTY-TRIE

## Check that warnings are truncated to the first 3 only.
# RUN: %no-fatal-warnings-lld -dylib %t/too-many-warnings.o -o %t/too-many.out \
# RUN:         -exported_symbol "_private_extern*" 2>&1 | \
# RUN:     FileCheck --check-prefix=TRUNCATE %s

# TRUNCATE: warning: cannot export hidden symbol _private_extern{{.+}}
# TRUNCATE: warning: cannot export hidden symbol _private_extern{{.+}}
# TRUNCATE: warning: cannot export hidden symbol _private_extern{{.+}}
# TRUNCATE: warning: <... 7 more similar warnings...>
# TRUNCATE-EMPTY:

#--- default.s

.globl _keep_globl, _hide_globl, _tlv
_keep_globl:
  retq
_hide_globl:
  retq
.private_extern _private_extern
_private_extern:
  retq
_private:
  retq

.section __DATA,__thread_vars,thread_local_variables
_tlv:

#--- lazydef.s

.globl _keep_lazy
_keep_lazy:
  retq

#--- symdefs.s

.globl literal_only, literal_also, globby_only, globby_also
literal_only:
  retq
literal_also:
  retq
globby_only:
  retq
globby_also:
  retq

#--- literals.txt

  literal_only # comment
  literal_also

# globby_only
  globby_also

#--- globbys.txt

# literal_only
  l?ter[aeiou]l_*[^y] # comment

  *gl?bby_*

#--- autohide.s
.globl _foo
.weak_def_can_be_hidden _foo
_foo:
  retq

#--- autohide-private-extern.s
.globl _foo
.weak_def_can_be_hidden _foo
.private_extern _foo
_foo:
  retq

#--- glob-private-extern.s
.global _foo
.private_extern _foo
_foo:
  retq

#--- weak-private-extern.s
.global _foo
.weak_definition _foo
.private_extern _foo
_foo:
  retq

#--- too-many-warnings.s
.private_extern _private_extern1
.private_extern _private_extern2
.private_extern _private_extern3
.private_extern _private_extern4
.private_extern _private_extern5
.private_extern _private_extern6
.private_extern _private_extern7
.private_extern _private_extern8
.private_extern _private_extern9
.private_extern _private_extern10

_private_extern1:
  retq

_private_extern2:
  retq

_private_extern3:
  retq

_private_extern4:
  retq

_private_extern5:
  retq

_private_extern6:
  retq

_private_extern7:
  retq

_private_extern8:
  retq

_private_extern9:
  retq

_private_extern10:
  retq
