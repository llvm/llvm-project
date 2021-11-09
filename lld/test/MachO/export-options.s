# REQUIRES: x86

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %t/default.s -o %t/default.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %t/lazydef.s -o %t/lazydef.o
# RUN: llvm-ar --format=darwin rcs %t/lazydef.a %t/lazydef.o

## Check that mixing exported and unexported symbol options yields an error
# RUN: not %lld -dylib %t/default.o -o /dev/null \
# RUN:         -exported_symbol a -unexported_symbol b 2>&1 | \
# RUN:     FileCheck --check-prefix=CONFLICT %s

# CONFLICT: error: cannot use both -exported_symbol* and -unexported_symbol* options
# CONFLICT-NEXT: >>> ignoring unexports

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
# RUN: %no_fatal_warnings_lld -dylib %t/default.o -o %t/hidden-export \
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
# RUN: -o /dev/null  2>&1 | FileCheck %s --check-prefix=AUTOHIDE-PRIVATE

# RUN: not %lld -dylib -exported_symbol "_foo" %t/autohide.o \
# RUN:   %t/glob-private-extern.o -o /dev/null 2>&1 | \
# RUN:   FileCheck %s --check-prefix=AUTOHIDE-PRIVATE

# RUN: not %lld -dylib -exported_symbol "_foo" %t/autohide.o \
# RUN:   %t/weak-private-extern.o -o /dev/null 2>&1 | \
# RUN:   FileCheck %s --check-prefix=AUTOHIDE-PRIVATE

# EXP-AUTOHIDE: T _foo        
# AUTOHIDE-PRIVATE: error: cannot export hidden symbol _foo
        
#--- default.s

.globl _keep_globl, _hide_globl
_keep_globl:
  retq
_hide_globl:
  retq
.private_extern _private_extern
_private_extern:
  retq
_private:
  retq

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
