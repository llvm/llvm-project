# REQUIRES: x86

# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %t/default.s -o %t/default.o

## Check that mixing exported and unexported symbol options yields an error
# RUN: not %lld -dylib %t/default.o -o /dev/null \
# RUN:         -exported_symbol a -unexported_symbol b 2>&1 | \
# RUN:     FileCheck --check-prefix=CONFLICT %s

# CONFLICT: error: cannot use both -exported_symbol* and -unexported_symbol* options
# CONFLICT-NEXT: >>> ignoring unexports

## Check that exported literal symbol name is present in symbol table
# RUN: not %lld -dylib %t/default.o -o /dev/null \
# RUN:         -exported_symbol absent_literal \
# RUN:         -exported_symbol absent_gl?b 2>&1 | \
# RUN:     FileCheck --check-prefix=UNDEF %s

# UNDEF: error: undefined symbol absent_literal
# UNDEF-NEXT: >>> referenced from option -exported_symbol(s_list)
# UNDEF-NOT: error: {{.*}} absent_gl{{.}}b

## Check that exported symbol is global
# RUN: not %lld -dylib %t/default.o -o /dev/null \
# RUN:         -exported_symbol _private_extern 2>&1 | \
# RUN:     FileCheck --check-prefix=PRIVATE %s

# PRIVATE: error: cannot export hidden symbol _private_extern

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

#--- globbys.txt

# literal_only
  l?ter[aeiou]l_*[^y] # comment

  *gl?bby_*

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
