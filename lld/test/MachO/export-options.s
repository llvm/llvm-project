# REQUIRES: x86

# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %t/default.s -o %t/default.o

## Check that mixing exported and unexported symbol options yields an error
# RUN: not %lld -dylib %t/default.o -o /dev/null \
# RUN:         -exported_symbol a -unexported_symbol b 2>&1 | \
# RUN:     FileCheck --check-prefix=CONFLICT %s

# CONFLICT: error: cannot use both -exported_symbol* and -unexported_symbol* options
# CONFLICT-NEXT: >>> ignoring unexports

#--- default.s

.macro DEFSYM, type, sym
\type \sym
\sym:
  retq
.endm

DEFSYM .globl, _keep_globl
DEFSYM .globl, _hide_globl
DEFSYM .private_extern, _keep_private
DEFSYM .private_extern, _show_private

## Check that the export trie is unaltered
# RUN: %lld -dylib %t/default.o -o %t/default
# RUN: llvm-objdump --macho --exports-trie %t/default | \
# RUN:     FileCheck --check-prefix=DEFAULT %s

# DEFAULT-LABEL: Exports trie:
# DEFAULT-DAG:   _hide_globl
# DEFAULT-DAG:   _keep_globl
# DEFAULT-NOT:   _hide_private
# DEFAULT-NOT:   _show_private

## Check that the export trie is properly augmented
## Check that non-matching literal pattern has no effect
# RUN: %lld -dylib %t/default.o -o %t/export \
# RUN:     -exported_symbol _show_private \
# RUN:     -exported_symbol _extra_cruft -exported_symbol '*xtra_cr?ft'
# RUN: llvm-objdump --macho --exports-trie %t/export | \
# RUN:     FileCheck --check-prefix=EXPORTED %s

# EXPORTED-LABEL: Exports trie:
# EXPORTED-DAG:   _show_private
# EXPORTED-NOT:   _hide_globl
# EXPORTED-NOT:   _keep_globl
# EXPORTED-NOT:   _hide_private
# EXPORTED-NOT:   {{.*}}xtra_cr{{.}}ft

## Check that the export trie is properly diminished
## Check that non-matching glob pattern has no effect
# RUN: %lld -dylib %t/default.o -o %t/unexport \
# RUN:     -unexported_symbol _hide_global
# RUN: llvm-objdump --macho --exports-trie %t/unexport | \
# RUN:     FileCheck --check-prefix=UNEXPORTED %s

# UNEXPORTED-LABEL: Exports trie:
# UNEXPORTED-DAG:   _keep_globl
# UNEXPORTED-NOT:   _hide_globl
# UNEXPORTED-NOT:   _show_private
# UNEXPORTED-NOT:   _hide_private

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos \
# RUN:     %t/symdefs.s -o %t/symdefs.o

#--- symdefs.s

.macro DEFSYM, sym
.private_extern \sym
\sym:
  retq
.endm

DEFSYM literal_only
DEFSYM literal_also
DEFSYM globby_only
DEFSYM globby_also

#--- literals

  literal_only # comment
  literal_also

# globby_only
  globby_also

## Check that only string-literal patterns match
## Check that comments and blank lines are stripped from symbol list
# RUN: %lld -dylib %t/symdefs.o -o %t/literal \
# RUN:     -exported_symbols_list %t/literals
# RUN: llvm-objdump --macho --exports-trie %t/literal | \
# RUN:     FileCheck --check-prefix=LITERAL %s

# LITERAL-DAG: literal_only
# LITERAL-DAG: literal_also
# LITERAL-DAG: globby_also
# LITERAL-NOT: globby_only

#--- globbys

# literal_only
  l?ter[aeiou]l_*[^y] # comment

  *gl?bby_*

## Check that only glob patterns match
## Check that comments and blank lines are stripped from symbol list
# RUN: %lld -dylib %t/symdefs.o -o %t/globby \
# RUN:     -exported_symbols_list %t/globbys
# RUN: llvm-objdump --macho --exports-trie %t/globby | \
# RUN:     FileCheck --check-prefix=GLOBBY %s

# GLOBBY-DAG: literal_also
# GLOBBY-DAG: globby_only
# GLOBBY-DAG: globby_also
# GLOBBY-NOT: literal_only
