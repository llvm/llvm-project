# REQUIRES: x86

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %t/main.s -o %t/main.o

## Check that -non_global_symbols_no_strip_list and -non_global_symbols_strip_list
## can't be used at the same time.
# RUN: not %lld %t/main.o -o /dev/null \
# RUN:       -non_global_symbols_no_strip_list %t/foo.txt \
# RUN:       -non_global_symbols_strip_list %t/foo.txt 2>&1 | \
# RUN:     FileCheck --check-prefix=CONFLICT %s

# CONFLICT: error: cannot use both -non_global_symbols_no_strip_list and -non_global_symbols_strip_list

## Check that -x causes none of the local symbols to be emitted.
# RUN: %lld %t/main.o -x -o %t/no_local.out
# RUN: llvm-nm %t/no_local.out | FileCheck --check-prefix NO_LOCAL %s

# NO_LOCAL-NOT: t _foo
# NO_LOCAL-NOT: t _bar
# NO_LOCAL-NOT: t _baz
# NO_LOCAL: T _main

## Check that when using -x with -non_global_symbols_no_strip_list, whichever appears
## last in the command line arg list will take precedence.
# RUN: %lld %t/main.o -x -non_global_symbols_no_strip_list %t/foo.txt -o %t/x_then_no_strip.out
# RUN: llvm-nm %t/x_then_no_strip.out | FileCheck --check-prefix X-NO-STRIP %s

# RUN: %lld %t/main.o -non_global_symbols_no_strip_list %t/foo.txt -x -o %t/no_strip_then_x.out
# RUN: llvm-nm %t/no_strip_then_x.out | FileCheck --check-prefix NO_LOCAL %s

# X-NO-STRIP-NOT: t _bar
# X-NO-STRIP-DAG: t _foo
# X-NO-STRIP-DAG: T _main

## Check that -non_global_symbols_no_strip_list can be specified more than once
## (The final no-strip list is the union of all these)
# RUN: %lld %t/main.o -o %t/no_strip_multi.out \
# RUN:    -non_global_symbols_no_strip_list %t/foo.txt \
# RUN:    -non_global_symbols_no_strip_list %t/bar.txt
# RUN: llvm-nm %t/no_strip_multi.out | FileCheck --check-prefix NO-STRIP-MULTI %s

# NO-STRIP-MULTI-NOT: t _baz
# NO-STRIP-MULTI-DAG: t _foo
# NO-STRIP-MULTI-DAG: t _bar
# NO-STRIP-MULTI-DAG: T _main

## Check that when using -x with -non_global_symbols_strip_list, whichever appears
## last in the command line arg list will take precedence.
# RUN: %lld %t/main.o -x -non_global_symbols_strip_list %t/foo.txt -o %t/x_then_strip.out
# RUN: llvm-nm %t/x_then_strip.out | FileCheck --check-prefix X-STRIP %s

# RUN: %lld %t/main.o -non_global_symbols_strip_list %t/foo.txt -x -o %t/strip_then_x.out
# RUN: llvm-nm %t/no_strip_then_x.out | FileCheck --check-prefix NO_LOCAL %s

# X-STRIP-NOT: t _foo
# X-STRIP-DAG: t _bar
# X-STRIP-DAG: t _baz
# X-STRIP-DAG: T _main

## Check that -non_global_symbols_strip_list can be specified more than once
## (The final strip list is the union of all these)
# RUN: %lld %t/main.o -o %t/strip_multi.out \
# RUN:    -non_global_symbols_strip_list %t/foo.txt \
# RUN:    -non_global_symbols_strip_list %t/bar.txt
# RUN: llvm-nm %t/strip_multi.out | FileCheck --check-prefix STRIP-MULTI %s

# STRIP-MULTI-NOT: t _foo
# STRIP-MULTI-NOT: t _bar
# STRIP-MULTI-DAG: t _baz
# STRIP-MULTI-DAG: T _main

## Test interactions with exported_symbol.
# RUN: %lld %t/main.o -o %t/strip_all_export_one.out \
# RUN:    -x -exported_symbol _foo \
# RUN:    -undefined dynamic_lookup
# RUN: llvm-nm %t/strip_all_export_one.out | FileCheck --check-prefix STRIP-EXP %s

# STRIP-EXP: U _foo
# STRIP-EXP: U dyld_stub_binder
# STRIP-EXP-EMPTY:

## Test interactions of -x and -non_global_symbols_strip_list with unexported_symbol.
# RUN: %lld %t/main.o -o %t/strip_x_unexport_one.out \
# RUN:    -x -unexported_symbol _globby \
# RUN:    -undefined dynamic_lookup

# RUN: %lld %t/main.o -o %t/strip_all_unexport_one.out \
# RUN:    -non_global_symbols_strip_list %t/globby.txt \
# RUN:    -non_global_symbols_strip_list %t/foo.txt \
# RUN:    -non_global_symbols_strip_list %t/bar.txt \
# RUN:    -unexported_symbol _globby \
# RUN:    -undefined dynamic_lookup

# RUN: llvm-nm %t/strip_x_unexport_one.out | FileCheck --check-prefix STRIP-UNEXP %s
# RUN: llvm-nm %t/strip_all_unexport_one.out | FileCheck --check-prefix STRIP-UNEXP %s

## -unexported_symbol made _globby a local, therefore it should be stripped by -x too
# STRIP-UNEXP: T __mh_execute_header
# STRIP-UNEXP-DAG: T _main
# STRIP-UNEXP-DAG: U dyld_stub_binder
# STRIP-UNEXP-EMPTY:

## Test interactions of -non_global_symbols_strip_list and unexported_symbol.
# RUN: %lld %t/main.o -undefined dynamic_lookup -o %t/no_strip_unexport.out \
# RUN:    -non_global_symbols_no_strip_list %t/globby.txt \
# RUN:    -unexported_symbol _globby

# RUN: llvm-nm %t/no_strip_unexport.out | FileCheck --check-prefix NOSTRIP-UNEXP %s

# NOSTRIP-UNEXP: T __mh_execute_header
# NOSTRIP-UNEXP-DAG: T _main
# NOSTRIP-UNEXP-DAG: t _globby
# NOSTRIP-UNEXP-DAG: U dyld_stub_binder
# NOSTRIP-UNEXP-EMPTY:

#--- foo.txt
_foo

#--- bar.txt
_bar

#--- globby.txt
_globby

#--- main.s
.globl _main
.globl _globby

_foo:
  ret

_bar:
  ret

_baz:
  ret

_main:
  callq _foo
  ret

 _globby:
  ret
