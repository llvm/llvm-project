# REQUIRES: x86

# RUN: rm -rf %t; split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/undefined-symbol.s -o %t/undefined-symbol.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/export-symbols.s -o %t/export-symbols.o

# RUN: not %lld %t/undefined-symbol.o -o /dev/null 2>&1 | FileCheck --check-prefix=UNDEF %s
# RUN: not %lld -demangle %t/undefined-symbol.o -o /dev/null 2>&1 | \
# RUN:     FileCheck --check-prefix=DEMANGLE-UNDEF %s

# RUN: not %lld -exported_symbol __ZTIN3foo3bar4MethE -exported_symbol __ZTSN3foo3bar4MethE %t/export-symbols.o -o /dev/null 2>&1 | FileCheck --check-prefix=EXPORT %s
# RUN: not %lld -demangle -exported_symbol __ZTIN3foo3bar4MethE -exported_symbol __ZTSN3foo3bar4MethE %t/export-symbols.o -o /dev/null 2>&1 | FileCheck --check-prefix=DEMANGLE-EXPORT %s

# UNDEF: undefined symbol: __Z1fv
# DEMANGLE-UNDEF: undefined symbol: f()

# EXPORT-DAG: cannot export hidden symbol __ZTSN3foo3bar4MethE
# EXPORT-DAG: cannot export hidden symbol __ZTIN3foo3bar4MethE

# DEMANGLE-EXPORT-DAG: cannot export hidden symbol typeinfo name for foo::bar::Meth
# DEMANGLE-EXPORT-DAG: cannot export hidden symbol typeinfo for foo::bar::Meth

#--- undefined-symbol.s
.globl _main
_main:
  callq __Z1fv
  ret


#--- export-symbols.s
.globl _main
_main:
  ret

.globl __ZTIN3foo3bar4MethE
.weak_def_can_be_hidden __ZTIN3foo3bar4MethE
.private_extern __ZTIN3foo3bar4MethE
__ZTIN3foo3bar4MethE:
  retq

.globl __ZTSN3foo3bar4MethE
.weak_def_can_be_hidden __ZTSN3foo3bar4MethE
.private_extern __ZTSN3foo3bar4MethE
__ZTSN3foo3bar4MethE:
  retq
