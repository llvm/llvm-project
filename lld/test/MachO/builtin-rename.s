# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin \
# RUN:     %t/main.s -o %t/main.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin \
# RUN:     %t/renames.s -o %t/renames.o

## Check that section and segment renames happen as expected
# RUN: %lld                -o %t/ydata %t/main.o %t/renames.o -lSystem
# RUN: %lld -no_data_const -o %t/ndata %t/main.o %t/renames.o -lSystem
# RUN: %lld -no_pie        -o %t/nopie %t/main.o %t/renames.o -lSystem
# RUN: %lld -platform_version macos 10.14 11.0 -o %t/old %t/main.o %t/renames.o -lSystem

# RUN: llvm-objdump --syms %t/ydata | \
# RUN:     FileCheck %s --check-prefixes=CHECK,YDATA
# RUN: llvm-objdump --syms %t/ndata | \
# RUN:     FileCheck %s --check-prefixes=CHECK,NDATA
# RUN: llvm-objdump --syms %t/nopie | \
# RUN:     FileCheck %s --check-prefixes=CHECK,NDATA
# RUN: llvm-objdump --syms %t/old | \
# RUN:     FileCheck %s --check-prefixes=CHECK,NDATA

# CHECK-LABEL: {{^}}SYMBOL TABLE:

# CHECK-DAG: __TEXT,__text __TEXT__StaticInit

# NDATA-DAG: __DATA,__auth_got __DATA__auth_got
# NDATA-DAG: __DATA,__auth_ptr __DATA__auth_ptr
# NDATA-DAG: __DATA,__nl_symbol_ptr __DATA__nl_symbol_ptr
# NDATA-DAG: __DATA,__const __DATA__const
# NDATA-DAG: __DATA,__cfstring __DATA__cfstring
# NDATA-DAG: __DATA,__mod_init_func __DATA__mod_init_func
# NDATA-DAG: __DATA,__mod_term_func __DATA__mod_term_func
# NDATA-DAG: __DATA,__objc_classlist __DATA__objc_classlist
# NDATA-DAG: __DATA,__objc_nlclslist __DATA__objc_nlclslist
# NDATA-DAG: __DATA,__objc_catlist __DATA__objc_catlist
# NDATA-DAG: __DATA,__objc_nlcatlist __DATA__objc_nlcatlist
# NDATA-DAG: __DATA,__objc_protolist __DATA__objc_protolist
# NDATA-DAG: __DATA,__objc_imageinfo __DATA__objc_imageinfo
# NDATA-DAG: __DATA,__nl_symbol_ptr __IMPORT__pointers

# YDATA-DAG: __DATA_CONST,__auth_got __DATA__auth_got
# YDATA-DAG: __DATA_CONST,__auth_ptr __DATA__auth_ptr
# YDATA-DAG: __DATA_CONST,__nl_symbol_ptr __DATA__nl_symbol_ptr
# YDATA-DAG: __DATA_CONST,__const __DATA__const
# YDATA-DAG: __DATA_CONST,__cfstring __DATA__cfstring
# YDATA-DAG: __DATA_CONST,__mod_init_func __DATA__mod_init_func
# YDATA-DAG: __DATA_CONST,__mod_term_func __DATA__mod_term_func
# YDATA-DAG: __DATA_CONST,__objc_classlist __DATA__objc_classlist
# YDATA-DAG: __DATA_CONST,__objc_nlclslist __DATA__objc_nlclslist
# YDATA-DAG: __DATA_CONST,__objc_catlist __DATA__objc_catlist
# YDATA-DAG: __DATA_CONST,__objc_nlcatlist __DATA__objc_nlcatlist
# YDATA-DAG: __DATA_CONST,__objc_protolist __DATA__objc_protolist
# YDATA-DAG: __DATA_CONST,__objc_imageinfo __DATA__objc_imageinfo
# YDATA-DAG: __DATA_CONST,__nl_symbol_ptr __IMPORT__pointers

## LLD doesn't support defining symbols in synthetic sections, so we test them
## via this slightly more awkward route.
# RUN: llvm-readobj --section-headers %t/ydata | \
# RUN:     FileCheck %s --check-prefix=SYNTH -DSEGNAME=__DATA_CONST
# RUN: llvm-readobj --section-headers %t/ndata | \
# RUN:     FileCheck %s --check-prefix=SYNTH -DSEGNAME=__DATA
# RUN: llvm-readobj --section-headers %t/nopie | \
# RUN:     FileCheck %s --check-prefix=SYNTH -DSEGNAME=__DATA
# RUN: llvm-readobj --section-headers %t/old | \
# RUN:     FileCheck %s --check-prefix=SYNTH -DSEGNAME=__DATA

# SYNTH:      Name: __got
# SYNTH-NEXT: Segment: [[SEGNAME]] ({{.*}})
## Note that __la_symbol_ptr always remains in the non-const data segment.
# SYNTH:      Name: __la_symbol_ptr
# SYNTH-NEXT: Segment: __DATA ({{.*}})

#--- renames.s
.section __DATA,__auth_got
.global __DATA__auth_got
__DATA__auth_got:
  .space 8

.section __DATA,__auth_ptr
.global __DATA__auth_ptr
__DATA__auth_ptr:
  .space 8

.section __DATA,__nl_symbol_ptr
.global __DATA__nl_symbol_ptr
__DATA__nl_symbol_ptr:
  .space 8

.section __DATA,__const
.global __DATA__const
__DATA__const:
  .space 8

.section __DATA,__cfstring
.global __DATA__cfstring
__DATA__cfstring:
  .space 8

.section __DATA,__mod_init_func,mod_init_funcs
.global __DATA__mod_init_func
__DATA__mod_init_func:
  .space 8

.section __DATA,__mod_term_func,mod_term_funcs
.global __DATA__mod_term_func
__DATA__mod_term_func:
  .space 8

.section __DATA,__objc_classlist
.global __DATA__objc_classlist
__DATA__objc_classlist:
  .space 8

.section __DATA,__objc_nlclslist
.global __DATA__objc_nlclslist
__DATA__objc_nlclslist:
  .space 8

.section __DATA,__objc_catlist
.global __DATA__objc_catlist
__DATA__objc_catlist:
  .space 8

.section __DATA,__objc_nlcatlist
.global __DATA__objc_nlcatlist
__DATA__objc_nlcatlist:
  .space 8

.section __DATA,__objc_protolist
.global __DATA__objc_protolist
__DATA__objc_protolist:
  .space 8

.section __DATA,__objc_imageinfo
.global __DATA__objc_imageinfo
__DATA__objc_imageinfo:
  .space 8

.section __IMPORT,__pointers,non_lazy_symbol_pointers
.global __IMPORT__pointers
__IMPORT__pointers:
  .space 8

.section __TEXT,__StaticInit
.section __TEXT,__StaticInit
.global __TEXT__StaticInit
__TEXT__StaticInit:
  .space 8

#--- main.s
.text
.global _main
_main:
  mov ___nan@GOTPCREL(%rip), %rax ## ensure the __got section is created
  callq ___isnan ## ensure the __la_symbol_ptr section is created
  ret
