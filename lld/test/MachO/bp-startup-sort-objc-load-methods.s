# REQUIRES: aarch64

# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos11.0 %t/input.s -o %t/input.o
# RUN: llvm-profdata merge %t/profile.proftext -o %t/profile.profdata
# RUN: not %lld -arch arm64 -lSystem -e _main -o /dev/null %t/input.o --irpgo-profile=%t/profile.profdata --bp-startup-sort-objc-load-methods 2>&1 | FileCheck %s --check-prefix=ERROR
# RUN: %lld -arch arm64 -lSystem -e _main -o %t/out %t/input.o --irpgo-profile=%t/profile.profdata --bp-startup-sort=function --bp-startup-sort-objc-load-methods --verbose-bp-section-orderer 2> %t/verbose
# RUN: FileCheck %s --input-file=%t/verbose --check-prefix=VERBOSE
# RUN: llvm-nm --numeric-sort --format=just-symbols %t/out | FileCheck %s --check-prefix=ORDER

# ERROR: --bp-startup-sort-objc-load-methods must be used with --bp-startup-sort=function
# VERBOSE: Objective-C +load functions for startup: 2
# VERBOSE: Functions for startup: 3
# ORDER: _class_load
# ORDER-NEXT: _category_load
# ORDER-NEXT: _profile_hot
# ORDER-NEXT: _main

#--- input.s
 .subsections_via_symbols

 .text
 .globl _main
_main:
  ret

 .globl _profile_hot
_profile_hot:
  mov w0, #1
  ret

 .globl _class_load
_class_load:
  mov w0, #2
  ret

 .globl _category_load
_category_load:
  mov w0, #3
  ret

 .section __TEXT,__objc_methname,cstring_literals
L_load_name:
  .asciz "load"

 .section __TEXT,__objc_methtype,cstring_literals
L_method_type:
  .asciz "v16@0:8"

 .section __TEXT,__objc_classname,cstring_literals
L_class_name:
  .asciz "Foo"
L_category_name:
  .asciz "Category"

 .section __DATA,__objc_data
 .p2align 3
L_class_container:
  .space 8
  .quad L_metaclass_container + 8
  .quad 0
  .quad 0
  .quad 0
  .quad L_class_ro_container + 8

L_metaclass_container:
  .space 8
  .quad L_metaclass_container + 8
  .quad L_class_container + 8
  .quad 0
  .quad 0
  .quad L_metaclass_ro_container + 8

 .section __DATA,__objc_const
 .p2align 3
L_class_ro_container:
  .space 8
  .long 2
  .long 0
  .long 0
  .space 4
  .quad 0
  .quad L_class_name
  .quad 0
  .quad 0
  .quad 0
  .quad 0
  .quad 0

L_metaclass_ro_container:
  .space 8
  .long 3
  .long 40
  .long 40
  .space 4
  .quad 0
  .quad L_class_name
  .quad L_class_methods_container + 8
  .quad 0
  .quad 0
  .quad 0
  .quad 0

L_class_methods_container:
  .space 8
  .long 24
  .long 1
  .quad L_load_name
  .quad L_method_type
  .quad _class_load

L_category_container:
  .space 8
  .quad L_category_name
  .quad 0
  .quad 0
  .quad L_category_methods_container + 8
  .quad 0
  .quad 0
  .quad 0
  .long 64
  .space 4

L_category_methods_container:
  .space 8
  .long 0x8000000c
  .long 1
L_category_method_name:
  .long L_load_name - L_category_method_name
L_category_method_type:
  .long L_method_type - L_category_method_type
L_category_method_impl:
  .long _category_load - L_category_method_impl

 .section __DATA,__objc_nlclslist,regular,no_dead_strip
  .quad L_class_container + 8

 .section __DATA,__objc_nlcatlist,regular,no_dead_strip
  .quad L_category_container + 8

 .section __DATA,__objc_imageinfo,regular,no_dead_strip
  .long 0
  .long 64

#--- profile.proftext
:ir
:temporal_prof_traces
# Num Traces
1
# Trace Stream Size:
1
# Weight
1
profile_hot

profile_hot
# Func Hash:
1111
# Num Counters:
1
# Counter Values:
1
