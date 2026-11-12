# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos -o %t/objc.o %t/objc.s
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos -o %t/foo.o %t/foo.s
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos -o %t/bar.o %t/bar.s
# RUN: llvm-ar csr  %t/lib.a %t/objc.o %t/foo.o %t/bar.o

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos -o %t/main.o %t/main.s

# The first line checks that we never demangle symbols in -why_load output.
# RUN: %lld %t/main.o %t/lib.a -o /dev/null -why_load -demangle | \
# RUN:     FileCheck %s --check-prefix=WHY
# RUN: %lld %t/main.o -force_load %t/lib.a -o /dev/null -whyload | \
# RUN:     FileCheck %s --check-prefix=WHYFORCE
# RUN: %lld %t/main.o %t/lib.a -o /dev/null -all_load -why_load | \
# RUN:     FileCheck %s --check-prefix=WHYALL
# RUN: %lld %t/main.o -force_load %t/lib.a -o /dev/null -all_load -why_load | \
# RUN:     FileCheck %s --check-prefix=WHYALLFORCE

# RUN: %lld %t/main.o %t/lib.a -o /dev/null -ObjC -why_load | \
# RUN:     FileCheck %s --check-prefix=WHYOBJC
# RUN: %lld %t/main.o -force_load %t/lib.a -o /dev/null -ObjC -why_load | \
# RUN:     FileCheck %s --check-prefix=WHYOBJCFORCE
# RUN: %lld %t/main.o %t/lib.a -o /dev/null -ObjC -all_load -why_load | \
# RUN:     FileCheck %s --check-prefix=WHYOBJCALL
# RUN: %lld %t/main.o -force_load %t/lib.a -o /dev/null -ObjC -all_load -why_load | \
# RUN:     FileCheck %s --check-prefix=WHYOBJCALLFORCE

# WHY-DAG: _bar forced load of {{.+}}lib.a(bar.o)
# WHY-DAG: __Z3foo forced load of {{.+}}lib.a(foo.o)
# WHY-NOT: {{.+}} forced load of {{.+}}lib.a(objc.o)

# WHYFORCE-DAG: -force_load forced load of {{.+}}lib.a(bar.o)
# WHYFORCE-DAG: -force_load forced load of {{.+}}lib.a(foo.o)
# WHYFORCE-DAG: -force_load forced load of {{.+}}lib.a(objc.o)

# WHYALL-DAG: -all_load forced load of {{.+}}lib.a(bar.o)
# WHYALL-DAG: -all_load forced load of {{.+}}lib.a(foo.o)
# WHYALL-DAG: -all_load forced load of {{.+}}lib.a(objc.o)

# WHYALLFORCE-DAG: -force_load forced load of {{.+}}lib.a(bar.o)
# WHYALLFORCE-DAG: -force_load forced load of {{.+}}lib.a(foo.o)
# WHYALLFORCE-DAG: -force_load forced load of {{.+}}lib.a(objc.o)

# WHYOBJC-DAG: _bar forced load of {{.+}}lib.a(bar.o)
# WHYOBJC-DAG: __Z3foo forced load of {{.+}}lib.a(foo.o)
# WHYOBJC-DAG: -ObjC forced load of {{.+}}lib.a(objc.o)

# WHYOBJCFORCE-DAG: -force_load forced load of {{.+}}lib.a(bar.o)
# WHYOBJCFORCE-DAG: -force_load forced load of {{.+}}lib.a(foo.o)
# WHYOBJCFORCE-DAG: -force_load forced load of {{.+}}lib.a(objc.o)

# WHYOBJCALL-DAG: -all_load forced load of {{.+}}lib.a(bar.o)
# WHYOBJCALL-DAG: -all_load forced load of {{.+}}lib.a(foo.o)
# WHYOBJCALL-DAG: -all_load forced load of {{.+}}lib.a(objc.o)

# WHYOBJCALLFORCE-DAG: -force_load forced load of {{.+}}lib.a(bar.o)
# WHYOBJCALLFORCE-DAG: -force_load forced load of {{.+}}lib.a(foo.o)
# WHYOBJCALLFORCE-DAG: -force_load forced load of {{.+}}lib.a(objc.o)

#--- objc.s
.section __DATA,__objc_catlist
.quad 0x1234

#--- foo.s
.globl __Z3foo
__Z3foo:
  ret

#--- bar.s
.globl _bar
_bar:
  callq __Z3foo
  ret

#--- main.s
.globl _main
_main:
  callq _bar
  callq __Z3foo
  ret
