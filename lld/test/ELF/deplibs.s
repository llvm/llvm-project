# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: echo ".global foo; foo:" | \
# RUN:   llvm-mc -filetype=obj -triple=x86_64 - -o %tfoo.o
# RUN: rm -rf %t.dir %t.cwd
# RUN: mkdir -p %t.dir

## Error if dependent libraries cannot be found.
# RUN: not ld.lld %t.o -o /dev/null 2>&1 | FileCheck %s -DOBJ=%t.o --check-prefix MISSING
# MISSING:      error: [[OBJ]]: unable to find library from dependent library specifier: foo.a
# MISSING-NEXT: error: [[OBJ]]: unable to find library from dependent library specifier: bar

## Can ignore dependent libraries.
# RUN: not ld.lld %t.o -o /dev/null --no-dependent-libraries 2>&1 | FileCheck %s --check-prefix IGNORE
# IGNORE: error: undefined symbol: foo

## -r links preserve dependent libraries.
# RUN: ld.lld %t.o %t.o -r -o %t-r.o
# RUN: not ld.lld %t-r.o -o /dev/null 2>&1 | sort | FileCheck %s -DOBJ=%t-r.o --check-prefixes MINUSR
# MINUSR:      error: [[OBJ]]: unable to find library from dependent library specifier: bar
# MINUSR-NEXT: error: [[OBJ]]: unable to find library from dependent library specifier: foo.a
# MINUSR-NOT:  unable to find library from dependent library specifier

## Dependent libraries searched for symbols after libraries on the command line.
# RUN: mkdir -p %t.cwd
# RUN: cd %t.cwd
# RUN: llvm-ar rc %t.dir/foo.a %tfoo.o
# RUN: touch %t.dir/libbar.so
# RUN: cp %t.dir/foo.a %t.cwd/libcmdline.a
# RUN: ld.lld %t.o libcmdline.a -o /dev/null -L %t.dir --trace 2>&1 | \
# RUN:   FileCheck %s -DOBJ=%t.o -DSO=%t.dir --check-prefix CMDLINE \
# RUN:                --implicit-check-not=foo.a --implicit-check-not=libbar.so
# CMDLINE:      [[OBJ]]
# CMDLINE-NEXT: {{^libcmdline\.a}}

## LLD tries to resolve dependent library specifiers in the following order:
##   1) The name, prefixed with "lib" and suffixed with ".so" or ".a" in a
##      library search path. This means that a directive of "foo.a" could lead
##      to a library named "libfoo.a.a" being linked in.
##   2) The literal name in a library search path.
##   3) The literal name in the current working directory.
## When using library search paths for dependent libraries, LLD follows the same
## rules as for libraries specified on the command line.
# RUN: cp %t.dir/foo.a %t.cwd/foo.a
# RUN: cp %t.dir/foo.a %t.dir/libfoo.a.a

# RUN: ld.lld %t.o -o /dev/null -L %t.dir --trace 2>&1 | \
# RUN:   FileCheck %s -DOBJ=%t.o -DDIR=%t.dir --check-prefix=LIBA-DIR \
# RUN:                --implicit-check-not=foo.a --implicit-check-not=libbar.so

# LIBA-DIR:      [[OBJ]]
# LIBA-DIR-NEXT: [[DIR]]{{[\\/]}}libfoo.a.a

# RUN: rm %t.dir/libfoo.a.a
# RUN: ld.lld %t.o -o /dev/null -L %t.dir --trace 2>&1 | \
# RUN:   FileCheck %s -DOBJ=%t.o -DDIR=%t.dir --check-prefix=PLAIN-DIR \
# RUN:                --implicit-check-not=foo.a --implicit-check-not=libbar.so

# PLAIN-DIR:      [[OBJ]]
# PLAIN-DIR-NEXT: [[DIR]]{{[\\/]}}foo.a

# RUN: rm %t.dir/foo.a
# RUN: ld.lld %t.o -o /dev/null -L %t.dir --trace 2>&1 | \
# RUN:   FileCheck %s -DOBJ=%t.o -DDIR=%t.dir --check-prefix=CWD \
# RUN:                --implicit-check-not=foo.a --implicit-check-not=libbar.so

# CWD:      [[OBJ]]
# CWD-NEXT: {{^foo\.a}}

    call foo
.section ".deplibs","MS",@llvm_dependent_libraries,1
    .asciz "foo.a"
    ## Show that an unneeded archive must be present but may not be linked in.
    .asciz "bar"
