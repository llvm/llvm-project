# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: echo 'v1 {bar;};' > %t.ver
# RUN: ld.lld -shared --version-script %t.ver %t.o -o %t.so

## For an unversioned undefined symbol, check we can suggest the symbol with the
## default version.
# RUN: echo 'call bat' | llvm-mc -filetype=obj -triple=x86_64 - -o %tdef1.o
# RUN: not ld.lld %t.so %tdef1.o -o /dev/null 2>&1 | FileCheck --check-prefix=DEFAULT1 %s

# DEFAULT1:      error: undefined symbol: bat
# DEFAULT1-NEXT: >>> referenced by {{.*}}.o:(.text+0x1)
# DEFAULT1-NEXT: >>> did you mean: bar{{$}}
# DEFAULT1-NEXT: >>> defined in: {{.*}}.so

## For a versioned undefined symbol, check we can suggest the symbol with the
## default version.
# RUN: echo '.symver bar.v2,bar@v2; call bar.v2' | llvm-mc -filetype=obj -triple=x86_64 - -o %tdef2.o
# RUN: not ld.lld %t.so %tdef2.o -o /dev/null 2>&1 | FileCheck --check-prefix=DEFAULT2 %s

# DEFAULT2:      error: undefined symbol: bar@v2
# DEFAULT2-NEXT: >>> referenced by {{.*}}.o:(.text+0x1)
# DEFAULT2-NEXT: >>> did you mean: bar{{$}}
# DEFAULT2-NEXT: >>> defined in: {{.*}}.so

## For an unversioned undefined symbol, check we can suggest a symbol with
## a non-default version.
# RUN: echo 'call foo; call _Z3fooi' | llvm-mc -filetype=obj -triple=x86_64 - -o %thidden1.o
# RUN: not ld.lld %t.so %thidden1.o -o /dev/null 2>&1 | FileCheck --check-prefix=HIDDEN1 %s

# HIDDEN1:      error: undefined symbol: foo
# HIDDEN1-NEXT: >>> referenced by {{.*}}.o:(.text+0x1)
# HIDDEN1-NEXT: >>> did you mean: foo@v1
# HIDDEN1-NEXT: >>> defined in: {{.*}}.so
# HIDDEN1-EMPTY:
# HIDDEN1-NEXT: error: undefined symbol: foo(int)
# HIDDEN1-NEXT: >>> referenced by {{.*}}.o:(.text+0x6)
# HIDDEN1-NEXT: >>> did you mean: foo(int)@v1
# HIDDEN1-NEXT: >>> defined in: {{.*}}.so

## For a versioned undefined symbol, check we can suggest a symbol with
## a different version.
# RUN: echo '.symver foo.v2,foo@v2; call foo.v2' | llvm-mc -filetype=obj -triple=x86_64 - -o %thidden2.o
# RUN: not ld.lld %t.so %thidden2.o -o /dev/null 2>&1 | FileCheck --check-prefix=HIDDEN2 %s

# HIDDEN2:      error: undefined symbol: foo@v2
# HIDDEN2-NEXT: >>> referenced by {{.*}}.o:(.text+0x1)
# HIDDEN2-NEXT: >>> did you mean: foo@v1
# HIDDEN2-NEXT: >>> defined in: {{.*}}.so

## %t.so exports bar@@v1 and two VERSYM_HIDDEN symbols: foo@v1 and _Z3fooi@v1.
.globl foo.v1, _Z3fooi.v1, bar
.symver foo.v1,foo@v1
.symver _Z3fooi.v1,_Z3fooi@v1
foo.v1:
_Z3fooi.v1:
bar:
