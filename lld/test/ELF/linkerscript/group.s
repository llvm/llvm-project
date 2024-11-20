# REQUIRES: x86
# UNSUPPORTED: system-windows

# RUN: rm -rf %t.dir && mkdir %t.dir
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 b.s -o b.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %p/Inputs/libsearch-st.s -o xyz.o
# RUN: llvm-ar rc %t.dir/libb.a b.o
# RUN: llvm-ar rc %t.dir/libxyz.a xyz.o

# RUN: echo 'GROUP("a.o" libxyz.a -lxyz b.o )' > 1.t
# RUN: not ld.lld 1.t 2>&1 | FileCheck %s --check-prefix=NOLIB
# RUN: ld.lld 1.t -L%t.dir
# RUN: llvm-nm a.out | FileCheck %s

# RUN: echo 'GROUP( "a.o" b.o =libxyz.a )' > 2.t
# RUN: not ld.lld 2.t 2>&1 | FileCheck %s --check-prefix=CANNOT_OPEN -DFILE=libxyz.a
# RUN: ld.lld 2.t --sysroot=%t.dir
# RUN: llvm-nm a.out | FileCheck %s

# RUN: echo 'GROUP("%t.dir/3a.t")' > 3.t
# RUN: echo 'INCLUDE "%t.dir/3a.t"' > 3i.t
# RUN: echo 'GROUP(AS_NEEDED("a.o"))INPUT(/libb.a)' > %t.dir/3a.t
# RUN: ld.lld 3.t --sysroot=%t.dir
# RUN: llvm-nm a.out | FileCheck %s
# RUN: ld.lld 3i.t --sysroot=%t.dir
# RUN: llvm-nm a.out | FileCheck %s

# RUN: echo 'GROUP("%t.dir/4a.t")INPUT(/libb.a)' > 4.t
# RUN: echo 'GROUP(AS_NEEDED("a.o"))' > %t.dir/4a.t
# RUN: not ld.lld 4.t --sysroot=%t.dir 2>&1 | FileCheck %s --check-prefix=CANNOT_OPEN -DFILE=/libb.a

# RUN: echo 'INCLUDE "%t.dir/5a.t" INPUT(/libb.a)' > 5.t
# RUN: echo 'GROUP(a.o)' > %t.dir/5a.t
# RUN: not ld.lld 5.t --sysroot=%t.dir 2>&1 | FileCheck %s --check-prefix=CANNOT_OPEN -DFILE=/libb.a

# CHECK: T _start

# NOLIB: error: {{.*}}unable to find

# RUN: echo 'GROUP("a.o" /libxyz.a )' > a.t
# RUN: echo 'GROUP("%t/a.o" /libxyz.a )' > %t.dir/xyz.t
# RUN: not ld.lld a.t 2>&1 | FileCheck %s --check-prefix=CANNOT_OPEN -DFILE=/libxyz.a
# RUN: not ld.lld a.t --sysroot=%t.dir 2>&1 | FileCheck %s --check-prefix=CANNOT_OPEN -DFILE=/libxyz.a

## Since %t.dir/%t does not exist, report an error, instead of falling back to %t
## without the syroot prefix.
# RUN: not ld.lld %t.dir/xyz.t --sysroot=%t.dir 2>&1 | FileCheck %s --check-prefix=CANNOT_FIND_SYSROOT -DTMP=%t/a.o

# CANNOT_FIND_SYSROOT:      error: {{.*}}xyz.t:1: cannot find [[TMP]] inside {{.*}}.dir
# CANNOT_FIND_SYSROOT-NEXT: >>> GROUP({{.*}}

# CANNOT_OPEN: error: cannot open [[FILE]]: {{.*}}

#--- a.s
.globl _start
_start:
  call b

#--- b.s
.globl b
b:
