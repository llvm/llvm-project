# REQUIRES: x86
# UNSUPPORTED: system-windows

# RUN: rm -rf %t.dir && mkdir %t.dir
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %p/Inputs/libsearch-st.s -o b.o
# RUN: llvm-ar rc %t.dir/libxyz.a b.o

# RUN: echo 'GROUP("a.o")' > %t.t
# RUN: ld.lld -o %t2 %t.t
# RUN: llvm-readobj %t2 > /dev/null

# RUN: echo 'INPUT("a.o")' > %t.t
# RUN: ld.lld -o %t2 %t.t
# RUN: llvm-readobj %t2 > /dev/null

# RUN: echo 'GROUP("a.o" libxyz.a )' > %t.t
# RUN: not ld.lld -o /dev/null %t.t 2>/dev/null
# RUN: ld.lld -o %t2 %t.t -L%t.dir
# RUN: llvm-readobj %t2 > /dev/null

# RUN: echo 'GROUP("a.o" =libxyz.a )' > %t.t
# RUN: not ld.lld -o /dev/null %t.t  2>/dev/null
# RUN: ld.lld -o %t2 %t.t --sysroot=%t.dir
# RUN: llvm-readobj %t2 > /dev/null

# RUN: echo 'GROUP("a.o" -lxyz )' > %t.t
# RUN: not ld.lld -o /dev/null %t.t  2>/dev/null
# RUN: ld.lld -o %t2 %t.t -L%t.dir
# RUN: llvm-readobj %t2 > /dev/null

# RUN: echo 'GROUP("a.o" libxyz.a )' > %t.t
# RUN: not ld.lld -o /dev/null %t.t  2>/dev/null
# RUN: ld.lld -o %t2 %t.t -L%t.dir
# RUN: llvm-readobj %t2 > /dev/null

# RUN: echo 'GROUP("a.o" /libxyz.a )' > %t.t
# RUN: echo 'GROUP("%t/a.o" /libxyz.a )' > %t.dir/xyz.t
# RUN: not ld.lld -o /dev/null %t.t 2>&1 | FileCheck %s --check-prefix=CANNOT_OPEN -DFILE=/libxyz.a
# RUN: not ld.lld -o /dev/null %t.t --sysroot=%t.dir 2>&1 | FileCheck %s --check-prefix=CANNOT_OPEN -DFILE=/libxyz.a

## Since %t.dir/%t does not exist, report an error, instead of falling back to %t
## without the syroot prefix.
# RUN: not ld.lld -o /dev/null %t.dir/xyz.t --sysroot=%t.dir 2>&1 | FileCheck %s --check-prefix=CANNOT_FIND_SYSROOT -DTMP=%t/a.o

# CANNOT_FIND_SYSROOT:      error: {{.*}}xyz.t:1: cannot find [[TMP]] inside {{.*}}.dir
# CANNOT_FIND_SYSROOT-NEXT: >>> GROUP({{.*}}

# RUN: echo 'GROUP("2.t")' > 1.t
# RUN: echo 'GROUP("a.o")' > 2.t
# RUN: ld.lld 1.t
# RUN: llvm-readobj a.out > /dev/null

# RUN: echo 'GROUP(AS_NEEDED("a.o"))' > 1.t
# RUN: ld.lld 1.t
# RUN: llvm-readobj a.out > /dev/null

# CANNOT_OPEN: error: cannot open [[FILE]]: {{.*}}

#--- a.s
.globl _start
_start:
  ret
