# REQUIRES: x86
# UNSUPPORTED: system-windows

# RUN: mkdir -p %t.dir
# RUN: rm -f %t.dir/libxyz.a
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux \
# RUN:   %p/Inputs/libsearch-st.s -o %t2.o
# RUN: llvm-ar rcs %t.dir/libxyz.a %t2.o

# RUN: echo "GROUP(\"%t\")" > %t.script
# RUN: ld.lld -o %t2 %t.script
# RUN: llvm-readobj %t2 > /dev/null

# RUN: echo "INPUT(\"%t\")" > %t.script
# RUN: ld.lld -o %t2 %t.script
# RUN: llvm-readobj %t2 > /dev/null

# RUN: echo "GROUP(\"%t\" libxyz.a )" > %t.script
# RUN: not ld.lld -o /dev/null %t.script 2>/dev/null
# RUN: ld.lld -o %t2 %t.script -L%t.dir
# RUN: llvm-readobj %t2 > /dev/null

# RUN: echo "GROUP(\"%t\" =libxyz.a )" > %t.script
# RUN: not ld.lld -o /dev/null %t.script  2>/dev/null
# RUN: ld.lld -o %t2 %t.script --sysroot=%t.dir
# RUN: llvm-readobj %t2 > /dev/null

# RUN: echo "GROUP(\"%t\" -lxyz )" > %t.script
# RUN: not ld.lld -o /dev/null %t.script  2>/dev/null
# RUN: ld.lld -o %t2 %t.script -L%t.dir
# RUN: llvm-readobj %t2 > /dev/null

# RUN: echo "GROUP(\"%t\" libxyz.a )" > %t.script
# RUN: not ld.lld -o /dev/null %t.script  2>/dev/null
# RUN: ld.lld -o %t2 %t.script -L%t.dir
# RUN: llvm-readobj %t2 > /dev/null

# RUN: echo "GROUP(\"%t\" /libxyz.a )" > %t.script
# RUN: echo "GROUP(\"%t\" /libxyz.a )" > %t.dir/xyz.script
# RUN: not ld.lld -o /dev/null %t.script 2>&1 | FileCheck %s --check-prefix=CANNOT_OPEN -DFILE=/libxyz.a
# RUN: not ld.lld -o /dev/null %t.script --sysroot=%t.dir 2>&1 | FileCheck %s --check-prefix=CANNOT_OPEN -DFILE=/libxyz.a

## Since %t.dir/%t does not exist, report an error, instead of falling back to %t
## without the syroot prefix.
# RUN: not ld.lld -o /dev/null %t.dir/xyz.script --sysroot=%t.dir 2>&1 | FileCheck %s --check-prefix=CANNOT_FIND_SYSROOT -DTMP=%t

# CANNOT_FIND_SYSROOT:      error: {{.*}}xyz.script:1: cannot find [[TMP]] inside [[TMP]].dir
# CANNOT_FIND_SYSROOT-NEXT: >>> GROUP({{.*}}

# RUN: echo "GROUP(\"%t.script2\")" > %t.script1
# RUN: echo "GROUP(\"%t\")" > %t.script2
# RUN: ld.lld -o %t2 %t.script1
# RUN: llvm-readobj %t2 > /dev/null

# RUN: echo "GROUP(AS_NEEDED(\"%t\"))" > %t.script
# RUN: ld.lld -o %t2 %t.script
# RUN: llvm-readobj %t2 > /dev/null

# CANNOT_OPEN: error: cannot open [[FILE]]: {{.*}}

.globl _start
_start:
  ret
