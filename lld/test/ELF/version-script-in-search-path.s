# REQUIRES: x86
# Check that we fall back to search paths if a version script was not found
# This behaviour matches ld.bfd.

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: rm -rf %t.dir && mkdir -p %t.dir
# RUN: echo '{};' > %t.dir/script
# RUN: ld.lld -L%t.dir --version-script=script %t.o -o /dev/null
# RUN: not ld.lld --version-script=script %t.o 2>&1 | FileCheck -check-prefix ERROR %s
# ERROR: error: cannot find version script
