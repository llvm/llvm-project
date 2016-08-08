# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t1.o

# RUN: echo "SECTIONS {     \
# RUN:   ASSERT(1, \"true\") \
# RUN:  }" > %t1.script
# RUN: ld.lld -shared -o %t1 --script %t1.script %t1.o
# RUN: llvm-readobj %t1 > /dev/null

# RUN: echo "SECTIONS {                                  \
# RUN:   ASSERT(ASSERT(42, \"true\") == 42, \"true\") \
# RUN:  }" > %t2.script
# RUN: ld.lld -shared -o %t2 --script %t2.script %t1.o
# RUN: llvm-readobj %t2 > /dev/null

# RUN: echo "SECTIONS {     \
# RUN:   ASSERT(0, \"fail\") \
# RUN:  }" > %t3.script
# RUN: not ld.lld -shared -o %t3 --script %t3.script %t1.o > %t.log 2>&1
# RUN: FileCheck %s -check-prefix=FAIL < %t.log
# FAIL: fail

# RUN: echo "SECTIONS {     \
# RUN:   . = ASSERT(0x1000, \"true\"); \
# RUN:  }" > %t4.script
# RUN: ld.lld -shared -o %t4 --script %t4.script %t1.o
# RUN: llvm-readobj %t4 > /dev/null
