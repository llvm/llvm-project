# REQUIRES: x86, cpio

# RUN: rm -rf %t.dir
# RUN: mkdir -p %t.dir
# RUN: cd %t.dir

# RUN: not ld.lld --reproduce repro abc -o t 2>&1 | FileCheck %s
# CHECK: cannot open abc: {{N|n}}o such file or directory

# RUN: grep TRAILER repro.cpio
# RUN: echo "*response.txt" > list.txt
# RUN: cpio -i --to-stdout --pattern-file=list.txt < repro.cpio \
# RUN:   | FileCheck %s --check-prefix=RSP
# RSP: abc
# RSP: -o t
