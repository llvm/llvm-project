# Extracting the cpio archive can get over the path limit on windows.
# REQUIRES: shell

# RUN: rm -rf %t.dir
# RUN: mkdir -p %t.dir
# RUN: cd %t.dir

# RUN: not ld.lld --reproduce repro abc -o t 2>&1 | FileCheck %s
# CHECK: cannot open abc: {{N|n}}o such file or directory

# RUN: grep TRAILER repro.cpio
# RUN: cpio -id < repro.cpio
# RUN: FileCheck --check-prefix=RSP %s < repro/response.txt
# RSP: abc
# RSP: -o t
