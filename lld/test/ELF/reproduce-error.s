# RUN: rm -rf %t.dir
# RUN: mkdir -p %t.dir
# RUN: cd %t.dir

# RUN: not ld.lld --reproduce repro.tar abc -o t 2>&1 | FileCheck %s
# CHECK: cannot open abc: {{N|n}}o such file or directory

# RUN: tar xOf repro.tar repro/response.txt | FileCheck --check-prefix=RSP %s
# RSP: abc
# RSP: -o t
