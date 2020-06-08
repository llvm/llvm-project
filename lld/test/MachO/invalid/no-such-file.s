# REQUIRES: x86
# RUN: not lld -flavor darwinnew -arch x86_64 -o /dev/null %t-no-such-file.o 2>&1 | FileCheck %s

# CHECK: error: cannot open {{.*}}no-such-file.o
