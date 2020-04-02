# REQUIRES: x86
# RUN: not lld -flavor darwinnew -o /dev/null %t-no-such-file.o 2>&1 | FileCheck %s

# CHECK: cannot open {{.*}}no-such-file.o
