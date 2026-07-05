# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin19.0.0 %t/test.s -o %t/test.o

## Check that we correctly dedup sections that reference dynamic-lookup symbols.
# RUN: %lld -lSystem -dylib --icf=all -undefined dynamic_lookup -o %t/test %t/test.o
# RUN: llvm-objdump --macho --syms %t/test | FileCheck %s

## Check that we still raise an error when using regular undefined symbol
## treatment.
# RUN: not %lld -lSystem -dylib --icf=all -o /dev/null %t/test.o 2>&1 | \
# RUN:   FileCheck %s --check-prefix=ERR

# CHECK: [[#%x,ADDR:]] l    F __TEXT,__text _foo
# CHECK: [[#ADDR]]     l    F __TEXT,__text _bar

# ERR: error: undefined symbol: _undef

#--- test.s

.subsections_via_symbols

_foo:
  callq _undef + 1

_bar:
  callq _undef + 1
