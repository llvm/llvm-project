# REQUIRES: x86

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/2.s -o %t/2.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/3.s -o %t/3.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/main.s -o %t/main.o

# RUN: %lld -lSystem -dylib -install_name %t/my_lib.dylib -o %t/mylib.dylib %t/2.o
# RUN: %lld -lSystem %t/2.o %t/main.o -o %t/main
# RUN: %lld -lSystem -bundle -bundle_loader %t/main -o %t/bundle.bundle %t/3.o %t/mylib.dylib
## Check bundle.bundle to ensure the `my_func` symbol is from executable
# RUN: llvm-nm -m %t/bundle.bundle | FileCheck %s --check-prefix BUNDLE
# BUNDLE: (undefined) external my_func (from executable)
# RUN: llvm-objdump  --macho --lazy-bind %t/bundle.bundle | FileCheck %s --check-prefix BUNDLE-OBJ
# BUNDLE-OBJ: segment  section             address            dylib                 symbol
# BUNDLE-OBJ: __DATA   __la_symbol_ptr     0x{{[0-9a-f]*}}    main-executable       my_fun

# RUN: %lld -lSystem -bundle -bundle_loader %t/main -o %t/bundle2.bundle %t/3.o %t/2.o
## Check bundle.bundle to ensure the `my_func` symbol is not from executable
# RUN: llvm-nm -m %t/bundle2.bundle | FileCheck %s --check-prefix BUNDLE2
# BUNDLE2: (__TEXT,__text) external my_func

# Test that bundle_loader can only be used with MachO bundle output.
# RUN: not %lld -lSystem -bundle_loader %t/main -o %t/bundle3.bundle 2>&1 | FileCheck %s --check-prefix ERROR
# ERROR: -bundle_loader can only be used with MachO bundle output

#--- 2.s
# my_lib: This contains the exported function
.globl my_func
my_func:
  retq

#--- 3.s
# my_user.s: This is the user/caller of the
#            exported function
.text
my_user:
  callq my_func()
  retq

#--- main.s
# main.s: dummy exec/main loads the exported function.
# This is basically a way to say `my_user` should get
# `my_func` from this executable.
.globl _main
.text
 _main:
  retq
