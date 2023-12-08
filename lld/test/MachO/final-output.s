# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos -o %t.o %s

## -final_output sets the default for -install_name, but an explicit
## -install_name wins
# RUN: %lld -dylib -o %t.dylib -final_output /lib/foo.dylib %t.o
# RUN: llvm-otool -D %t.dylib | FileCheck -DID=/lib/foo.dylib %s

# RUN: %lld -dylib -o %t.dylib -install_name /foo/bar.dylib \
# RUN:     -final_output /lib/foo.dylib %t.o
# RUN: llvm-otool -D %t.dylib | FileCheck -DID=/foo/bar.dylib %s

# CHECK: [[ID]]

.globl __Z3foo
__Z3foo:
  ret
