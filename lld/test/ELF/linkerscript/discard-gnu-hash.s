# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o

# RUN: ld.lld --hash-style both -shared -o %t %t.o
# RUN: llvm-readelf -S -d %t | FileCheck %s
# CHECK: .gnu.hash
# CHECK: .hash

# CHECK: (GNU_HASH)
# CHECK: (HASH)

# RUN: echo "SECTIONS { /DISCARD/ : { *(.hash) } }" > %t.script
# RUN: ld.lld --hash-style both -shared -o %t -T %t.script %t.o
# RUN: llvm-readelf -S -d %t | FileCheck %s --check-prefix=HASH
# HASH-NOT: .hash
# HASH:     .gnu.hash
# HASH-NOT: .hash

# HASH-NOT: (HASH)
# HASH:     (GNU_HASH)
# HASH-NOT: (HASH)

# RUN: echo "SECTIONS { /DISCARD/ : { *(.gnu.hash) } }" > %t.script
# RUN: ld.lld --hash-style both -shared -o %t -T %t.script %t.o
# RUN: llvm-readelf -S -d %t | FileCheck %s --check-prefix=GNUHASH
# GNUHASH-NOT: .gnu.hash
# GNUHASH:     .hash
# GNUHASH-NOT: .gnu.hash

# GNUHASH-NOT: (GNU_HASH)
# GNUHASH:     (HASH)
# GNUHASH-NOT: (GNU_HASH)
