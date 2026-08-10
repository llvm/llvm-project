// RUN: touch %t.o

// RUN: %clang -target i386-unknown-linux --dyld-prefix /foo -### %t.o 2>&1 | FileCheck --check-prefix=CHECK-32 %s
// CHECK-32: "-dynamic-linker" "/foo{{(/usr/i386-unknown-linux)?}}/lib/ld-linux.so.2"

// RUN: %clang -target x86_64-unknown-linux --dyld-prefix /foo -### %t.o 2>&1 | FileCheck --check-prefix=CHECK-64 %s
// CHECK-64: "-dynamic-linker" "/foo{{(/usr/x86_64-unknown-linux)?}}/lib{{(64)?}}/ld-linux-x86-64.so.2"

// RUN: %clang -target x86_64-unknown-linux-gnux32 --dyld-prefix /foo -### %t.o 2>&1 | FileCheck --check-prefix=CHECK-X32 %s
// CHECK-X32: "-dynamic-linker" "/foo{{(/x86_64-unknown-linux-gnux32)?}}/lib{{(x32)?}}/ld-linux-x32.so.2"
