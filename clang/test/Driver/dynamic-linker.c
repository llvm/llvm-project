// RUN: %clang -target armv7-unknown-linux-gnueabi -### /dev/null -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-DYNAMIC-LINKER %s
// RUN: %clang -target i386-unknown-linux-gnu -### /dev/null -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-DYNAMIC-LINKER %s
// RUN: %clang -target mips64-unknown-linux-gnu -### /dev/null -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-DYNAMIC-LINKER %s
// RUN: %clang -target powerpc64-unknown-linux-gnu -### /dev/null -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-DYNAMIC-LINKER %s
// RUN: %clang -target x86_64-unknown-linux-gnu -### /dev/null -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-DYNAMIC-LINKER %s

// RUN: %clang -target armv7-unknown-linux-gnueabi -### -shared /dev/null -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-SHARED %s
// RUN: %clang -target i386-unknown-linux-gnu -### -shared /dev/null -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-SHARED %s
// RUN: %clang -target mips64-unknown-linux-gnu -### -shared /dev/null -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-SHARED %s
// RUN: %clang -target powerpc64-unknown-linux-gnu -### -shared /dev/null -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-SHARED %s
// RUN: %clang -target x86_64-unknown-linux-gnu -### -shared /dev/null -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-SHARED %s


// RUN: %clang --target=armv7-unknown-linux-gnueabi -### -Werror -shared -rdynamic /dev/null -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-SHARED %s
// RUN: %clang -target i386-unknown-linux-gnu -### -shared -rdynamic /dev/null -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-SHARED %s
// RUN: %clang -target mips64-unknown-linux-gnu -### -shared -rdynamic /dev/null -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-SHARED %s
// RUN: %clang -target powerpc64-unknown-linux-gnu -### -shared -rdynamic /dev/null -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-SHARED %s
// RUN: %clang -target x86_64-unknown-linux-gnu -### -shared -rdynamic /dev/null -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-SHARED %s

// RUN: %clang -target armv7-unknown-linux-gnueabi -### -static /dev/null -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-STATIC %s
// RUN: %clang -target i386-unknown-linux-gnu -### -static /dev/null -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-STATIC %s
// RUN: %clang -target mips64-unknown-linux-gnu -### -static /dev/null -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-STATIC %s
// RUN: %clang -target powerpc64-unknown-linux-gnu -### -static /dev/null -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-STATIC %s
// RUN: %clang -target x86_64-unknown-linux-gnu -### -static /dev/null -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-STATIC %s

// CHECK-SHARED: "-shared"
// CHECK-RDYNAMIC: "-export-dynamic"
// CHECK-STATIC: "-{{B?}}static"
// CHECK-DYNAMIC-LINKER: "-dynamic-linker"
// CHECK-SHARED-NOT: "-dynamic-linker"
// CHECK-STATIC-NOT: "-dynamic-linker"

