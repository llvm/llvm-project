// RUN: %clang -### -c -finline-stringops=memset %s 2>&1 | FileCheck --check-prefix=CHECK-MEMSET %s
// RUN: %clang -### -c -finline-stringops %s 2>&1 | FileCheck --check-prefix=CHECK-MEMSET %s
// RUN: %clang -### -c -fno-inline-stringops -finline-stringops=memset %s 2>&1 | FileCheck --check-prefix=CHECK-MEMSET %s
// RUN: %clang -### -c -fno-inline-stringops -finline-stringops %s 2>&1 | FileCheck --check-prefix=CHECK-MEMSET %s
// CHECK-MEMSET: "-finline-stringops=memset"

// RUN: %clang -### -c -finline-stringops=memset -fno-inline-stringops %s 2>&1 | FileCheck --check-prefix=CHECK-NO-INLINE %s
// RUN: %clang -### -c -finline-stringops -fno-inline-stringops %s 2>&1 | FileCheck --check-prefix=CHECK-NO-INLINE %s
// CHECK-NO-INLINE-NOT: "-finline-stringops="

// RUN: not %clang -### -c -finline-stringops=invalid %s 2>&1 | FileCheck --check-prefix=CHECK-INVALID %s
// CHECK-INVALID: error: invalid value 'invalid' in '-finline-stringops=invalid'
