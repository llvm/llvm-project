// Check that the object size check is disabled at -O0.
//
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=object-size %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-OSIZE
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=object-size %s -O0 -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-OSIZE
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=null,object-size %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-OSIZE
// RUN: %clang --target=x86_64-linux-gnu -Werror -fsanitize=null,object-size %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-OSIZE
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=undefined %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-OSIZE-NO-WARNING

// Check that the object size check is enabled at other optimization levels.
//
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=undefined -O1 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-HAS-OSIZE
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=object-size -O2 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-HAS-OSIZE
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=object-size -O3 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-HAS-OSIZE
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=object-size -O4 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-HAS-OSIZE
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=object-size -Ofast %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-HAS-OSIZE
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=object-size -Os %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-HAS-OSIZE
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=object-size -Oz %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-HAS-OSIZE
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=object-size -Og %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-HAS-OSIZE

// Use of trap mode shouldn't affect the object size check.
//
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=undefined -fsanitize-trap=undefined -O1 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-HAS-OSIZE
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=undefined-trap -O1 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-HAS-OSIZE
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=undefined-trap -fsanitize-undefined-trap-on-error -O1 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-HAS-OSIZE

// CHECK-HAS-OSIZE-NOT: warning: the object size sanitizer
// CHECK-HAS-OSIZE: -fsanitize={{[^ ]*}}object-size

// CHECK-NO-OSIZE: warning: the object size sanitizer has no effect at -O0, but is explicitly enabled: -fsanitize={{[^ ]*}}object-size
// CHECK-NO-OSIZE-NOT: -fsanitize={{[^ ]*}}object-size

// CHECK-NO-OSIZE-NO-WARNING-NOT: -fsanitize={{[^ ]*}}object-size
