// Test -R passing search directories to the linker
// RUN: %clang %s -rpath /dir1/ -rpath /dir2/ -### 2>&1  --target=powerpc-ibm-aix | FileCheck %s
// RUN: %clang %s -rpath /dir1/ -rpath /dir2/ -### 2>&1  --target=powerpc64-ibm-aix | FileCheck %s
// RUN: %clang %s -rpath /dir1/ -rpath /dir2/ -bfakelibpath -### 2>&1  --target=powerpc-ibm-aix | FileCheck %s
// RUN: %clang %s -rpath /dir1/ -rpath /dir2/ -bfakelibpath -### 2>&1  --target=powerpc64-ibm-aix | FileCheck %s
// RUN: %clang %s -rpath /dir1/ -rpath /dir2/ -Wl,-bloadmap:-blibpath -### 2>&1  --target=powerpc-ibm-aix | FileCheck %s
// RUN: %clang %s -rpath /dir1/ -rpath /dir2/ -Wl,-bloadmap:-blibpath -### 2>&1  --target=powerpc64-ibm-aix | FileCheck %s
// RUN: %clang %s -rpath /dir1/ -rpath /dir2/ -Wl,-fakeblibpath -### 2>&1  --target=powerpc-ibm-aix | FileCheck %s
// RUN: %clang %s -rpath /dir1/ -rpath /dir2/ -Wl,-fakeblibpath -### 2>&1  --target=powerpc64-ibm-aix | FileCheck %s

// RUN: %clang %s -bsvr4 -Wl,-R/dir1/ -Wl,-blibpath:/dir2/ -### 2>&1  --target=powerpc-ibm-aix | FileCheck --check-prefix=CHECK-LAST %s
// RUN: %clang %s -bsvr4 -Wl,-R/dir1/ -Wl,-blibpath:/dir2/ -### 2>&1  --target=powerpc64-ibm-aix | FileCheck --check-prefix=CHECK-LAST %s
// RUN: %clang %s -bsvr4 -Xlinker -R/dir1/ -Xlinker -blibpath:/dir2/ -### 2>&1  --target=powerpc-ibm-aix | FileCheck --check-prefix=CHECK-LAST %s
// RUN: %clang %s -bsvr4 -Xlinker -R/dir1/ -Xlinker -blibpath:/dir2/ -### 2>&1  --target=powerpc64-ibm-aix | FileCheck --check-prefix=CHECK-LAST %s
//
// RUN: not %clang %s -rpath /dir1/ -rpath /dir2/ -bnolibpath -### 2>&1  --target=powerpc-ibm-aix | FileCheck --check-prefix=CHECK-ERBN %s
// RUN: not %clang %s -rpath /dir1/ -rpath /dir2/ -bnolibpath -### 2>&1  --target=powerpc64-ibm-aix | FileCheck --check-prefix=CHECK-ERBN %s
// RUN: not %clang %s -rpath /dir1/ -rpath /dir2/ -Wl,-bnolibpath -### 2>&1  --target=powerpc-ibm-aix | FileCheck --check-prefix=CHECK-ERWLBN %s
// RUN: not %clang %s -rpath /dir1/ -rpath /dir2/ -Wl,-bnolibpath -### 2>&1  --target=powerpc64-ibm-aix | FileCheck --check-prefix=CHECK-ERWLBN %s
// RUN: not %clang %s -rpath /dir1/ -rpath /dir2/ -Wl,-bnoentry,-bnolibpath -### 2>&1  --target=powerpc-ibm-aix | FileCheck --check-prefix=CHECK-ERWLBN %s
// RUN: not %clang %s -rpath /dir1/ -rpath /dir2/ -Wl,-bnoentry,-bnolibpath -### 2>&1  --target=powerpc64-ibm-aix | FileCheck --check-prefix=CHECK-ERWLBN %s
// RUN: not %clang %s -rpath /dir1/ -rpath /dir2/ -Xlinker -bnolibpath -### 2>&1  --target=powerpc-ibm-aix | FileCheck --check-prefix=CHECK-ERXBN %s
// RUN: not %clang %s -rpath /dir1/ -rpath /dir2/ -Xlinker -bnolibpath -### 2>&1  --target=powerpc64-ibm-aix | FileCheck --check-prefix=CHECK-ERXBN %s
//
// RUN: not %clang %s -rpath /dir1/ -rpath /dir2/ -blibpath:/dir3/ -### 2>&1  --target=powerpc-ibm-aix | FileCheck --check-prefix=CHECK-ERB %s
// RUN: not %clang %s -rpath /dir1/ -rpath /dir2/ -blibpath:/dir3/ -### 2>&1  --target=powerpc64-ibm-aix | FileCheck --check-prefix=CHECK-ERB %s
// RUN: not %clang %s -rpath /dir1/ -rpath /dir2/ -Wl,-blibpath:/dir3/ -### 2>&1  --target=powerpc-ibm-aix | FileCheck --check-prefix=CHECK-ERWL %s
// RUN: not %clang %s -rpath /dir1/ -rpath /dir2/ -Wl,-blibpath:/dir3/ -### 2>&1  --target=powerpc64-ibm-aix | FileCheck --check-prefix=CHECK-ERWL %s
// RUN: not %clang %s -rpath /dir1/ -rpath /dir2/ -Wl,-bnoentr,-blibpath:/dir3/ -### 2>&1  --target=powerpc-ibm-aix | FileCheck --check-prefix=CHECK-ERWL %s
// RUN: not %clang %s -rpath /dir1/ -rpath /dir2/ -Wl,-bnoentr,-blibpath:/dir3/ -### 2>&1  --target=powerpc64-ibm-aix | FileCheck --check-prefix=CHECK-ERWL %s
// RUN: not %clang %s -rpath /dir1/ -rpath /dir2/ -Xlinker -blibpath:/dir3/ -### 2>&1  --target=powerpc-ibm-aix | FileCheck --check-prefix=CHECK-ERX %s
// RUN: not %clang %s -rpath /dir1/ -rpath /dir2/ -Xlinker -blibpath:/dir3/ -### 2>&1  --target=powerpc64-ibm-aix | FileCheck --check-prefix=CHECK-ERX %s

//CHECK: -blibpath:/dir1/:/dir2/:/usr/lib:/lib
//CHECK-LAST: -blibpath:/dir2/
//CHECK-ERBN: error: cannot specify '-bnolibpath' along with '-rpath'
//CHECK-ERWLBN: error: cannot specify '-Wl,-bnolibpath' along with '-rpath'
//CHECK-ERXBN: error: cannot specify '-Xlinker -bnolibpath' along with '-rpath'
//CHECK-ERB: error: cannot specify '-blibpath:/dir3/' along with '-rpath'
//CHECK-ERWL: error: cannot specify '-Wl,-blibpath:/dir3/' along with '-rpath'
//CHECK-ERX: error: cannot specify '-Xlinker -blibpath:/dir3/' along with '-rpath'
