// RUN: env CCC_OVERRIDE_OPTIONS="#+-Os +-Oz +-O +-O3 +-Oignore +a +b +c xb Xa Omagic ^-###  " %clang -target x86_64-apple-darwin %s -O2 b -O3 2>&1 | FileCheck %s
// RUN: env CCC_OVERRIDE_OPTIONS="x-Werror +-msse" %clang -target x86_64-apple-darwin -Werror %s -c -### 2>&1 | FileCheck %s -check-prefix=RM-WERROR

// CHECK: "-cc1"
// CHECK-NOT: "-Oignore"
// CHECK: "-Omagic"
// CHECK-NOT: "-Oignore"

// RM-WERROR: ### CCC_OVERRIDE_OPTIONS: x-Werror +-msse
// RM-WERROR-NEXT: ### Deleting argument -Werror
// RM-WERROR-NEXT: ### Adding argument -msse at end
// RM-WERROR-NOT: "-Werror"
