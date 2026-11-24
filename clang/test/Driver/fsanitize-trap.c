// RUN: %clang --target=x86_64-linux-gnu -fsanitize=undefined -fsanitize-trap=undefined %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UNDEFINED-TRAP
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=undefined -fsanitize-trap=undefined -fno-sanitize-trap=signed-integer-overflow %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UNDEFINED-TRAP2
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=undefined -fsanitize-undefined-trap-on-error %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UNDEFINED-TRAP
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=undefined-trap -fsanitize-undefined-trap-on-error %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UNDEFINED-TRAP
// RUN: %clang --target=x86_64-linux-gnu -fsanitize-undefined-trap-on-error -fsanitize=undefined-trap %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UNDEFINED-TRAP
// RUN: %clang --target=x86_64-linux-gnu -fsanitize-trap -fsanitize=undefined-trap %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UNDEFINED-TRAP
// CHECK-UNDEFINED-TRAP-NOT: -fsanitize-recover
// CHECK-UNDEFINED-TRAP: "-fsanitize={{((signed-integer-overflow|integer-divide-by-zero|shift-base|shift-exponent|unreachable|return|vla-bound|alignment|null|pointer-overflow|float-cast-overflow|array-bounds|enum|bool|builtin|returns-nonnull-attribute|nonnull-attribute|function),?){18}"}}
// CHECK-UNDEFINED-TRAP: "-fsanitize-trap=alignment,array-bounds,bool,builtin,enum,float-cast-overflow,function,integer-divide-by-zero,nonnull-attribute,null,pointer-overflow,return,returns-nonnull-attribute,shift-base,shift-exponent,signed-integer-overflow,unreachable,vla-bound"
// CHECK-UNDEFINED-TRAP2: "-fsanitize-trap=alignment,array-bounds,bool,builtin,enum,float-cast-overflow,function,integer-divide-by-zero,nonnull-attribute,null,pointer-overflow,return,returns-nonnull-attribute,shift-base,shift-exponent,unreachable,vla-bound"

// Disabling every member of an enabled group must not leave a stale group bit
// that emits an empty flag to cc1.
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=bounds -fsanitize-trap=bounds -fno-sanitize-trap=array-bounds,local-bounds -fsanitize-recover=bounds -fno-sanitize-recover=array-bounds,local-bounds -fsanitize-merge=bounds -fno-sanitize-merge=array-bounds,local-bounds -fsanitize-annotate-debug-info=bounds -fno-sanitize-annotate-debug-info=array-bounds,local-bounds %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CLEAR-GROUPS
// CHECK-CLEAR-GROUPS: "-fsanitize=array-bounds,local-bounds"
// CHECK-CLEAR-GROUPS-NOT: "-fsanitize-trap=
// CHECK-CLEAR-GROUPS-NOT: "-fsanitize-recover=
// CHECK-CLEAR-GROUPS-NOT: "-fsanitize-merge=
// CHECK-CLEAR-GROUPS-NOT: "-fsanitize-annotate-debug-info=
