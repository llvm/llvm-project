// -fsanitize-skip-hot-cutoff=undefined=0.5
// RUN: %clang -Werror --target=x86_64-linux-gnu -fsanitize=undefined -fsanitize-skip-hot-cutoff=undefined=0.5 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SKIP-HOT-CUTOFF1
// CHECK-SKIP-HOT-CUTOFF1: "-fsanitize-skip-hot-cutoff={{((signed-integer-overflow|integer-divide-by-zero|shift-base|shift-exponent|unreachable|return|vla-bound|alignment|null|pointer-overflow|float-cast-overflow|array-bounds|enum|bool|builtin|returns-nonnull-attribute|nonnull-attribute|function)=0.5(0*),?){18}"}}

// No-op: no sanitizers are specified
// RUN: %clang -Werror --target=x86_64-linux-gnu -fsanitize-skip-hot-cutoff=undefined=0.5 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SKIP-HOT-CUTOFF2
// CHECK-SKIP-HOT-CUTOFF2-NOT: "-fsanitize"
// CHECK-SKIP-HOT-CUTOFF2-NOT: "-fsanitize-skip-hot-cutoff"

// Enable undefined, then cancel out integer using a cutoff of 0.0
// RUN: %clang -Werror --target=x86_64-linux-gnu -fsanitize=undefined -fsanitize-skip-hot-cutoff=undefined=0.5,integer=0.0 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SKIP-HOT-CUTOFF3
// CHECK-SKIP-HOT-CUTOFF3: "-fsanitize-skip-hot-cutoff={{((unreachable|return|vla-bound|alignment|null|pointer-overflow|float-cast-overflow|array-bounds|enum|bool|builtin|returns-nonnull-attribute|nonnull-attribute|function)=0.5(0*),?){14}"}}

// Enable undefined, then cancel out integer using a cutoff of 0.0, then re-enable signed-integer-overflow
// RUN: %clang -Werror --target=x86_64-linux-gnu -fsanitize=undefined -fsanitize-skip-hot-cutoff=undefined=0.5,integer=0.0,signed-integer-overflow=0.7 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SKIP-HOT-CUTOFF4
// CHECK-SKIP-HOT-CUTOFF4: "-fsanitize-skip-hot-cutoff={{((signed-integer-overflow|unreachable|return|vla-bound|alignment|null|pointer-overflow|float-cast-overflow|array-bounds|enum|bool|builtin|returns-nonnull-attribute|nonnull-attribute|function)=0.[57]0*,?){15}"}}

// Check that -fsanitize-skip-hot-cutoff=undefined=0.4 does not widen the set of -fsanitize=integer checks.
// RUN: %clang -Werror --target=x86_64-linux-gnu -fsanitize=integer -fsanitize-skip-hot-cutoff=undefined=0.4 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SKIP-HOT-CUTOFF5
// CHECK-SKIP-HOT-CUTOFF5: "-fsanitize-skip-hot-cutoff={{((integer-divide-by-zero|shift-base|shift-exponent|signed-integer-overflow)=0.40*,?){4}"}}

// No-op: it's allowed for the user to specify a cutoff of 0.0, though the argument is not passed along by the driver.
// RUN: %clang -Werror --target=x86_64-linux-gnu -fsanitize=undefined -fsanitize-skip-hot-cutoff=undefined=0.0 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SKIP-HOT-CUTOFF6
// CHECK-SKIP-HOT-CUTOFF6: "-fsanitize={{((signed-integer-overflow|integer-divide-by-zero|shift-base|shift-exponent|unreachable|return|vla-bound|alignment|null|pointer-overflow|float-cast-overflow|array-bounds|enum|bool|builtin|returns-nonnull-attribute|nonnull-attribute|function),?){18}"}}
// CHECK-SKIP-HOT-CUTOFF6-NOT: "-fsanitize-skip-hot-cutoff"

// Invalid: bad sanitizer
// RUN: not %clang --target=x86_64-linux-gnu -fsanitize-skip-hot-cutoff=pot=0.0 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SKIP-HOT-CUTOFF7
// CHECK-SKIP-HOT-CUTOFF7: unsupported argument 'pot=0.0' to option '-fsanitize-skip-hot-cutoff='

// Invalid: bad cutoff
// RUN: not %clang --target=x86_64-linux-gnu -fsanitize-skip-hot-cutoff=undefined=xyzzy %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SKIP-HOT-CUTOFF8
// CHECK-SKIP-HOT-CUTOFF8: unsupported argument 'undefined=xyzzy' to option '-fsanitize-skip-hot-cutoff='

// Invalid: -fsanitize-skip-hot-cutoff without parameters
// RUN: not %clang --target=x86_64-linux-gnu -fsanitize-skip-hot-cutoff %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SKIP-HOT-CUTOFF9
// CHECK-SKIP-HOT-CUTOFF9: unknown argument: '-fsanitize-skip-hot-cutoff'

// Invalid: -fsanitize-skip-hot-cutoff=undefined without cutoff
// RUN: not %clang --target=x86_64-linux-gnu -fsanitize-skip-hot-cutoff=undefined %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SKIP-HOT-CUTOFF10
// CHECK-SKIP-HOT-CUTOFF10: unsupported argument 'undefined' to option '-fsanitize-skip-hot-cutoff='

// Invalid: -fsanitize-skip-hot-cutoff=undefined= without cutoff
// RUN: not %clang --target=x86_64-linux-gnu -fsanitize-skip-hot-cutoff=undefined= %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SKIP-HOT-CUTOFF11
// CHECK-SKIP-HOT-CUTOFF11: unsupported argument 'undefined=' to option '-fsanitize-skip-hot-cutoff='

// No-op: -fsanitize-skip-hot-cutoff= without parameters is unusual but valid
// RUN: %clang -Werror --target=x86_64-linux-gnu -fsanitize-skip-hot-cutoff= %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SKIP-HOT-CUTOFF12
// CHECK-SKIP-HOT-CUTOFF12-NOT: "-fsanitize-skip-hot-cutoff"

// Invalid: out of range cutoff
// RUN: not %clang --target=x86_64-linux-gnu -fsanitize-skip-hot-cutoff=undefined=1.1 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SKIP-HOT-CUTOFF13
// CHECK-SKIP-HOT-CUTOFF13: unsupported argument 'undefined=1.1' to option '-fsanitize-skip-hot-cutoff='

// Invalid: out of range cutoff
// RUN: not %clang --target=x86_64-linux-gnu -fsanitize-skip-hot-cutoff=undefined=-0.1 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SKIP-HOT-CUTOFF14
// CHECK-SKIP-HOT-CUTOFF14: unsupported argument 'undefined=-0.1' to option '-fsanitize-skip-hot-cutoff='
