// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=address -fsanitize-minimal-runtime %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-MINIMAL
// CHECK-ASAN-MINIMAL: error: invalid argument '-fsanitize-minimal-runtime' not allowed with '-fsanitize=address'

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=thread -fsanitize-minimal-runtime %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-MINIMAL
// CHECK-TSAN-MINIMAL: error: invalid argument '-fsanitize-minimal-runtime' not allowed with '-fsanitize=thread'

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=undefined -fsanitize-minimal-runtime %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UBSAN-MINIMAL
// CHECK-UBSAN-MINIMAL: "-fsanitize={{((signed-integer-overflow|integer-divide-by-zero|shift-base|shift-exponent|unreachable|return|vla-bound|alignment|null|pointer-overflow|float-cast-overflow|array-bounds|enum|bool|builtin|returns-nonnull-attribute|nonnull-attribute|function),?){18}"}}
// CHECK-UBSAN-MINIMAL: "-fsanitize-minimal-runtime"

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=undefined -fsanitize-minimal-runtime -fsanitize-handler-preserve-all-regs %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UBSAN-MINIMAL-PRESERVE-X86-64
// CHECK-UBSAN-MINIMAL-PRESERVE-X86-64: "-fsanitize={{((signed-integer-overflow|integer-divide-by-zero|shift-base|shift-exponent|unreachable|return|vla-bound|alignment|null|pointer-overflow|float-cast-overflow|array-bounds|enum|bool|builtin|returns-nonnull-attribute|nonnull-attribute|function),?){18}"}}
// CHECK-UBSAN-MINIMAL-PRESERVE-X86-64: "-fsanitize-minimal-runtime"
// CHECK-UBSAN-MINIMAL-PRESERVE-X86-64: "-fsanitize-handler-preserve-all-regs

// RUN: %clang --target=aarch64-linux-gnu -fsanitize=undefined -fsanitize-minimal-runtime -fsanitize-handler-preserve-all-regs %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UBSAN-MINIMAL-PRESERVE-AARCH64
// CHECK-UBSAN-MINIMAL-PRESERVE-AARCH64: "-fsanitize={{((signed-integer-overflow|integer-divide-by-zero|shift-base|shift-exponent|unreachable|return|vla-bound|alignment|null|pointer-overflow|float-cast-overflow|array-bounds|enum|bool|builtin|returns-nonnull-attribute|nonnull-attribute|function),?){18}"}}
// CHECK-UBSAN-MINIMAL-PRESERVE-AARCH64: "-fsanitize-minimal-runtime"
// CHECK-UBSAN-MINIMAL-PRESERVE-AARCH64: "-fsanitize-handler-preserve-all-regs

// RUN: %clang --target=i386-linux-gnu -fsanitize=undefined -fsanitize-minimal-runtime -fsanitize-handler-preserve-all-regs %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UBSAN-MINIMAL-PRESERVE-I386
// CHECK-UBSAN-MINIMAL-PRESERVE-I386: "-fsanitize={{((signed-integer-overflow|integer-divide-by-zero|shift-base|shift-exponent|unreachable|return|vla-bound|alignment|null|pointer-overflow|float-cast-overflow|array-bounds|enum|bool|builtin|returns-nonnull-attribute|nonnull-attribute|function),?){18}"}}
// CHECK-UBSAN-MINIMAL-PRESERVE-I386: "-fsanitize-minimal-runtime"
// CHECK-UBSAN-MINIMAL-PRESERVE-I386-NOT: "-fsanitize-handler-preserve-all-regs

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=integer -fsanitize-trap=integer %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-INTSAN-TRAP
// CHECK-INTSAN-TRAP: "-fsanitize-trap=integer-divide-by-zero,shift-base,shift-exponent,signed-integer-overflow,unsigned-integer-overflow,unsigned-shift-base,implicit-unsigned-integer-truncation,implicit-signed-integer-truncation,implicit-integer-sign-change"

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=integer -fsanitize-minimal-runtime %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-INTSAN-MINIMAL
// CHECK-INTSAN-MINIMAL: "-fsanitize=integer-divide-by-zero,shift-base,shift-exponent,signed-integer-overflow,unsigned-integer-overflow,unsigned-shift-base,implicit-unsigned-integer-truncation,implicit-signed-integer-truncation,implicit-integer-sign-change"
// CHECK-INTSAN-MINIMAL: "-fsanitize-minimal-runtime"

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=implicit-conversion -fsanitize-trap=implicit-conversion %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-IMPL-CONV-TRAP
// CHECK-IMPL-CONV-TRAP: "-fsanitize-trap=implicit-unsigned-integer-truncation,implicit-signed-integer-truncation,implicit-integer-sign-change,implicit-bitfield-conversion"

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=implicit-conversion -fsanitize-minimal-runtime %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-IMPL-CONV-MINIMAL
// CHECK-IMPL-CONV-MINIMAL: "-fsanitize=implicit-unsigned-integer-truncation,implicit-signed-integer-truncation,implicit-integer-sign-change,implicit-bitfield-conversion"
// CHECK-IMPL-CONV-MINIMAL: "-fsanitize-minimal-runtime"

// RUN: %clang --target=aarch64-linux-android -march=armv8-a+memtag -fsanitize=memtag -fsanitize-minimal-runtime %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-MEMTAG-MINIMAL
// CHECK-MEMTAG-MINIMAL: "-fsanitize=memtag-stack,memtag-heap,memtag-globals"
// CHECK-MEMTAG-MINIMAL: "-fsanitize-minimal-runtime"

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=undefined -fsanitize=function -fsanitize-minimal-runtime %s -### 2>&1 | FileCheck /dev/null --implicit-check-not=error:

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=undefined -fsanitize=vptr -fsanitize-minimal-runtime %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UBSAN-VPTR-MINIMAL
// CHECK-UBSAN-VPTR-MINIMAL: error: invalid argument '-fsanitize=vptr' not allowed with '-fsanitize-minimal-runtime'

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=address -fsanitize-minimal-runtime -fsanitize=undefined %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-UBSAN-MINIMAL
// CHECK-ASAN-UBSAN-MINIMAL: error: invalid argument '-fsanitize-minimal-runtime' not allowed with '-fsanitize=address'
// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=hwaddress -fsanitize-minimal-runtime %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-HWASAN-MINIMAL
// CHECK-HWASAN-MINIMAL: error: invalid argument '-fsanitize-minimal-runtime' not allowed with '-fsanitize=hwaddress'

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=cfi -flto -fvisibility=hidden -fsanitize-minimal-runtime -resource-dir=%S/Inputs/resource_dir %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-MINIMAL
// CHECK-CFI-MINIMAL: "-fsanitize=cfi-derived-cast,cfi-icall,cfi-mfcall,cfi-unrelated-cast,cfi-nvcall,cfi-vcall"
// CHECK-CFI-MINIMAL: "-fsanitize-trap=cfi-derived-cast,cfi-icall,cfi-mfcall,cfi-unrelated-cast,cfi-nvcall,cfi-vcall"
// CHECK-CFI-MINIMAL: "-fsanitize-minimal-runtime"

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=cfi -flto -fvisibility=hidden -fsanitize-minimal-runtime -fsanitize-recover=cfi -resource-dir=%S/Inputs/resource_dir %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-RECOVER-MINIMAL
// CHECK-CFI-RECOVER-MINIMAL: "-fsanitize=cfi-derived-cast,cfi-icall,cfi-mfcall,cfi-unrelated-cast,cfi-nvcall,cfi-vcall"
// CHECK-CFI-RECOVER-MINIMAL: "-fsanitize-trap=cfi-derived-cast,cfi-icall,cfi-mfcall,cfi-unrelated-cast,cfi-nvcall,cfi-vcall"
// CHECK-CFI-RECOVER-MINIMAL: "-fsanitize-minimal-runtime"

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=cfi -flto -fvisibility=hidden -fsanitize-minimal-runtime -fno-sanitize-recover=cfi -resource-dir=%S/Inputs/resource_dir %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-ABORT-MINIMAL
// CHECK-CFI-ABORT-MINIMAL: "-fsanitize=cfi-derived-cast,cfi-icall,cfi-mfcall,cfi-unrelated-cast,cfi-nvcall,cfi-vcall"
// CHECK-CFI-ABORT-MINIMAL: "-fsanitize-trap=cfi-derived-cast,cfi-icall,cfi-mfcall,cfi-unrelated-cast,cfi-nvcall,cfi-vcall"
// CHECK-CFI-ABORT-MINIMAL: "-fsanitize-minimal-runtime"

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=cfi -flto -fvisibility=hidden -fsanitize-minimal-runtime -fno-sanitize-trap=cfi -fsanitize-recover=cfi -resource-dir=%S/Inputs/resource_dir %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-NOTRAP-RECOVER-MINIMAL --
// CHECK-CFI-NOTRAP-RECOVER-MINIMAL: "-fsanitize=cfi-derived-cast,cfi-icall,cfi-mfcall,cfi-unrelated-cast,cfi-nvcall,cfi-vcall"
// CHECK-CFI-NOTRAP-RECOVER-MINIMAL: "-fsanitize-recover=cfi-derived-cast,cfi-icall,cfi-mfcall,cfi-unrelated-cast,cfi-nvcall,cfi-vcall"
// CHECK-CFI-NOTRAP-RECOVER-MINIMAL: "-fsanitize-minimal-runtime"

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=cfi -flto -fvisibility=hidden -fsanitize-minimal-runtime -fno-sanitize-trap=cfi -fno-sanitize-recover=cfi -resource-dir=%S/Inputs/resource_dir %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-NOTRAP-ABORT-MINIMAL
// CHECK-CFI-NOTRAP-ABORT-MINIMAL: "-fsanitize=cfi-derived-cast,cfi-icall,cfi-mfcall,cfi-unrelated-cast,cfi-nvcall,cfi-vcall"
// CHECK-CFI-NOTRAP-ABORT-MINIMAL: "-fsanitize-minimal-runtime"

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=cfi -fno-sanitize-trap=cfi-icall -flto -fvisibility=hidden -fsanitize-minimal-runtime -resource-dir=%S/Inputs/resource_dir %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-NOTRAP-MINIMAL
// CHECK-CFI-NOTRAP-MINIMAL: "-fsanitize=cfi-derived-cast,cfi-icall,cfi-mfcall,cfi-unrelated-cast,cfi-nvcall,cfi-vcall"
// CHECK-CFI-NOTRAP-MINIMAL: "-fsanitize-trap=cfi-derived-cast,cfi-mfcall,cfi-unrelated-cast,cfi-nvcall,cfi-vcall"

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=cfi -fno-sanitize-trap=cfi-icall -fno-sanitize=cfi-icall -flto -fvisibility=hidden -fsanitize-minimal-runtime -resource-dir=%S/Inputs/resource_dir %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-NOICALL-MINIMAL
// CHECK-CFI-NOICALL-MINIMAL: "-fsanitize=cfi-derived-cast,cfi-mfcall,cfi-unrelated-cast,cfi-nvcall,cfi-vcall"
// CHECK-CFI-NOICALL-MINIMAL: "-fsanitize-trap=cfi-derived-cast,cfi-mfcall,cfi-unrelated-cast,cfi-nvcall,cfi-vcall"
// CHECK-CFI-NOICALL-MINIMAL: "-fsanitize-minimal-runtime"

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=shadow-call-stack -fsanitize-minimal-runtime %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SCS-MINIMAL
// CHECK-SCS-MINIMAL: "-fsanitize=shadow-call-stack"
// CHECK-SCS-MINIMAL: "-fsanitize-minimal-runtime"
