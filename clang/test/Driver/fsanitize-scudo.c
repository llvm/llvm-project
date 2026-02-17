// RUN: %clang --target=aarch64-linux-gnu -fsanitize=scudo %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SCUDO
// RUN: %clang --target=arm-linux-androideabi -fsanitize=scudo %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SCUDO
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=scudo %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SCUDO
// RUN: %clang --target=i386-linux-gnu -fsanitize=scudo %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SCUDO
// RUN: %clang --target=loongarch64-unknown-linux-gnu -fsanitize=scudo %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SCUDO
// RUN: %clang --target=mips64-unknown-linux-gnu -fsanitize=scudo %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SCUDO
// RUN: %clang --target=mips64el-unknown-linux-gnu -fsanitize=scudo %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SCUDO
// RUN: %clang --target=mips-unknown-linux-gnu -fsanitize=scudo %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SCUDO
// RUN: %clang --target=mipsel-unknown-linux-gnu -fsanitize=scudo %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SCUDO
// RUN: %clang --target=powerpc64-unknown-linux -fsanitize=scudo %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SCUDO
// RUN: %clang --target=powerpc64le-unknown-linux -fsanitize=scudo %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SCUDO
// RUN: %clang --target=riscv64-linux-gnu -fsanitize=scudo %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SCUDO
// CHECK-SCUDO: "-fsanitize=scudo"

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=scudo,undefined %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SCUDO-UBSAN
// CHECK-SCUDO-UBSAN: "-fsanitize={{.*}}scudo"

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=scudo -fsanitize-minimal-runtime %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SCUDO-MINIMAL
// CHECK-SCUDO-MINIMAL: "-fsanitize=scudo"
// CHECK-SCUDO-MINIMAL: "-fsanitize-minimal-runtime"

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=undefined,scudo -fsanitize-minimal-runtime %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SCUDO-UBSAN-MINIMAL
// CHECK-SCUDO-UBSAN-MINIMAL: "-fsanitize={{.*}}scudo"
// CHECK-SCUDO-UBSAN-MINIMAL: "-fsanitize-minimal-runtime"

// RUN: not %clang --target=powerpc-unknown-linux -fsanitize=scudo %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-SCUDO
// CHECK-NO-SCUDO: unsupported option

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=scudo,address  %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SCUDO-ASAN
// CHECK-SCUDO-ASAN: error: invalid argument '-fsanitize=scudo' not allowed with '-fsanitize=address'
// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=scudo,leak  %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SCUDO-LSAN
// CHECK-SCUDO-LSAN: error: invalid argument '-fsanitize=scudo' not allowed with '-fsanitize=leak'
// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=scudo,memory  %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SCUDO-MSAN
// CHECK-SCUDO-MSAN: error: invalid argument '-fsanitize=scudo' not allowed with '-fsanitize=memory'
// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=scudo,thread  %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SCUDO-TSAN
// CHECK-SCUDO-TSAN: error: invalid argument '-fsanitize=scudo' not allowed with '-fsanitize=thread'
// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=scudo,hwaddress  %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SCUDO-HWASAN
// CHECK-SCUDO-HWASAN: error: invalid argument '-fsanitize=scudo' not allowed with '-fsanitize=hwaddress'
//
// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=scudo,kernel-memory  %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SCUDO-KMSAN
// CHECK-SCUDO-KMSAN: error: invalid argument '-fsanitize=kernel-memory' not allowed with '-fsanitize=scudo'
