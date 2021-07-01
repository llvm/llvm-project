// Test the -mgeneral-regs-only option on x86

// RUN: %clang -target i386-unknown-linux-gnu -mgeneral-regs-only %s -### 2>&1 | FileCheck --check-prefix=CMD %s
// RUN: %clang -target x86_64-unknown-linux-gnu -mgeneral-regs-only %s -### 2>&1 | FileCheck --check-prefix=CMD %s
// RUN: %clang -target i386-unknown-linux-gnu -mavx2 -mgeneral-regs-only %s -### 2>&1 | FileCheck --check-prefixes=CMD,CMD-BEFORE %s
// RUN: %clang -target x86_64-unknown-linux-gnu -mavx2 -mgeneral-regs-only %s -### 2>&1 | FileCheck --check-prefixes=CMD,CMD-BEFORE %s
// RUN: %clang -target i386-unknown-linux-gnu -mgeneral-regs-only -mavx2 %s -### 2>&1 | FileCheck --check-prefixes=CMD,CMD-AFTER %s
// RUN: %clang -target x86_64-unknown-linux-gnu -mgeneral-regs-only -mavx2 %s -### 2>&1 | FileCheck --check-prefixes=CMD,CMD-AFTER %s

// RUN: %clang -target i386-unknown-linux-gnu -mgeneral-regs-only -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=IR-GPR %s
// RUN: %clang -target x86_64-unknown-linux-gnu -mgeneral-regs-only -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=IR-GPR %s
// RUN: %clang -target i386-unknown-linux-gnu -mavx2 -mgeneral-regs-only -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=IR-GPR %s
// RUN: %clang -target x86_64-unknown-linux-gnu -mavx2 -mgeneral-regs-only -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=IR-GPR %s
// RUN: %clang -target i386-unknown-linux-gnu -mgeneral-regs-only -mavx2 -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=IR-AVX2 %s
// RUN: %clang -target x86_64-unknown-linux-gnu -mgeneral-regs-only -mavx2 -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=IR-AVX2 %s

// CMD-BEFORE: "-target-feature" "+avx2"
// CMD: "-target-feature" "-x87"
// CMD: "-target-feature" "-mmx"
// CMD: "-target-feature" "-sse"
// CMD-AFTER: "-target-feature" "+avx2"

void foo() { }

// IR-GPR: attributes {{.*}} = { {{.*}} "target-features"="{{.*}}-avx{{.*}}-avx2{{.*}}-avx512f{{.*}}-sse{{.*}}-sse2{{.*}}-ssse3{{.*}}-x87{{.*}}"
// IR-AVX2: attributes {{.*}} = { {{.*}} "target-features"="{{.*}}+avx{{.*}}+avx2{{.*}}+sse{{.*}}+sse2{{.*}}+ssse3{{.*}}-avx512f{{.*}}-x87{{.*}}"
