// RUN: %clang --target=loongarch64 -march=loongarch64 -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1-LOONGARCH64
// RUN: %clang --target=loongarch64 -march=la464 -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1-LA464
// RUN: %clang --target=loongarch64 -march=la64v1.0 -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1-LA64V1P0
// RUN: %clang --target=loongarch64 -march=la64v1.1 -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1-LA64V1P1
// RUN: %clang --target=loongarch64 -march=la664 -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1-LA664
// RUN: %clang --target=loongarch64 -march=loongarch64 -S -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=IR-LOONGARCH64
// RUN: %clang --target=loongarch64 -march=la464 -S -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=IR-LA464
// RUN: %clang --target=loongarch64 -march=la64v1.0 -S -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=IR-LA64V1P0
// RUN: %clang --target=loongarch64 -march=la64v1.1 -S -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=IR-LA64V1P1
// RUN: %clang --target=loongarch64 -march=la664 -S -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=IR-LA664

// CC1-LOONGARCH64: "-target-cpu" "loongarch64"
// CC1-LOONGARCH64-NOT: "-target-feature"
// CC1-LOONGARCH64: "-target-feature" "+64bit" "-target-feature" "+f" "-target-feature" "+d" "-target-feature" "+ual"
// CC1-LOONGARCH64-NOT: "-target-feature"
// CC1-LOONGARCH64: "-target-abi" "lp64d"

// CC1-LA464: "-target-cpu" "la464"
// CC1-LA464-NOT: "-target-feature"
// CC1-LA464: "-target-feature" "+64bit" "-target-feature" "+f" "-target-feature" "+d" "-target-feature" "+lsx" "-target-feature" "+lasx" "-target-feature" "+ual"
// CC1-LA464-NOT: "-target-feature"
// CC1-LA464: "-target-abi" "lp64d"

// CC1-LA64V1P0: "-target-cpu" "loongarch64"
// CC1-LA64V1P0-NOT: "-target-feature"
// CC1-LA64V1P0: "-target-feature" "+64bit" "-target-feature" "+d" "-target-feature" "+lsx" "-target-feature" "+ual"
// CC1-LA64V1P0-NOT: "-target-feature"
// CC1-LA64V1P0: "-target-abi" "lp64d"

// CC1-LA64V1P1: "-target-cpu" "loongarch64"
// CC1-LA64V1P1-NOT: "-target-feature"
// CC1-LA64V1P1: "-target-feature" "+64bit" "-target-feature" "+d" "-target-feature" "+lsx" "-target-feature" "+ual" "-target-feature" "+frecipe"
// CC1-LA64V1P1-NOT: "-target-feature"
// CC1-LA64V1P1: "-target-abi" "lp64d"

// CC1-LA664: "-target-cpu" "la664"
// CC1-LA664-NOT: "-target-feature"
// CC1-LA664: "-target-feature" "+64bit" "-target-feature" "+f" "-target-feature" "+d" "-target-feature" "+lsx" "-target-feature" "+lasx" "-target-feature" "+ual" "-target-feature" "+frecipe"
// CC1-LA664-NOT: "-target-feature"
// CC1-LA664: "-target-abi" "lp64d"

// IR-LOONGARCH64: attributes #[[#]] ={{.*}}"target-cpu"="loongarch64" {{.*}}"target-features"="+64bit,+d,+f,+ual"
// IR-LA464: attributes #[[#]] ={{.*}}"target-cpu"="la464" {{.*}}"target-features"="+64bit,+d,+f,+lasx,+lsx,+ual"
// IR-LA64V1P0: attributes #[[#]] ={{.*}}"target-cpu"="loongarch64" {{.*}}"target-features"="+64bit,+d,+lsx,+ual"
// IR-LA64V1P1: attributes #[[#]] ={{.*}}"target-cpu"="loongarch64" {{.*}}"target-features"="+64bit,+d,+frecipe,+lsx,+ual"
// IR-LA664: attributes #[[#]] ={{.*}}"target-cpu"="la664" {{.*}}"target-features"="+64bit,+d,+f,+frecipe,+lasx,+lsx,+ual"

int foo(void) {
  return 3;
}
