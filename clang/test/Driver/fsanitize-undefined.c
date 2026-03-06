// RUN: %clang --target=x86_64-linux-gnu -fsanitize=undefined %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UNDEFINED
// CHECK-UNDEFINED: "-fsanitize={{((signed-integer-overflow|integer-divide-by-zero|function|shift-base|shift-exponent|unreachable|return|vla-bound|alignment|null|pointer-overflow|float-cast-overflow|array-bounds|enum|bool|builtin|returns-nonnull-attribute|nonnull-attribute),?){18}"}}

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=thread,undefined -fno-sanitize=thread -fno-sanitize=float-cast-overflow,vptr,bool,builtin,enum %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-PARTIAL-UNDEFINED
// CHECK-PARTIAL-UNDEFINED: "-fsanitize={{((signed-integer-overflow|integer-divide-by-zero|function|shift-base|shift-exponent|unreachable|return|vla-bound|alignment|null|pointer-overflow|array-bounds|returns-nonnull-attribute|nonnull-attribute),?){14}"}}

// RUN: %clang --target=x86_64-apple-darwin10 -fsanitize=undefined %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UNDEFINED-DARWIN
// CHECK-UNDEFINED-DARWIN: "-fsanitize={{((signed-integer-overflow|integer-divide-by-zero|function|shift-base|shift-exponent|unreachable|return|vla-bound|alignment|null|pointer-overflow|float-cast-overflow|array-bounds|enum|bool|builtin|returns-nonnull-attribute|nonnull-attribute),?){18}"}}

// RUN: %clang --target=i386-pc-win32 -fsanitize=undefined %s -### 2>&1 | FileCheck %s --check-prefixes=CHECK-UNDEFINED-WIN32,CHECK-UNDEFINED-MSVC
// RUN: %clang --target=i386-pc-win32 -fsanitize=undefined -x c++ %s -### 2>&1 | FileCheck %s --check-prefixes=CHECK-UNDEFINED-WIN32,CHECK-UNDEFINED-WIN-CXX,CHECK-UNDEFINED-MSVC
// RUN: %clang --target=x86_64-pc-win32 -fsanitize=undefined %s -### 2>&1 | FileCheck %s --check-prefixes=CHECK-UNDEFINED-WIN64,CHECK-UNDEFINED-MSVC
// RUN: %clang --target=x86_64-w64-mingw32 -fsanitize=undefined %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UNDEFINED-WIN64-MINGW
// RUN: %clang --target=x86_64-pc-win32 -fsanitize=undefined -x c++ %s -### 2>&1 | FileCheck %s --check-prefixes=CHECK-UNDEFINED-WIN64,CHECK-UNDEFINED-WIN-CXX,CHECK-UNDEFINED-MSVC
// CHECK-UNDEFINED-WIN32: "--dependent-lib={{[^"]*}}ubsan_standalone{{(-i386)?}}.lib"
// CHECK-UNDEFINED-WIN64: "--dependent-lib={{[^"]*}}ubsan_standalone{{(-x86_64)?}}.lib"
// CHECK-UNDEFINED-WIN64-MINGW: "--dependent-lib={{[^"]*}}libclang_rt.ubsan_standalone{{(-x86_64)?}}.a"
// CHECK-UNDEFINED-WIN-CXX: "--dependent-lib={{[^"]*}}ubsan_standalone_cxx{{[^"]*}}.lib"
// CHECK-UNDEFINED-MSVC-SAME: "-fsanitize={{((signed-integer-overflow|integer-divide-by-zero|shift-base|shift-exponent|unreachable|return|vla-bound|alignment|null|pointer-overflow|float-cast-overflow|array-bounds|enum|bool|builtin|returns-nonnull-attribute|nonnull-attribute|function),?){18}"}}
// CHECK-UNDEFINED-WIN64-MINGW-SAME: "-fsanitize={{((signed-integer-overflow|integer-divide-by-zero|shift-base|shift-exponent|unreachable|return|vla-bound|alignment|null|pointer-overflow|float-cast-overflow|array-bounds|enum|bool|builtin|returns-nonnull-attribute|nonnull-attribute|function),?){18}"}}

// RUN: %clang --target=i386-pc-win32 -fsanitize-coverage=bb %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-COVERAGE-WIN32
// CHECK-COVERAGE-WIN32: "--dependent-lib={{[^"]*}}ubsan_standalone{{(-i386)?}}.lib"
// RUN: %clang --target=x86_64-pc-win32 -fsanitize-coverage=bb %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-COVERAGE-WIN64
// CHECK-COVERAGE-WIN64: "--dependent-lib={{[^"]*}}ubsan_standalone{{(-x86_64)?}}.lib"

// RUN: %clang -fsanitize=shift -fno-sanitize=shift-base %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FSANITIZE-SHIFT-PARTIAL
// CHECK-FSANITIZE-SHIFT-PARTIAL: "-fsanitize=shift-exponent"

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=vptr -fsanitize-trap=vptr %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-VPTR-TRAP-UNDEF
// CHECK-VPTR-TRAP-UNDEF: error: unsupported argument 'vptr' to option '-fsanitize-trap='

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=vptr -fsanitize-undefined-trap-on-error %s -###

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=vptr -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-VPTR-NO-RTTI
// CHECK-VPTR-NO-RTTI: '-fsanitize=vptr' not allowed with '-fno-rtti'

// RUN: %clang -fsanitize=undefined -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UNDEFINED-NO-RTTI
// CHECK-UNDEFINED-NO-RTTI-NOT: vptr

// RUN: %clang --target=%itanium_abi_triple -fsanitize=integer %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-INTEGER -implicit-check-not="-fsanitize-address-use-after-scope"
// CHECK-INTEGER: "-fsanitize={{((signed-integer-overflow|unsigned-integer-overflow|integer-divide-by-zero|shift-base|shift-exponent|implicit-unsigned-integer-truncation|implicit-signed-integer-truncation|implicit-integer-sign-change|unsigned-shift-base),?){9}"}}

// RUN: %clang -fsanitize=implicit-integer-conversion %s -### 2>&1 | FileCheck %s --check-prefixes=CHECK-implicit-integer-conversion,CHECK-implicit-integer-conversion-RECOVER
// RUN: %clang -fsanitize=implicit-integer-conversion -fsanitize-recover=implicit-integer-conversion %s -### 2>&1 | FileCheck %s --check-prefixes=CHECK-implicit-integer-conversion,CHECK-implicit-integer-conversion-RECOVER
// RUN: %clang -fsanitize=implicit-integer-conversion -fno-sanitize-recover=implicit-integer-conversion %s -### 2>&1 | FileCheck %s --check-prefixes=CHECK-implicit-integer-conversion,CHECK-implicit-integer-conversion-NORECOVER
// RUN: %clang -fsanitize=implicit-integer-conversion -fsanitize-trap=implicit-integer-conversion %s -### 2>&1 | FileCheck %s --check-prefixes=CHECK-implicit-integer-conversion,CHECK-implicit-integer-conversion-TRAP
// CHECK-implicit-integer-conversion: "-fsanitize={{((implicit-unsigned-integer-truncation|implicit-signed-integer-truncation|implicit-integer-sign-change),?){3}"}}
// CHECK-implicit-integer-conversion-RECOVER: "-fsanitize-recover={{((implicit-unsigned-integer-truncation|implicit-signed-integer-truncation|implicit-integer-sign-change),?){3}"}}
// CHECK-implicit-integer-conversion-RECOVER-NOT: "-fno-sanitize-recover={{((implicit-unsigned-integer-truncation|implicit-signed-integer-truncation|implicit-integer-sign-change),?){3}"}}
// CHECK-implicit-integer-conversion-RECOVER-NOT: "-fsanitize-trap={{((implicit-unsigned-integer-truncation|implicit-signed-integer-truncation|implicit-integer-sign-change),?){3}"}}
// CHECK-implicit-integer-conversion-NORECOVER-NOT: "-fno-sanitize-recover={{((implicit-unsigned-integer-truncation|implicit-signed-integer-truncation|implicit-integer-sign-change),?){3}"}} // ???
// CHECK-implicit-integer-conversion-NORECOVER-NOT: "-fsanitize-recover={{((implicit-unsigned-integer-truncation|implicit-signed-integer-truncation|implicit-integer-sign-change),?){3}"}}
// CHECK-implicit-integer-conversion-NORECOVER-NOT: "-fsanitize-trap={{((implicit-unsigned-integer-truncation|implicit-signed-integer-truncation|implicit-integer-sign-change),?){3}"}}
// CHECK-implicit-integer-conversion-TRAP: "-fsanitize-trap={{((implicit-unsigned-integer-truncation|implicit-signed-integer-truncation|implicit-integer-sign-change),?){3}"}}
// CHECK-implicit-integer-conversion-TRAP-NOT: "-fsanitize-recover={{((implicit-unsigned-integer-truncation|implicit-signed-integer-truncation|implicit-integer-sign-change),?){3}"}}
// CHECK-implicit-integer-conversion-TRAP-NOT: "-fno-sanitize-recover={{((implicit-unsigned-integer-truncation|implicit-signed-integer-truncation|implicit-integer-sign-change),?){3}"}}

// RUN: %clang -fsanitize=implicit-integer-arithmetic-value-change %s -### 2>&1 | FileCheck %s --check-prefixes=CHECK-implicit-integer-arithmetic-value-change,CHECK-implicit-integer-arithmetic-value-change-RECOVER
// RUN: %clang -fsanitize=implicit-integer-arithmetic-value-change -fsanitize-recover=implicit-integer-arithmetic-value-change %s -### 2>&1 | FileCheck %s --check-prefixes=CHECK-implicit-integer-arithmetic-value-change,CHECK-implicit-integer-arithmetic-value-change-RECOVER
// RUN: %clang -fsanitize=implicit-integer-arithmetic-value-change -fno-sanitize-recover=implicit-integer-arithmetic-value-change %s -### 2>&1 | FileCheck %s --check-prefixes=CHECK-implicit-integer-arithmetic-value-change,CHECK-implicit-integer-arithmetic-value-change-NORECOVER
// RUN: %clang -fsanitize=implicit-integer-arithmetic-value-change -fsanitize-trap=implicit-integer-arithmetic-value-change %s -### 2>&1 | FileCheck %s --check-prefixes=CHECK-implicit-integer-arithmetic-value-change,CHECK-implicit-integer-arithmetic-value-change-TRAP
// CHECK-implicit-integer-arithmetic-value-change: "-fsanitize={{((implicit-signed-integer-truncation|implicit-integer-sign-change),?){2}"}}
// CHECK-implicit-integer-arithmetic-value-change-RECOVER: "-fsanitize-recover={{((implicit-signed-integer-truncation|implicit-integer-sign-change),?){2}"}}
// CHECK-implicit-integer-arithmetic-value-change-RECOVER-NOT: "-fno-sanitize-recover={{((implicit-signed-integer-truncation|implicit-integer-sign-change),?){2}"}}
// CHECK-implicit-integer-arithmetic-value-change-RECOVER-NOT: "-fsanitize-trap={{((implicit-signed-integer-truncation|implicit-integer-sign-change),?){2}"}}
// CHECK-implicit-integer-arithmetic-value-change-NORECOVER-NOT: "-fno-sanitize-recover={{((implicit-signed-integer-truncation|implicit-integer-sign-change),?){2}"}} // ???
// CHECK-implicit-integer-arithmetic-value-change-NORECOVER-NOT: "-fsanitize-recover={{((implicit-signed-integer-truncation|implicit-integer-sign-change),?){2}"}}
// CHECK-implicit-integer-arithmetic-value-change-NORECOVER-NOT: "-fsanitize-trap={{((implicit-signed-integer-truncation|implicit-integer-sign-change),?){2}"}}
// CHECK-implicit-integer-arithmetic-value-change-TRAP: "-fsanitize-trap={{((implicit-signed-integer-truncation|implicit-integer-sign-change),?){2}"}}
// CHECK-implicit-integer-arithmetic-value-change-TRAP-NOT: "-fsanitize-recover={{((implicit-signed-integer-truncation|implicit-integer-sign-change),?){2}"}}
// CHECK-implicit-integer-arithmetic-value-change-TRAP-NOT: "-fno-sanitize-recover={{((implicit-signed-integer-truncation|implicit-integer-sign-change),?){2}"}}

// RUN: %clang -fsanitize=implicit-integer-truncation %s -### 2>&1 | FileCheck %s --check-prefixes=CHECK-implicit-integer-truncation,CHECK-implicit-integer-truncation-RECOVER
// RUN: %clang -fsanitize=implicit-integer-truncation -fsanitize-recover=implicit-integer-truncation %s -### 2>&1 | FileCheck %s --check-prefixes=CHECK-implicit-integer-truncation,CHECK-implicit-integer-truncation-RECOVER
// RUN: %clang -fsanitize=implicit-integer-truncation -fno-sanitize-recover=implicit-integer-truncation %s -### 2>&1 | FileCheck %s --check-prefixes=CHECK-implicit-integer-truncation,CHECK-implicit-integer-truncation-NORECOVER
// RUN: %clang -fsanitize=implicit-integer-truncation -fsanitize-trap=implicit-integer-truncation %s -### 2>&1 | FileCheck %s --check-prefixes=CHECK-implicit-integer-truncation,CHECK-implicit-integer-truncation-TRAP
// CHECK-implicit-integer-truncation: "-fsanitize={{((implicit-unsigned-integer-truncation|implicit-signed-integer-truncation),?){2}"}}
// CHECK-implicit-integer-truncation-RECOVER: "-fsanitize-recover={{((implicit-unsigned-integer-truncation|implicit-signed-integer-truncation),?){2}"}}
// CHECK-implicit-integer-truncation-RECOVER-NOT: "-fno-sanitize-recover={{((implicit-unsigned-integer-truncation|implicit-signed-integer-truncation),?){2}"}}
// CHECK-implicit-integer-truncation-RECOVER-NOT: "-fsanitize-trap={{((implicit-unsigned-integer-truncation|implicit-signed-integer-truncation),?){2}"}}
// CHECK-implicit-integer-truncation-NORECOVER-NOT: "-fno-sanitize-recover={{((implicit-unsigned-integer-truncation|implicit-signed-integer-truncation),?){2}"}} // ???
// CHECK-implicit-integer-truncation-NORECOVER-NOT: "-fsanitize-recover={{((implicit-unsigned-integer-truncation|implicit-signed-integer-truncation),?){2}"}}
// CHECK-implicit-integer-truncation-NORECOVER-NOT: "-fsanitize-trap={{((implicit-unsigned-integer-truncation|implicit-signed-integer-truncation),?){2}"}}
// CHECK-implicit-integer-truncation-TRAP: "-fsanitize-trap={{((implicit-unsigned-integer-truncation|implicit-signed-integer-truncation),?){2}"}}
// CHECK-implicit-integer-truncation-TRAP-NOT: "-fsanitize-recover={{((implicit-unsigned-integer-truncation|implicit-signed-integer-truncation),?){2}"}}
// CHECK-implicit-integer-truncation-TRAP-NOT: "-fno-sanitize-recover={{((implicit-unsigned-integer-truncation|implicit-signed-integer-truncation),?){2}"}}

// RUN: %clang -fsanitize=bounds -### -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=CHECK-BOUNDS
// CHECK-BOUNDS: "-fsanitize={{((array-bounds|local-bounds),?){2}"}}

// RUN: %clang -fsanitize=float-divide-by-zero %s -### 2>&1 | FileCheck %s --check-prefixes=CHECK-DIVBYZERO,CHECK-DIVBYZERO-RECOVER
// RUN: %clang -fsanitize=float-divide-by-zero %s -fno-sanitize-recover=float-divide-by-zero -### 2>&1 | FileCheck %s --check-prefixes=CHECK-DIVBYZERO,CHECK-DIVBYZERO-NORECOVER
// RUN: %clang -fsanitize=float-divide-by-zero -fsanitize-trap=float-divide-by-zero %s -### 2>&1 | FileCheck %s --check-prefixes=CHECK-DIVBYZERO,CHECK-DIVBYZERO-NORECOVER,CHECK-DIVBYZERO-TRAP
// CHECK-DIVBYZERO: "-fsanitize=float-divide-by-zero"
// CHECK-DIVBYZERO-RECOVER: "-fsanitize-recover=float-divide-by-zero"
// CHECK-DIVBYZERO-NORECOVER-NOT: "-fsanitize-recover=float-divide-by-zero"
// CHECK-DIVBYZERO-TRAP: "-fsanitize-trap=float-divide-by-zero"

// RUN: %clang -fsanitize=float-divide-by-zero -fsanitize-minimal-runtime %s -### 2>&1 | FileCheck %s --check-prefixes=CHECK-DIVBYZERO,CHECK-DIVBYZERO-MINIMAL
// CHECK-DIVBYZERO-MINIMAL: "-fsanitize-minimal-runtime"

// RUN: %clang -fsanitize=undefined,float-divide-by-zero %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-DIVBYZERO-UBSAN
// CHECK-DIVBYZERO-UBSAN: "-fsanitize={{.*}},float-divide-by-zero,{{.*}}"

// RUN: %clang --target=x86_64-apple-darwin10 -fsanitize=function %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FUNCTION
// RUN: %clang --target=aarch64-unknown-linux-gnu -fsanitize=function %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FUNCTION
// RUN: %clang --target=riscv64-pc-freebsd -fsanitize=function %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FUNCTION
// CHECK-FUNCTION: -cc1{{.*}}"-fsanitize=function" "-fsanitize-recover=function"

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=undefined,function -mcmodel=large %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UBSAN-FUNCTION-CODE-MODEL
// CHECK-UBSAN-FUNCTION-CODE-MODEL: error: invalid argument '-fsanitize=function' only allowed with '-mcmodel=small'

// RUN: %clang --target=i386--solaris -fsanitize=function %s -### 2>&1 | FileCheck %s -check-prefix=FUNCTION-SOLARIS
// RUN: %clang --target=x86_64--solaris -fsanitize=function %s -### 2>&1 | FileCheck %s -check-prefix=FUNCTION-SOLARIS
// FUNCTION-SOLARIS: "-fsanitize=function"
