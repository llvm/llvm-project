// RUN: %clang -target x86_64-linux-gnu -fcatch-undefined-behavior %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UNDEFINED
// RUN: %clang -target x86_64-linux-gnu -fsanitize=undefined %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UNDEFINED
// CHECK-UNDEFINED: "-fsanitize={{((signed-integer-overflow|divide-by-zero|shift|unreachable|return|vla-bound|alignment|null|vptr|object-size|float-cast-overflow),?){11}"}}

// RUN: %clang -target x86_64-linux-gnu -fsanitize=thread,undefined -fno-thread-sanitizer -fno-sanitize=float-cast-overflow,vptr %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-PARTIAL-UNDEFINED
// CHECK-PARTIAL-UNDEFINED: "-fsanitize={{((signed-integer-overflow|divide-by-zero|shift|unreachable|return|vla-bound|alignment|null|object-size),?){9}"}}

// RUN: %clang -target x86_64-linux-gnu -fsanitize=vptr -fno-rtti %s -c -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-VPTR-NO-RTTI
// RUN: %clang -target x86_64-linux-gnu -fsanitize=undefined -fno-rtti %s -c -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-VPTR-NO-RTTI
// CHECK-VPTR-NO-RTTI: '-fsanitize=vptr' not allowed with '-fno-rtti'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=address,thread -fno-rtti %s -c -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-SANA-SANT
// CHECK-SANA-SANT: '-fsanitize=address' not allowed with '-fsanitize=thread'

// RUN: %clang -target x86_64-linux-gnu -faddress-sanitizer -fthread-sanitizer -fno-rtti %s -c -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-TSAN
// CHECK-ASAN-TSAN: '-faddress-sanitizer' not allowed with '-fthread-sanitizer'

// RUN: %clang -target x86_64-linux-gnu -fcatch-undefined-behavior -fthread-sanitizer -fno-thread-sanitizer -faddress-sanitizer -fno-address-sanitizer -c -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=CHECK-DEPRECATED
// CHECK-DEPRECATED: argument '-fcatch-undefined-behavior' is deprecated, use '-fsanitize=undefined' instead
// CHECK-DEPRECATED: argument '-fthread-sanitizer' is deprecated, use '-fsanitize=thread' instead
// CHECK-DEPRECATED: argument '-fno-thread-sanitizer' is deprecated, use '-fno-sanitize=thread' instead
// CHECK-DEPRECATED: argument '-faddress-sanitizer' is deprecated, use '-fsanitize=address' instead
// CHECK-DEPRECATED: argument '-fno-address-sanitizer' is deprecated, use '-fno-sanitize=address' instead
