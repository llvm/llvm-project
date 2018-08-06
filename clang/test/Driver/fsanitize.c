// RUN: %clang -target x86_64-linux-gnu -fsanitize=undefined -fsanitize-trap=undefined %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UNDEFINED-TRAP
// RUN: %clang -target x86_64-linux-gnu -fsanitize=undefined -fsanitize-trap=undefined -fno-sanitize-trap=signed-integer-overflow %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UNDEFINED-TRAP2
// RUN: %clang -target x86_64-linux-gnu -fsanitize=undefined -fsanitize-undefined-trap-on-error %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UNDEFINED-TRAP
// RUN: %clang -target x86_64-linux-gnu -fsanitize=undefined-trap -fsanitize-undefined-trap-on-error %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UNDEFINED-TRAP
// RUN: %clang -target x86_64-linux-gnu -fsanitize-undefined-trap-on-error -fsanitize=undefined-trap %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UNDEFINED-TRAP
// CHECK-UNDEFINED-TRAP-NOT: -fsanitize-recover
// CHECK-UNDEFINED-TRAP: "-fsanitize={{((signed-integer-overflow|integer-divide-by-zero|float-divide-by-zero|shift-base|shift-exponent|unreachable|return|vla-bound|alignment|null|pointer-overflow|float-cast-overflow|array-bounds|enum|bool|builtin|returns-nonnull-attribute|nonnull-attribute|function),?){19}"}}
// CHECK-UNDEFINED-TRAP: "-fsanitize-trap=alignment,array-bounds,bool,builtin,enum,float-cast-overflow,float-divide-by-zero,function,integer-divide-by-zero,nonnull-attribute,null,pointer-overflow,return,returns-nonnull-attribute,shift-base,shift-exponent,signed-integer-overflow,unreachable,vla-bound"
// CHECK-UNDEFINED-TRAP2: "-fsanitize-trap=alignment,array-bounds,bool,builtin,enum,float-cast-overflow,float-divide-by-zero,function,integer-divide-by-zero,nonnull-attribute,null,pointer-overflow,return,returns-nonnull-attribute,shift-base,shift-exponent,unreachable,vla-bound"

// RUN: %clang -target x86_64-linux-gnu -fsanitize=undefined %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UNDEFINED
// CHECK-UNDEFINED: "-fsanitize={{((signed-integer-overflow|integer-divide-by-zero|float-divide-by-zero|function|shift-base|shift-exponent|unreachable|return|vla-bound|alignment|null|vptr|pointer-overflow|float-cast-overflow|array-bounds|enum|bool|builtin|returns-nonnull-attribute|nonnull-attribute),?){20}"}}

// RUN: %clang -target x86_64-apple-darwin10 -fsanitize=undefined %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UNDEFINED-DARWIN
// CHECK-UNDEFINED-DARWIN: "-fsanitize={{((signed-integer-overflow|integer-divide-by-zero|float-divide-by-zero|function|shift-base|shift-exponent|unreachable|return|vla-bound|alignment|null|pointer-overflow|float-cast-overflow|array-bounds|enum|bool|builtin|returns-nonnull-attribute|nonnull-attribute),?){19}"}}

// RUN: %clang -target i386-pc-win32 -fsanitize=undefined %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UNDEFINED-WIN --check-prefix=CHECK-UNDEFINED-WIN32
// RUN: %clang -target i386-pc-win32 -fsanitize=undefined -x c++ %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UNDEFINED-WIN --check-prefix=CHECK-UNDEFINED-WIN32 --check-prefix=CHECK-UNDEFINED-WIN-CXX
// RUN: %clang -target x86_64-pc-win32 -fsanitize=undefined %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UNDEFINED-WIN --check-prefix=CHECK-UNDEFINED-WIN64
// RUN: %clang -target x86_64-pc-win32 -fsanitize=undefined -x c++ %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UNDEFINED-WIN --check-prefix=CHECK-UNDEFINED-WIN64 --check-prefix=CHECK-UNDEFINED-WIN-CXX
// CHECK-UNDEFINED-WIN32: "--dependent-lib={{[^"]*}}ubsan_standalone-i386.lib"
// CHECK-UNDEFINED-WIN64: "--dependent-lib={{[^"]*}}ubsan_standalone-x86_64.lib"
// CHECK-UNDEFINED-WIN-CXX: "--dependent-lib={{[^"]*}}ubsan_standalone_cxx{{[^"]*}}.lib"
// CHECK-UNDEFINED-WIN-SAME: "-fsanitize={{((signed-integer-overflow|integer-divide-by-zero|float-divide-by-zero|shift-base|shift-exponent|unreachable|return|vla-bound|alignment|null|pointer-overflow|float-cast-overflow|array-bounds|enum|bool|builtin|returns-nonnull-attribute|nonnull-attribute),?){18}"}}

// RUN: %clang -target i386-pc-win32 -fsanitize-coverage=bb %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-COVERAGE-WIN32
// CHECK-COVERAGE-WIN32: "--dependent-lib={{[^"]*}}ubsan_standalone-i386.lib"
// RUN: %clang -target x86_64-pc-win32 -fsanitize-coverage=bb %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-COVERAGE-WIN64
// CHECK-COVERAGE-WIN64: "--dependent-lib={{[^"]*}}ubsan_standalone-x86_64.lib"

// RUN: %clang -target x86_64-linux-gnu -fsanitize=integer %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-INTEGER -implicit-check-not="-fsanitize-address-use-after-scope"
// CHECK-INTEGER: "-fsanitize={{((signed-integer-overflow|unsigned-integer-overflow|integer-divide-by-zero|shift-base|shift-exponent),?){5}"}}

// RUN: %clang -fsanitize=bounds -### -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=CHECK-BOUNDS
// CHECK-BOUNDS: "-fsanitize={{((array-bounds|local-bounds),?){2}"}}

// RUN: %clang -target x86_64-linux-gnu -fsanitize=all %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FSANITIZE-ALL
// CHECK-FSANITIZE-ALL: error: unsupported argument 'all' to option 'fsanitize='

// RUN: %clang -target x86_64-linux-gnu -fsanitize=address,undefined -fno-sanitize=all -fsanitize=thread %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FNO-SANITIZE-ALL
// CHECK-FNO-SANITIZE-ALL: "-fsanitize=thread"

// RUN: %clang -target x86_64-linux-gnu -fsanitize=thread,undefined -fno-sanitize=thread -fno-sanitize=float-cast-overflow,vptr,bool,builtin,enum %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-PARTIAL-UNDEFINED
// CHECK-PARTIAL-UNDEFINED: "-fsanitize={{((signed-integer-overflow|integer-divide-by-zero|float-divide-by-zero|function|shift-base|shift-exponent|unreachable|return|vla-bound|alignment|null|pointer-overflow|array-bounds|returns-nonnull-attribute|nonnull-attribute),?){15}"}}

// RUN: %clang -target x86_64-linux-gnu -fsanitize=shift -fno-sanitize=shift-base %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FSANITIZE-SHIFT-PARTIAL
// CHECK-FSANITIZE-SHIFT-PARTIAL: "-fsanitize=shift-exponent"

// RUN: %clang -target x86_64-linux-gnu -fsanitize=vptr -fsanitize-trap=undefined %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-VPTR-TRAP-UNDEF
// RUN: %clang -target x86_64-linux-gnu -fsanitize=vptr -fsanitize-undefined-trap-on-error %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-VPTR-TRAP-UNDEF
// CHECK-VPTR-TRAP-UNDEF: error: invalid argument '-fsanitize=vptr' not allowed with '-fsanitize-trap=undefined'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=vptr -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-VPTR-NO-RTTI
// CHECK-VPTR-NO-RTTI: '-fsanitize=vptr' not allowed with '-fno-rtti'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=undefined -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UNDEFINED-NO-RTTI
// CHECK-UNDEFINED-NO-RTTI-NOT: vptr

// RUN: %clang -target x86_64-linux-gnu -fsanitize=address,thread -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANA-SANT
// CHECK-SANA-SANT: '-fsanitize=address' not allowed with '-fsanitize=thread'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=address,memory -pie -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANA-SANM
// CHECK-SANA-SANM: '-fsanitize=address' not allowed with '-fsanitize=memory'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=thread,memory -pie -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANT-SANM
// CHECK-SANT-SANM: '-fsanitize=thread' not allowed with '-fsanitize=memory'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=memory,thread -pie -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANM-SANT
// CHECK-SANM-SANT: '-fsanitize=thread' not allowed with '-fsanitize=memory'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=leak,thread -pie -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANL-SANT
// CHECK-SANL-SANT: '-fsanitize=leak' not allowed with '-fsanitize=thread'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=leak,memory -pie -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANL-SANM
// CHECK-SANL-SANM: '-fsanitize=leak' not allowed with '-fsanitize=memory'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=kernel-address,thread -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANKA-SANT
// CHECK-SANKA-SANT: '-fsanitize=kernel-address' not allowed with '-fsanitize=thread'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=kernel-address,memory -pie -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANKA-SANM
// CHECK-SANKA-SANM: '-fsanitize=kernel-address' not allowed with '-fsanitize=memory'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=kernel-address,address -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANKA-SANA
// CHECK-SANKA-SANA: '-fsanitize=kernel-address' not allowed with '-fsanitize=address'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=kernel-address,leak -pie -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANKA-SANL
// CHECK-SANKA-SANL: '-fsanitize=kernel-address' not allowed with '-fsanitize=leak'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=kernel-hwaddress,thread -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANKHA-SANT
// CHECK-SANKHA-SANT: '-fsanitize=kernel-hwaddress' not allowed with '-fsanitize=thread'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=kernel-hwaddress,memory -pie -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANKHA-SANM
// CHECK-SANKHA-SANM: '-fsanitize=kernel-hwaddress' not allowed with '-fsanitize=memory'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=kernel-hwaddress,address -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANKHA-SANA
// CHECK-SANKHA-SANA: '-fsanitize=kernel-hwaddress' not allowed with '-fsanitize=address'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=kernel-hwaddress,leak -pie -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANKHA-SANL
// CHECK-SANKHA-SANL: '-fsanitize=kernel-hwaddress' not allowed with '-fsanitize=leak'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=kernel-hwaddress,hwaddress -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANKHA-SANHA
// CHECK-SANKHA-SANHA: '-fsanitize=kernel-hwaddress' not allowed with '-fsanitize=hwaddress'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=kernel-hwaddress,kernel-address -pie -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANKHA-SANKA
// CHECK-SANKHA-SANKA: '-fsanitize=kernel-hwaddress' not allowed with '-fsanitize=kernel-address'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=hwaddress,thread -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANHA-SANT
// CHECK-SANHA-SANT: '-fsanitize=hwaddress' not allowed with '-fsanitize=thread'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=hwaddress,memory -pie -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANHA-SANM
// CHECK-SANHA-SANM: '-fsanitize=hwaddress' not allowed with '-fsanitize=memory'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=hwaddress,address -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANHA-SANA
// CHECK-SANHA-SANA: '-fsanitize=hwaddress' not allowed with '-fsanitize=address'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=efficiency-cache-frag,address -pie -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANE-SANA
// RUN: %clang -target x86_64-linux-gnu -fsanitize=efficiency-working-set,address -pie -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANE-SANA
// CHECK-SANE-SANA: '-fsanitize=efficiency-{{.*}}' not allowed with '-fsanitize=address'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=efficiency-cache-frag,leak -pie -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANE-SANL
// RUN: %clang -target x86_64-linux-gnu -fsanitize=efficiency-working-set,leak -pie -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANE-SANL
// CHECK-SANE-SANL: '-fsanitize=efficiency-{{.*}}' not allowed with '-fsanitize=leak'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=efficiency-cache-frag,thread -pie -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANE-SANT
// RUN: %clang -target x86_64-linux-gnu -fsanitize=efficiency-working-set,thread -pie -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANE-SANT
// CHECK-SANE-SANT: '-fsanitize=efficiency-{{.*}}' not allowed with '-fsanitize=thread'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=efficiency-cache-frag,memory -pie -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANE-SANM
// RUN: %clang -target x86_64-linux-gnu -fsanitize=efficiency-working-set,memory -pie -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANE-SANM
// CHECK-SANE-SANM: '-fsanitize=efficiency-{{.*}}' not allowed with '-fsanitize=memory'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=efficiency-cache-frag,kernel-address -pie -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANE-SANKA
// RUN: %clang -target x86_64-linux-gnu -fsanitize=efficiency-working-set,kernel-address -pie -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANE-SANKA
// CHECK-SANE-SANKA: '-fsanitize=efficiency-{{.*}}' not allowed with '-fsanitize=kernel-address'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=address -fsanitize-address-use-after-scope %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-USE-AFTER-SCOPE
// RUN: %clang_cl --target=x86_64-windows -fsanitize=address -fsanitize-address-use-after-scope -### -- %s 2>&1 | FileCheck %s --check-prefix=CHECK-USE-AFTER-SCOPE
// CHECK-USE-AFTER-SCOPE: -cc1{{.*}}-fsanitize-address-use-after-scope

// RUN: %clang -target x86_64-linux-gnu -fsanitize=address -fno-sanitize-address-use-after-scope %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-USE-AFTER-SCOPE-OFF
// RUN: %clang_cl --target=x86_64-windows -fsanitize=address -fno-sanitize-address-use-after-scope -### -- %s 2>&1 | FileCheck %s --check-prefix=CHECK-USE-AFTER-SCOPE-OFF
// CHECK-USE-AFTER-SCOPE-OFF-NOT: -cc1{{.*}}address-use-after-scope

// RUN: %clang -target x86_64-linux-gnu -fsanitize=address -fno-sanitize-address-use-after-scope -fsanitize-address-use-after-scope %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-USE-AFTER-SCOPE-BOTH
// RUN: %clang_cl --target=x86_64-windows -fsanitize=address -fno-sanitize-address-use-after-scope -fsanitize-address-use-after-scope -### -- %s 2>&1 | FileCheck %s --check-prefix=CHECK-USE-AFTER-SCOPE-BOTH
// CHECK-USE-AFTER-SCOPE-BOTH: -cc1{{.*}}-fsanitize-address-use-after-scope

// RUN: %clang -target x86_64-linux-gnu -fsanitize=address -fsanitize-address-use-after-scope -fno-sanitize-address-use-after-scope %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-USE-AFTER-SCOPE-BOTH-OFF
// CHECK-USE-AFTER-SCOPE-BOTH-OFF-NOT: -cc1{{.*}}address-use-after-scope

// RUN: %clang -target x86_64-linux-gnu -fsanitize=address %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-WITHOUT-USE-AFTER-SCOPE
// CHECK-ASAN-WITHOUT-USE-AFTER-SCOPE: -cc1{{.*}}address-use-after-scope

// RUN: %clang -target x86_64-linux-gnu -fsanitize=address -fsanitize-address-globals-dead-stripping %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-GLOBALS
// RUN: %clang -target x86_64-linux-gnu -fsanitize=address %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ASAN-GLOBALS
// RUN: %clang_cl --target=x86_64-windows-msvc -fsanitize=address -fsanitize-address-globals-dead-stripping -### -- %s 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-GLOBALS
// RUN: %clang_cl --target=x86_64-windows-msvc -fsanitize=address -### -- %s 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-GLOBALS
// CHECK-ASAN-GLOBALS: -cc1{{.*}}-fsanitize-address-globals-dead-stripping
// CHECK-NO-ASAN-GLOBALS-NOT: -cc1{{.*}}-fsanitize-address-globals-dead-stripping

// RUN: %clang -target x86_64-linux-gnu -fsanitize-memory-track-origins -pie %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ONLY-TRACK-ORIGINS
// CHECK-ONLY-TRACK-ORIGINS: warning: argument unused during compilation: '-fsanitize-memory-track-origins'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=memory -fno-sanitize=memory -fsanitize-memory-track-origins -pie %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TRACK-ORIGINS-DISABLED-MSAN
// CHECK-TRACK-ORIGINS-DISABLED-MSAN-NOT: warning: argument unused

// RUN: %clang -target x86_64-linux-gnu -fsanitize=address %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-EXTRA-TRACK-ORIGINS
// CHECK-NO-EXTRA-TRACK-ORIGINS-NOT: "-fsanitize-memory-track-origins"

// RUN: %clang -target x86_64-linux-gnu -fsanitize=address -fsanitize=alignment -fsanitize=vptr -fno-sanitize=vptr %s -### 2>&1
// OK

// RUN: %clang -target x86_64-linux-gnu -fsanitize=memory -pie %s -### 2>&1
// OK

// RUN: %clang -target x86_64-linux-gnu -fsanitize=memory -fsanitize-memory-track-origins -pie %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TRACK-ORIGINS-2
// RUN: %clang -target x86_64-linux-gnu -fsanitize=memory -fsanitize-memory-track-origins=1 -pie %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TRACK-ORIGINS-1
// RUN: %clang -target x86_64-linux-gnu -fsanitize=memory -fsanitize-memory-track-origins=1 -fsanitize-memory-track-origins -pie %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TRACK-ORIGINS-2
// RUN: %clang -target x86_64-linux-gnu -fsanitize=memory -fsanitize-memory-track-origins=2 -fsanitize-memory-track-origins -pie %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TRACK-ORIGINS-2
// RUN: %clang -target x86_64-linux-gnu -fsanitize=memory -fno-sanitize-memory-track-origins -fsanitize-memory-track-origins -pie %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TRACK-ORIGINS-2
// RUN: %clang -target x86_64-linux-gnu -fsanitize=memory -fsanitize-memory-track-origins=0 -fsanitize-memory-track-origins=1 -pie %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TRACK-ORIGINS-1
// RUN: %clang -target x86_64-linux-gnu -fsanitize=memory -fsanitize-memory-track-origins=0 -fsanitize-memory-track-origins -pie %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TRACK-ORIGINS-2

// CHECK-TRACK-ORIGINS-1: -fsanitize-memory-track-origins=1

// RUN: %clang -target x86_64-linux-gnu -fsanitize=memory -fno-sanitize-memory-track-origins -pie %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-TRACK-ORIGINS
// RUN: %clang -target x86_64-linux-gnu -fsanitize=memory -fsanitize-memory-track-origins=0 -pie %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-TRACK-ORIGINS
// RUN: %clang -target x86_64-linux-gnu -fsanitize=memory -fsanitize-memory-track-origins -fno-sanitize-memory-track-origins -pie %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-TRACK-ORIGINS
// RUN: %clang -target x86_64-linux-gnu -fsanitize=memory -fsanitize-memory-track-origins -fsanitize-memory-track-origins=0 -pie %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-TRACK-ORIGINS
// RUN: %clang -target x86_64-linux-gnu -fsanitize=memory -fsanitize-memory-track-origins=2 -fno-sanitize-memory-track-origins -pie %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-TRACK-ORIGINS
// RUN: %clang -target x86_64-linux-gnu -fsanitize=memory -fsanitize-memory-track-origins=2 -fsanitize-memory-track-origins=0 -pie %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-TRACK-ORIGINS
// CHECK-NO-TRACK-ORIGINS-NOT: sanitize-memory-track-origins

// RUN: %clang -target x86_64-linux-gnu -fsanitize=memory -fsanitize-memory-track-origins=2 -pie %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TRACK-ORIGINS-2
// CHECK-TRACK-ORIGINS-2: -fsanitize-memory-track-origins=2

// RUN: %clang -target x86_64-linux-gnu -fsanitize=memory -fsanitize-memory-track-origins=3 -pie %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TRACK-ORIGINS-3
// CHECK-TRACK-ORIGINS-3: error: invalid value '3' in '-fsanitize-memory-track-origins=3'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=memory -fsanitize-memory-use-after-dtor %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-USE-AFTER-DTOR
// RUN: %clang -target x86_64-linux-gnu -fsanitize=memory -fno-sanitize-memory-use-after-dtor -fsanitize-memory-use-after-dtor %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-USE-AFTER-DTOR
// RUN: %clang -target x86_64-linux-gnu -fsanitize=memory %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-USE-AFTER-DTOR
// CHECK-USE-AFTER-DTOR: -cc1{{.*}}-fsanitize-memory-use-after-dtor

// RUN: %clang -target x86_64-linux-gnu -fsanitize=memory -fno-sanitize-memory-use-after-dtor %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-USE-AFTER-DTOR-OFF
// RUN: %clang -target x86_64-linux-gnu -fsanitize=memory -fsanitize-memory-use-after-dtor -fno-sanitize-memory-use-after-dtor %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-USE-AFTER-DTOR-OFF
// CHECK-USE-AFTER-DTOR-OFF-NOT: -cc1{{.*}}memory-use-after-dtor

// RUN: %clang -target x86_64-linux-gnu -fsanitize=address -fsanitize-address-field-padding=0 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-FIELD-PADDING-0
// CHECK-ASAN-FIELD-PADDING-0-NOT: -fsanitize-address-field-padding
// RUN: %clang -target x86_64-linux-gnu -fsanitize=address -fsanitize-address-field-padding=1 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-FIELD-PADDING-1
// CHECK-ASAN-FIELD-PADDING-1: -fsanitize-address-field-padding=1
// RUN: %clang -target x86_64-linux-gnu -fsanitize=address -fsanitize-address-field-padding=2 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-FIELD-PADDING-2
// CHECK-ASAN-FIELD-PADDING-2: -fsanitize-address-field-padding=2
// RUN: %clang -target x86_64-linux-gnu -fsanitize=address -fsanitize-address-field-padding=3 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-FIELD-PADDING-3
// CHECK-ASAN-FIELD-PADDING-3: error: invalid value '3' in '-fsanitize-address-field-padding=3'
// RUN: %clang -target x86_64-linux-gnu -fsanitize-address-field-padding=2 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-FIELD-PADDING-NO-ASAN
// CHECK-ASAN-FIELD-PADDING-NO-ASAN: warning: argument unused during compilation: '-fsanitize-address-field-padding=2'
// RUN: %clang -target x86_64-linux-gnu -fsanitize-address-field-padding=2 -fsanitize=address -fno-sanitize=address %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-FIELD-PADDING-DISABLED-ASAN
// CHECK-ASAN-FIELD-PADDING-DISABLED-ASAN-NOT: warning: argument unused


// RUN: %clang -target x86_64-linux-gnu -fsanitize=vptr -fno-sanitize=vptr -fsanitize=undefined,address %s -### 2>&1
// OK

// RUN: %clang -target x86_64-linux-gnu -fsanitize=thread %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-PIE
// RUN: %clang -target x86_64-linux-gnu -fsanitize=memory %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-PIE
// RUN: %clang -target x86_64-unknown-freebsd -fsanitize=memory %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-PIE
// RUN: %clang -target aarch64-linux-gnu -fsanitize=memory %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-PIE
// RUN: %clang -target arm-linux-androideabi -fsanitize=address %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-PIC-NO-PIE
// RUN: %clang -target arm-linux-androideabi24 -fsanitize=address %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-PIE
// RUN: %clang -target aarch64-linux-android -fsanitize=address %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-PIE
// RUN: %clang -target x86_64-linux-gnu -fsanitize=address %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-PIE
// RUN: %clang -target i386-linux-gnu -fsanitize=address %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-PIE

// CHECK-NO-PIE-NOT: "-pie"
// CHECK-NO-PIE: "-mrelocation-model" "static"
// CHECK-NO-PIE-NOT: "-pie"

// CHECK-PIC-NO-PIE-NOT: "-pie"
// CHECK-PIC-NO-PIE: "-mrelocation-model" "pic"
// CHECK-PIC-NO-PIE-NOT: "-pie"

// CHECK-PIE: "-mrelocation-model" "pic" "-pic-level" "2" "-pic-is-pie"
// CHECK-PIE: "-pie"

// RUN: %clang -target arm-linux-androideabi %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ANDROID-NO-ASAN
// CHECK-ANDROID-NO-ASAN: "-mrelocation-model" "pic"

// RUN: %clang -target x86_64-linux-gnu %s -fsanitize=undefined -### 2>&1 | FileCheck %s --check-prefix=CHECK-RECOVER-UBSAN
// RUN: %clang -target x86_64-linux-gnu %s -fsanitize=undefined -fsanitize-recover -### 2>&1 | FileCheck %s --check-prefix=CHECK-RECOVER-UBSAN
// RUN: %clang -target x86_64-linux-gnu %s -fsanitize=undefined -fsanitize-recover=all -### 2>&1 | FileCheck %s --check-prefix=CHECK-RECOVER-UBSAN
// RUN: %clang -target x86_64-linux-gnu %s -fsanitize=undefined -fno-sanitize-recover -fsanitize-recover=undefined -### 2>&1 | FileCheck %s --check-prefix=CHECK-RECOVER-UBSAN
// RUN: %clang -target x86_64-linux-gnu %s -fsanitize=undefined -fno-sanitize-recover=undefined -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-RECOVER-UBSAN
// RUN: %clang -target x86_64-linux-gnu %s -fsanitize=undefined -fno-sanitize-recover=all -fsanitize-recover=thread -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-RECOVER-UBSAN
// RUN: %clang -target x86_64-linux-gnu %s -fsanitize=undefined -fsanitize-recover=all -fno-sanitize-recover=undefined -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-RECOVER-UBSAN
// CHECK-RECOVER-UBSAN: "-fsanitize-recover={{((signed-integer-overflow|integer-divide-by-zero|float-divide-by-zero|function|shift-base|shift-exponent|vla-bound|alignment|null|vptr|pointer-overflow|float-cast-overflow|array-bounds|enum|bool|builtin|returns-nonnull-attribute|nonnull-attribute),?){18}"}}
// CHECK-NO-RECOVER-UBSAN-NOT: sanitize-recover

// RUN: %clang -target x86_64-linux-gnu %s -fsanitize=undefined -fno-sanitize-recover=all -fsanitize-recover=object-size,shift-base -### 2>&1 | FileCheck %s --check-prefix=CHECK-PARTIAL-RECOVER
// CHECK-PARTIAL-RECOVER: "-fsanitize-recover={{((shift-base),?){1}"}}

// RUN: %clang -target x86_64-linux-gnu %s -fsanitize=address -fsanitize-recover=all -### 2>&1 | FileCheck %s --check-prefix=CHECK-RECOVER-ASAN
// CHECK-RECOVER-ASAN: "-fsanitize-recover=address"

// RUN: %clang -target x86_64-linux-gnu %s -fsanitize=undefined -fsanitize-recover=foobar,object-size,unreachable -### 2>&1 | FileCheck %s --check-prefix=CHECK-DIAG-RECOVER
// CHECK-DIAG-RECOVER: unsupported argument 'foobar' to option 'fsanitize-recover='
// CHECK-DIAG-RECOVER: unsupported argument 'unreachable' to option 'fsanitize-recover='

// RUN: %clang -target x86_64-linux-gnu %s -fsanitize=undefined -fsanitize-recover -fno-sanitize-recover -### 2>&1 | FileCheck %s --check-prefix=CHECK-DEPRECATED-RECOVER
// CHECK-DEPRECATED-RECOVER: argument '-fsanitize-recover' is deprecated, use '-fsanitize-recover=undefined,integer' or '-fsanitize-recover=all' instead
// CHECK-DEPRECATED-RECOVER: argument '-fno-sanitize-recover' is deprecated, use '-fno-sanitize-recover=undefined,integer' or '-fno-sanitize-recover=all' instead
// CHECK-DEPRECATED-RECOVER-NOT: is deprecated

// RUN: %clang -target x86_64-linux-gnu %s -fsanitize=kernel-address -fno-sanitize-recover=kernel-address -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-RECOVER-KASAN
// RUN: %clang -target x86_64-linux-gnu %s -fsanitize=kernel-hwaddress -fno-sanitize-recover=kernel-hwaddress -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-RECOVER-KHWASAN
// CHECK-NO-RECOVER-KASAN: unsupported argument 'kernel-address' to option 'fno-sanitize-recover='
// CHECK-NO-RECOVER-KHWASAN: unsupported argument 'kernel-hwaddress' to option 'fno-sanitize-recover='

// RUN: %clang -target x86_64-linux-gnu -fsanitize=leak %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANL
// CHECK-SANL: "-fsanitize=leak"

// RUN: %clang -target x86_64-linux-gnu -fsanitize=address,leak -fno-sanitize=address %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANA-SANL-NO-SANA
// CHECK-SANA-SANL-NO-SANA: "-fsanitize=leak"

// RUN: %clang -target i686-linux-gnu -fsanitize=leak %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANL-X86
// CHECK-SANL-X86: "-fsanitize=leak"

// RUN: %clang -target i686-linux-gnu -fsanitize=address,leak -fno-sanitize=address %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANA-SANL-NO-SANA-X86
// CHECK-SANA-SANL-NO-SANA-X86: "-fsanitize=leak"

// RUN: %clang -target arm-linux-gnu -fsanitize=leak %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANL-ARM
// CHECK-SANL-ARM: "-fsanitize=leak"

// RUN: %clang -target arm-linux-gnu -fsanitize=address,leak -fno-sanitize=address %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANA-SANL-NO-SANA-ARM
// CHECK-SANA-SANL-NO-SANA-ARM: "-fsanitize=leak"

// RUN: %clang -target thumb-linux -fsanitize=leak %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANL-THUMB
// CHECK-SANL-THUMB: "-fsanitize=leak"

// RUN: %clang -target thumb-linux -fsanitize=address,leak -fno-sanitize=address %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANA-SANL-NO-SANA-THUMB
// CHECK-SANA-SANL-NO-SANA-THUMB: "-fsanitize=leak"

// RUN: %clang -target armeb-linux-gnu -fsanitize=leak %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANL-ARMEB
// CHECK-SANL-ARMEB: "-fsanitize=leak"

// RUN: %clang -target armeb-linux-gnu -fsanitize=address,leak -fno-sanitize=address %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANA-SANL-NO-SANA-ARMEB
// CHECK-SANA-SANL-NO-SANA-ARMEB: "-fsanitize=leak"

// RUN: %clang -target thumbeb-linux -fsanitize=leak %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANL-THUMBEB
// CHECK-SANL-THUMBEB: "-fsanitize=leak"

// RUN: %clang -target thumbeb-linux -fsanitize=address,leak -fno-sanitize=address %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANA-SANL-NO-SANA-THUMBEB
// CHECK-SANA-SANL-NO-SANA-THUMBEB: "-fsanitize=leak"

// RUN: %clang -target mips-unknown-linux -fsanitize=leak %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANL-MIPS
// CHECK-SANL-MIPS: unsupported option '-fsanitize=leak' for target 'mips-unknown-linux'

// RUN: %clang -target mips-unknown-freebsd -fsanitize=leak %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANL-MIPS-FREEBSD
// CHECK-SANL-MIPS-FREEBSD: unsupported option '-fsanitize=leak' for target 'mips-unknown-freebsd'

// RUN: %clang -target mips64-unknown-freebsd -fsanitize=leak %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANL-MIPS64-FREEBSD
// CHECK-SANL-MIPS64-FREEBSD: "-fsanitize=leak"

// RUN: %clang -target powerpc64-unknown-linux -fsanitize=leak %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANL-PPC64
// RUN: %clang -target powerpc64le-unknown-linux -fsanitize=leak %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANL-PPC64
// CHECK-SANL-PPC64: "-fsanitize=leak"
// RUN: %clang -target powerpc-unknown-linux -fsanitize=leak %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANL-PPC
// CHECK-SANL-PPC: unsupported option '-fsanitize=leak' for target 'powerpc-unknown-linux'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=memory %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-MSAN
// CHECK-MSAN: "-fno-assume-sane-operator-new"
// RUN: %clang -target x86_64-linux-gnu -fsanitize=address %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN
// CHECK-ASAN: "-fno-assume-sane-operator-new"

// RUN: %clang -target x86_64-linux-gnu -fsanitize=zzz %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-DIAG1
// CHECK-DIAG1: unsupported argument 'zzz' to option 'fsanitize='
// CHECK-DIAG1-NOT: unsupported argument 'zzz' to option 'fsanitize='

// RUN: %clang -target x86_64-apple-darwin10 -fsanitize=memory %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-MSAN-DARWIN
// CHECK-MSAN-DARWIN: unsupported option '-fsanitize=memory' for target 'x86_64-apple-darwin10'

// RUN: %clang -target x86_64-apple-darwin10 -fsanitize=memory -fno-sanitize=memory %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-MSAN-NOMSAN-DARWIN
// CHECK-MSAN-NOMSAN-DARWIN-NOT: unsupported option

// RUN: %clang -target x86_64-apple-darwin10 -fsanitize=memory -fsanitize=thread,memory %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-MSAN-TSAN-MSAN-DARWIN
// CHECK-MSAN-TSAN-MSAN-DARWIN: unsupported option '-fsanitize=memory' for target 'x86_64-apple-darwin10'
// CHECK-MSAN-TSAN-MSAN-DARWIN-NOT: unsupported option

// RUN: %clang -target x86_64-apple-darwin10 -fsanitize=thread,memory -fsanitize=memory %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-MSAN-MSAN-DARWIN
// CHECK-TSAN-MSAN-MSAN-DARWIN: unsupported option '-fsanitize=memory' for target 'x86_64-apple-darwin10'
// CHECK-TSAN-MSAN-MSAN-DARWIN-NOT: unsupported option

// RUN: %clang -target x86_64-apple-darwin -fsanitize=thread %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-X86-64-DARWIN
// CHECK-TSAN-X86-64-DARWIN-NOT: unsupported option

// RUN: %clang -target x86_64-apple-iossimulator -fsanitize=thread %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-X86-64-IOSSIMULATOR
// CHECK-TSAN-X86-64-IOSSIMULATOR-NOT: unsupported option

// RUN: %clang -target x86_64-apple-tvossimulator -fsanitize=thread %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-X86-64-TVOSSIMULATOR
// CHECK-TSAN-X86-64-TVOSSIMULATOR-NOT: unsupported option

// RUN: %clang -target i386-apple-darwin -fsanitize=thread %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-I386-DARWIN
// CHECK-TSAN-I386-DARWIN: unsupported option '-fsanitize=thread' for target 'i386-apple-darwin'

// RUN: %clang -target arm-apple-ios -fsanitize=thread %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-ARM-IOS
// CHECK-TSAN-ARM-IOS: unsupported option '-fsanitize=thread' for target 'arm-apple-ios'

// RUN: %clang -target i386-apple-iossimulator -fsanitize=thread %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-I386-IOSSIMULATOR
// CHECK-TSAN-I386-IOSSIMULATOR: unsupported option '-fsanitize=thread' for target 'i386-apple-iossimulator-simulator'

// RUN: %clang -target i386-apple-tvossimulator -fsanitize=thread %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-I386-TVOSSIMULATOR
// CHECK-TSAN-I386-TVOSSIMULATOR: unsupported option '-fsanitize=thread' for target 'i386-apple-tvossimulator-simulator'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=thread -fsanitize-thread-memory-access %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-MEMORY-ACCESS
// CHECK-TSAN-MEMORY-ACCESS-NOT: -cc1{{.*}}tsan-instrument-memory-accesses=0
// CHECK-TSAN-MEMORY-ACCESS-NOT: -cc1{{.*}}tsan-instrument-memintrinsics=0
// RUN: %clang -target x86_64-linux-gnu -fsanitize=thread -fno-sanitize-thread-memory-access %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-MEMORY-ACCESS-OFF
// CHECK-TSAN-MEMORY-ACCESS-OFF: -cc1{{.*}}tsan-instrument-memory-accesses=0{{.*}}tsan-instrument-memintrinsics=0
// RUN: %clang -target x86_64-linux-gnu -fsanitize=thread -fno-sanitize-thread-memory-access -fsanitize-thread-memory-access %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-MEMORY-ACCESS-BOTH
// CHECK-TSAN-MEMORY-ACCESS-BOTH-NOT: -cc1{{.*}}tsan-instrument-memory-accesses=0
// CHECK-TSAN-MEMORY-ACCESS-BOTH-NOT: -cc1{{.*}}tsan-instrument-memintrinsics=0
// RUN: %clang -target x86_64-linux-gnu -fsanitize=thread -fsanitize-thread-memory-access -fno-sanitize-thread-memory-access %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-MEMORY-ACCESS-BOTH-OFF
// CHECK-TSAN-MEMORY-ACCESS-BOTH-OFF: -cc1{{.*}}tsan-instrument-memory-accesses=0{{.*}}tsan-instrument-memintrinsics=0

// RUN: %clang -target x86_64-linux-gnu -fsanitize=thread -fsanitize-thread-func-entry-exit %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-FUNC-ENTRY-EXIT
// CHECK-TSAN-FUNC-ENTRY-EXIT-NOT: -cc1{{.*}}tsan-instrument-func-entry-exit=0
// RUN: %clang -target x86_64-linux-gnu -fsanitize=thread -fno-sanitize-thread-func-entry-exit %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-FUNC-ENTRY-EXIT-OFF
// CHECK-TSAN-FUNC-ENTRY-EXIT-OFF: -cc1{{.*}}tsan-instrument-func-entry-exit=0
// RUN: %clang -target x86_64-linux-gnu -fsanitize=thread -fno-sanitize-thread-func-entry-exit -fsanitize-thread-func-entry-exit %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-FUNC-ENTRY-EXIT-BOTH
// CHECK-TSAN-FUNC-ENTRY-EXIT-BOTH-NOT: -cc1{{.*}}tsan-instrument-func-entry-exit=0
// RUN: %clang -target x86_64-linux-gnu -fsanitize=thread -fsanitize-thread-func-entry-exit -fno-sanitize-thread-func-entry-exit %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-FUNC-ENTRY-EXIT-BOTH-OFF
// CHECK-TSAN-FUNC-ENTRY-EXIT-BOTH-OFF: -cc1{{.*}}tsan-instrument-func-entry-exit=0

// RUN: %clang -target x86_64-linux-gnu -fsanitize=thread -fsanitize-thread-atomics %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-ATOMICS
// CHECK-TSAN-ATOMICS-NOT: -cc1{{.*}}tsan-instrument-atomics=0
// RUN: %clang -target x86_64-linux-gnu -fsanitize=thread -fno-sanitize-thread-atomics %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-ATOMICS-OFF
// CHECK-TSAN-ATOMICS-OFF: -cc1{{.*}}tsan-instrument-atomics=0
// RUN: %clang -target x86_64-linux-gnu -fsanitize=thread -fno-sanitize-thread-atomics -fsanitize-thread-atomics %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-ATOMICS-BOTH
// CHECK-TSAN-ATOMICS-BOTH-NOT: -cc1{{.*}}tsan-instrument-atomics=0
// RUN: %clang -target x86_64-linux-gnu -fsanitize=thread -fsanitize-thread-atomics -fno-sanitize-thread-atomics %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-ATOMICS-BOTH-OFF
// CHECK-TSAN-ATOMICS-BOTH-OFF: -cc1{{.*}}tsan-instrument-atomics=0

// RUN: %clang -target x86_64-apple-darwin10 -fsanitize=function %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FSAN-DARWIN
// RUN: %clang -target i386-apple-darwin10 -fsanitize=function %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FSAN-DARWIN
// CHECK-FSAN-DARWIN: -cc1{{.*}}"-fsanitize=function" "-fsanitize-recover=function"

// RUN: %clang -target x86_64-apple-darwin10 -mmacosx-version-min=10.8 -fsanitize=vptr %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-VPTR-DARWIN-OLD
// CHECK-VPTR-DARWIN-OLD: unsupported option '-fsanitize=vptr' for target 'x86_64-apple-darwin10'

// RUN: %clang -target x86_64-apple-darwin10 -mmacosx-version-min=10.9 -fsanitize=alignment,vptr %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-VPTR-DARWIN-NEW
// CHECK-VPTR-DARWIN-NEW: -fsanitize=alignment,vptr

// RUN: %clang -target armv7-apple-ios7 -miphoneos-version-min=7.0 -fsanitize=address %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-IOS
// CHECK-ASAN-IOS: -fsanitize=address

// RUN %clang -target i386-pc-openbsd -fsanitize=undefined %s -### 2>&1 | FileCheck --check-prefix=CHECK-UBSAN-OPENBSD
// CHECK-UBSAN-OPENBSD: -fsanitize=undefined

// RUN: %clang -target i386-pc-openbsd -fsanitize=address %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-OPENBSD
// CHECK-ASAN-OPENBSD: unsupported option '-fsanitize=address' for target 'i386-pc-openbsd'

// RUN: %clang -target i386-pc-openbsd -fsanitize=leak %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-LSAN-OPENBSD
// CHECK-LSAN-OPENBSD: unsupported option '-fsanitize=leak' for target 'i386-pc-openbsd'

// RUN: %clang -target i386-pc-openbsd -fsanitize=thread %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-OPENBSD
// CHECK-TSAN-OPENBSD: unsupported option '-fsanitize=thread' for target 'i386-pc-openbsd'

// RUN: %clang -target i386-pc-openbsd -fsanitize=memory %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-MSAN-OPENBSD
// CHECK-MSAN-OPENBSD: unsupported option '-fsanitize=memory' for target 'i386-pc-openbsd'

// RUN: %clang -target i386-pc-openbsd -fsanitize=efficiency-cache-frag %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ESAN-OPENBSD
// RUN: %clang -target i386-pc-openbsd -fsanitize=efficiency-working-set %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ESAN-OPENBSD
// CHECK-ESAN-OPENBSD: error: unsupported option '-fsanitize=efficiency-{{.*}}' for target 'i386-pc-openbsd'

// RUN: %clang -target x86_64-apple-darwin -fsanitize=leak %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-LSAN-X86-64-DARWIN
// CHECK-LSAN-X86-64-DARWIN-NOT: unsupported option

// RUN: %clang -target x86_64-apple-iossimulator -fsanitize=leak %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-LSAN-X86-64-IOSSIMULATOR
// CHECK-LSAN-X86-64-IOSSIMULATOR-NOT: unsupported option

// RUN: %clang -target x86_64-apple-tvossimulator -fsanitize=leak %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-LSAN-X86-64-TVOSSIMULATOR
// CHECK-LSAN-X86-64-TVOSSIMULATOR-NOT: unsupported option

// RUN: %clang -target i386-apple-darwin -fsanitize=leak %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-LSAN-I386-DARWIN
// CHECK-LSAN-I386-DARWIN-NOT: unsupported option

// RUN: %clang -target arm-apple-ios -fsanitize=leak %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-LSAN-ARM-IOS
// CHECK-LSAN-ARM-IOS-NOT: unsupported option

// RUN: %clang -target i386-apple-iossimulator -fsanitize=leak %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-LSAN-I386-IOSSIMULATOR
// CHECK-LSAN-I386-IOSSIMULATOR-NOT: unsupported option

// RUN: %clang -target i386-apple-tvossimulator -fsanitize=leak %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-LSAN-I386-TVOSSIMULATOR
// CHECK-LSAN-I386-TVOSSIMULATOR-NOT: unsupported option

// RUN: %clang -target i686-linux-gnu -fsanitize=efficiency-cache-frag %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ESAN-X86
// RUN: %clang -target i686-linux-gnu -fsanitize=efficiency-working-set %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ESAN-X86
// CHECK-ESAN-X86: error: unsupported option '-fsanitize=efficiency-{{.*}}' for target 'i686--linux-gnu'

// RUN: %clang -target x86_64-apple-darwin10 -fsanitize=efficiency-cache-frag %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ESAN-DARWIN
// RUN: %clang -target x86_64-apple-darwin10 -fsanitize=efficiency-working-set %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ESAN-DARWIN
// CHECK-ESAN-DARWIN: unsupported option '-fsanitize=efficiency-{{.*}}' for target 'x86_64-apple-darwin10'

// RUN: %clang -target i386-apple-darwin -fsanitize=efficiency-cache-frag %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ESAN-I386-DARWIN
// RUN: %clang -target i386-apple-darwin -fsanitize=efficiency-working-set %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ESAN-I386-DARWIN
// CHECK-ESAN-I386-DARWIN: unsupported option '-fsanitize=efficiency-{{.*}}' for target 'i386-apple-darwin'

// RUN: %clang -target arm-apple-ios -fsanitize=efficiency-cache-frag %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ESAN-ARM-IOS
// RUN: %clang -target arm-apple-ios -fsanitize=efficiency-working-set %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ESAN-ARM-IOS
// CHECK-ESAN-ARM-IOS: unsupported option '-fsanitize=efficiency-{{.*}}' for target 'arm-apple-ios'

// RUN: %clang -target i386-apple-iossimulator -fsanitize=efficiency-cache-frag %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ESAN-I386-IOSSIMULATOR
// RUN: %clang -target i386-apple-iossimulator -fsanitize=efficiency-working-set %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ESAN-I386-IOSSIMULATOR
// CHECK-ESAN-I386-IOSSIMULATOR: unsupported option '-fsanitize=efficiency-{{.*}}' for target 'i386-apple-iossimulator-simulator'

// RUN: %clang -target i386-apple-tvossimulator -fsanitize=efficiency-cache-frag %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ESAN-I386-TVOSSIMULATOR
// RUN: %clang -target i386-apple-tvossimulator -fsanitize=efficiency-working-set %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ESAN-I386-TVOSSIMULATOR
// CHECK-ESAN-I386-TVOSSIMULATOR: unsupported option '-fsanitize=efficiency-{{.*}}' for target 'i386-apple-tvossimulator-simulator'



// RUN: %clang -target x86_64-linux-gnu -fvisibility=hidden -fsanitize=cfi -flto -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI
// RUN: %clang -target x86_64-apple-darwin10 -fvisibility=hidden -fsanitize=cfi -flto -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI
// RUN: %clang -target x86_64-pc-win32 -fvisibility=hidden -fsanitize=cfi -flto -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-NOMFCALL
// RUN: %clang -target x86_64-linux-gnu -fvisibility=hidden -fsanitize=cfi -fsanitize-cfi-cross-dso -flto -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-NOMFCALL
// RUN: %clang -target x86_64-linux-gnu -fvisibility=hidden -fsanitize=cfi-derived-cast -flto -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-DCAST
// RUN: %clang -target x86_64-linux-gnu -fvisibility=hidden -fsanitize=cfi-unrelated-cast -flto -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-UCAST
// RUN: %clang -target x86_64-linux-gnu -flto -fvisibility=hidden -fsanitize=cfi-nvcall -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-NVCALL
// RUN: %clang -target x86_64-linux-gnu -flto -fvisibility=hidden -fsanitize=cfi-vcall -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-VCALL
// RUN: %clang -target arm-linux-gnu -fvisibility=hidden -fsanitize=cfi -flto -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI
// RUN: %clang -target aarch64-linux-gnu -fvisibility=hidden -fsanitize=cfi -flto -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI
// RUN: %clang -target arm-linux-android -fvisibility=hidden -fsanitize=cfi -flto -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI
// RUN: %clang -target aarch64-linux-android -fvisibility=hidden -fsanitize=cfi -flto -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI
// CHECK-CFI: -emit-llvm-bc{{.*}}-fsanitize=cfi-derived-cast,cfi-icall,cfi-mfcall,cfi-unrelated-cast,cfi-nvcall,cfi-vcall
// CHECK-CFI-NOMFCALL: -emit-llvm-bc{{.*}}-fsanitize=cfi-derived-cast,cfi-icall,cfi-unrelated-cast,cfi-nvcall,cfi-vcall
// CHECK-CFI-DCAST: -emit-llvm-bc{{.*}}-fsanitize=cfi-derived-cast
// CHECK-CFI-UCAST: -emit-llvm-bc{{.*}}-fsanitize=cfi-unrelated-cast
// CHECK-CFI-NVCALL: -emit-llvm-bc{{.*}}-fsanitize=cfi-nvcall
// CHECK-CFI-VCALL: -emit-llvm-bc{{.*}}-fsanitize=cfi-vcall

// RUN: %clang -target x86_64-linux-gnu -fvisibility=hidden -flto -fsanitize=cfi-derived-cast -fno-lto -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-NOLTO
// CHECK-CFI-NOLTO: '-fsanitize=cfi-derived-cast' only allowed with '-flto'

// RUN: %clang -target x86_64-linux-gnu -flto -fsanitize=cfi-derived-cast -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-NOVIS
// CHECK-CFI-NOVIS: '-fsanitize=cfi-derived-cast' only allowed with '-fvisibility='

// RUN: %clang -target x86_64-pc-win32 -flto -fsanitize=cfi-derived-cast -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-NOVIS-NOERROR
// RUN: echo > %t.o
// RUN: %clang -target x86_64-linux-gnu -flto -fsanitize=cfi-derived-cast %t.o -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-NOVIS-NOERROR
// CHECK-CFI-NOVIS-NOERROR-NOT: only allowed with

// RUN: %clang -target mips-unknown-linux -fsanitize=cfi-icall %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-ICALL-MIPS
// CHECK-CFI-ICALL-MIPS: unsupported option '-fsanitize=cfi-icall' for target 'mips-unknown-linux'

// RUN: %clang -target x86_64-linux-gnu -fsanitize-trap=address -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-TRAP
// CHECK-ASAN-TRAP: error: unsupported argument 'address' to option '-fsanitize-trap'

// RUN: %clang -target x86_64-linux-gnu -fsanitize-trap=hwaddress -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-HWASAN-TRAP
// CHECK-HWASAN-TRAP: error: unsupported argument 'hwaddress' to option '-fsanitize-trap'

// RUN: %clang -target x86_64-apple-darwin10 -mmacosx-version-min=10.7 -flto -fsanitize=cfi-vcall -fno-sanitize-trap=cfi -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-NOTRAP-OLD-MACOS
// CHECK-CFI-NOTRAP-OLD-MACOS: error: unsupported option '-fno-sanitize-trap=cfi-vcall' for target 'x86_64-apple-darwin10'

// RUN: %clang -target x86_64-pc-win32 -flto -fsanitize=cfi-vcall -fno-sanitize-trap=cfi -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-NOTRAP-WIN
// CHECK-CFI-NOTRAP-WIN: -emit-llvm-bc
// CHECK-CFI-NOTRAP-WIN-NOT: -fsanitize-trap=cfi

// RUN: %clang -target x86_64-linux-gnu -fsanitize=cfi -fsanitize-cfi-cross-dso -flto -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-CROSS-DSO
// RUN: %clang -target x86_64-linux-gnu -fsanitize=cfi -flto -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-NO-CROSS-DSO
// RUN: %clang -target x86_64-linux-gnu -fsanitize=cfi -fsanitize-cfi-cross-dso -fno-sanitize-cfi-cross-dso -flto -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-NO-CROSS-DSO
// RUN: %clang -target x86_64-linux-gnu -fsanitize=cfi -fno-sanitize-cfi-cross-dso -fsanitize-cfi-cross-dso -flto -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-CROSS-DSO
// CHECK-CFI-CROSS-DSO: -emit-llvm-bc
// CHECK-CFI-CROSS-DSO: -fsanitize-cfi-cross-dso
// CHECK-CFI-NO-CROSS-DSO: -emit-llvm-bc
// CHECK-CFI-NO-CROSS-DSO-NOT: -fsanitize-cfi-cross-dso

// RUN: %clang -target x86_64-linux-gnu -fvisibility=hidden -fsanitize=cfi-mfcall -fsanitize-cfi-cross-dso -flto -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-MFCALL-CROSS-DSO
// CHECK-CFI-MFCALL-CROSS-DSO: '-fsanitize=cfi-mfcall' not allowed with '-fsanitize-cfi-cross-dso'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=cfi-icall -fsanitize-cfi-icall-generalize-pointers -fvisibility=hidden -flto -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-GENERALIZE-POINTERS
// RUN: %clang -target x86_64-linux-gnu -fsanitize=cfi-icall -fvisibility=hidden -flto -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-CFI-GENERALIZE-POINTERS
// CHECK-CFI-GENERALIZE-POINTERS: -fsanitize-cfi-icall-generalize-pointers
// CHECK-NO-CFI-GENERALIZE-POINTERS-NOT: -fsanitize-cfi-icall-generalize-pointers

// RUN: %clang -target x86_64-linux-gnu -fsanitize=cfi-icall -fsanitize-cfi-icall-generalize-pointers -fsanitize-cfi-cross-dso -fvisibility=hidden -flto -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-GENERALIZE-AND-CROSS-DSO
// CHECK-CFI-GENERALIZE-AND-CROSS-DSO: error: invalid argument '-fsanitize-cfi-cross-dso' not allowed with '-fsanitize-cfi-icall-generalize-pointers'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=cfi -fsanitize-stats -flto -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-STATS
// CHECK-CFI-STATS: -fsanitize-stats

// RUN: %clang_cl -fsanitize=address -c -MDd -### -- %s 2>&1 | FileCheck %s -check-prefix=CHECK-ASAN-DEBUGRTL
// RUN: %clang_cl -fsanitize=address -c -MTd -### -- %s 2>&1 | FileCheck %s -check-prefix=CHECK-ASAN-DEBUGRTL
// RUN: %clang_cl -fsanitize=address -c -LDd -### -- %s 2>&1 | FileCheck %s -check-prefix=CHECK-ASAN-DEBUGRTL
// RUN: %clang_cl -fsanitize=address -c -MD -MDd -### -- %s 2>&1 | FileCheck %s -check-prefix=CHECK-ASAN-DEBUGRTL
// RUN: %clang_cl -fsanitize=address -c -MT -MTd -### -- %s 2>&1 | FileCheck %s -check-prefix=CHECK-ASAN-DEBUGRTL
// RUN: %clang_cl -fsanitize=address -c -LD -LDd -### -- %s 2>&1 | FileCheck %s -check-prefix=CHECK-ASAN-DEBUGRTL
// CHECK-ASAN-DEBUGRTL: error: invalid argument
// CHECK-ASAN-DEBUGRTL: not allowed with '-fsanitize=address'
// CHECK-ASAN-DEBUGRTL: note: AddressSanitizer doesn't support linking with debug runtime libraries yet

// RUN: %clang_cl -fsanitize=address -c -MT -### -- %s 2>&1 | FileCheck %s -check-prefix=CHECK-ASAN-RELEASERTL
// RUN: %clang_cl -fsanitize=address -c -MD -### -- %s 2>&1 | FileCheck %s -check-prefix=CHECK-ASAN-RELEASERTL
// RUN: %clang_cl -fsanitize=address -c -LD -### -- %s 2>&1 | FileCheck %s -check-prefix=CHECK-ASAN-RELEASERTL
// RUN: %clang_cl -fsanitize=address -c -MTd -MT -### -- %s 2>&1 | FileCheck %s -check-prefix=CHECK-ASAN-RELEASERTL
// RUN: %clang_cl -fsanitize=address -c -MDd -MD -### -- %s 2>&1 | FileCheck %s -check-prefix=CHECK-ASAN-RELEASERTL
// RUN: %clang_cl -fsanitize=address -c -LDd -LD -### -- %s 2>&1 | FileCheck %s -check-prefix=CHECK-ASAN-RELEASERTL
// CHECK-ASAN-RELEASERTL-NOT: error: invalid argument

// RUN: %clang -fno-sanitize=safe-stack -### %s 2>&1 | FileCheck %s -check-prefix=NOSP
// NOSP-NOT: "-fsanitize=safe-stack"

// RUN: %clang -target x86_64-linux-gnu -fsanitize=safe-stack -### %s 2>&1 | FileCheck %s -check-prefix=NO-SP
// RUN: %clang -target x86_64-linux-gnu -fsanitize=address,safe-stack -### %s 2>&1 | FileCheck %s -check-prefix=SP-ASAN
// RUN: %clang -target x86_64-linux-gnu -fstack-protector -fsanitize=safe-stack -### %s 2>&1 | FileCheck %s -check-prefix=SP
// RUN: %clang -target x86_64-linux-gnu -fsanitize=safe-stack -fstack-protector-all -### %s 2>&1 | FileCheck %s -check-prefix=SP
// RUN: %clang -target arm-linux-androideabi -fsanitize=safe-stack -### %s 2>&1 | FileCheck %s -check-prefix=NO-SP
// RUN: %clang -target aarch64-linux-android -fsanitize=safe-stack -### %s 2>&1 | FileCheck %s -check-prefix=NO-SP
// RUN: %clang -target i386-contiki-unknown -fsanitize=safe-stack -### %s 2>&1 | FileCheck %s -check-prefix=NO-SP
// NO-SP-NOT: stack-protector
// NO-SP: "-fsanitize=safe-stack"
// SP-ASAN: error: invalid argument '-fsanitize=safe-stack' not allowed with '-fsanitize=address'
// SP: "-fsanitize=safe-stack"
// SP: -stack-protector
// NO-SP-NOT: stack-protector

// RUN: %clang -target powerpc64-unknown-linux-gnu -fsanitize=memory %s -### 2>&1 | FileCheck %s -check-prefix=CHECK-SANM
// RUN: %clang -target powerpc64le-unknown-linux-gnu -fsanitize=memory %s -### 2>&1 | FileCheck %s -check-prefix=CHECK-SANM
// CHECK-SANM: "-fsanitize=memory"

// RUN: %clang -target aarch64-unknown-cloudabi -fsanitize=safe-stack %s -### 2>&1 | FileCheck %s -check-prefix=SAFESTACK-CLOUDABI
// RUN: %clang -target x86_64-unknown-cloudabi -fsanitize=safe-stack %s -### 2>&1 | FileCheck %s -check-prefix=SAFESTACK-CLOUDABI
// SAFESTACK-CLOUDABI: "-fsanitize=safe-stack"

// RUN: %clang -target i386--netbsd -fsanitize=address %s -### 2>&1 | FileCheck %s -check-prefix=ADDRESS-NETBSD
// RUN: %clang -target x86_64--netbsd -fsanitize=address %s -### 2>&1 | FileCheck %s -check-prefix=ADDRESS-NETBSD
// ADDRESS-NETBSD: "-fsanitize=address"

// RUN: %clang -target i386--netbsd -fsanitize=vptr %s -### 2>&1 | FileCheck %s -check-prefix=VPTR-NETBSD
// RUN: %clang -target x86_64--netbsd -fsanitize=vptr %s -### 2>&1 | FileCheck %s -check-prefix=VPTR-NETBSD
// VPTR-NETBSD: "-fsanitize=vptr"

// RUN: %clang -target i386--netbsd -fsanitize=safe-stack %s -### 2>&1 | FileCheck %s -check-prefix=SAFESTACK-NETBSD
// RUN: %clang -target x86_64--netbsd -fsanitize=safe-stack %s -### 2>&1 | FileCheck %s -check-prefix=SAFESTACK-NETBSD
// SAFESTACK-NETBSD: "-fsanitize=safe-stack"

// RUN: %clang -target i386--netbsd -fsanitize=function %s -### 2>&1 | FileCheck %s -check-prefix=FUNCTION-NETBSD
// RUN: %clang -target x86_64--netbsd -fsanitize=function %s -### 2>&1 | FileCheck %s -check-prefix=FUNCTION-NETBSD
// FUNCTION-NETBSD: "-fsanitize=function"

// RUN: %clang -target i386--netbsd -fsanitize=leak %s -### 2>&1 | FileCheck %s -check-prefix=LEAK-NETBSD
// RUN: %clang -target x86_64--netbsd -fsanitize=leak %s -### 2>&1 | FileCheck %s -check-prefix=LEAK-NETBSD
// LEAK-NETBSD: "-fsanitize=leak"

// RUN: %clang -target x86_64--netbsd -fsanitize=thread %s -### 2>&1 | FileCheck %s -check-prefix=THREAD-NETBSD
// THREAD-NETBSD: "-fsanitize=thread"

// RUN: %clang -target x86_64-scei-ps4 -fsanitize=function -fsanitize=undefined %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FSAN-UBSAN-PS4
// CHECK-FSAN-UBSAN-PS4: unsupported option '-fsanitize=function' for target 'x86_64-scei-ps4'
// RUN: %clang -target x86_64-scei-ps4 -fsanitize=function %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FSAN-PS4
// CHECK-FSAN-PS4: unsupported option '-fsanitize=function' for target 'x86_64-scei-ps4'
// RUN: %clang -target x86_64-scei-ps4 -fsanitize=dataflow %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-DFSAN-PS4
// CHECK-DFSAN-PS4: unsupported option '-fsanitize=dataflow' for target 'x86_64-scei-ps4'
// RUN: %clang -target x86_64-scei-ps4 -fsanitize=leak %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-LSAN-PS4
// CHECK-LSAN-PS4: unsupported option '-fsanitize=leak' for target 'x86_64-scei-ps4'
// RUN: %clang -target x86_64-scei-ps4 -fsanitize=memory %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-MSAN-PS4
// CHECK-MSAN-PS4: unsupported option '-fsanitize=memory' for target 'x86_64-scei-ps4'
// RUN: %clang -target x86_64-scei-ps4 -fsanitize=thread %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-PS4
// CHECK-TSAN-PS4: unsupported option '-fsanitize=thread' for target 'x86_64-scei-ps4'
// RUN: %clang -target x86_64-scei-ps4 -fsanitize=efficiency-cache-frag %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ESAN-PS4
// RUN: %clang -target x86_64-scei-ps4 -fsanitize=efficiency-working-set %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ESAN-PS4
// CHECK-ESAN-PS4: unsupported option '-fsanitize=efficiency-{{.*}}' for target 'x86_64-scei-ps4'
// RUN: %clang -target x86_64-scei-ps4 -fsanitize=address %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-PS4
// Make sure there are no *.{o,bc} or -l passed before the ASan library.
// CHECK-ASAN-PS4-NOT: {{(\.(o|bc)"? |-l).*-lSceDbgAddressSanitizer_stub_weak}}
// CHECK-ASAN-PS4: --dependent-lib=libSceDbgAddressSanitizer_stub_weak.a
// CHECK-ASAN-PS4-NOT: {{(\.(o|bc)"? |-l).*-lSceDbgAddressSanitizer_stub_weak}}
// CHECK-ASAN-PS4: -lSceDbgAddressSanitizer_stub_weak

// RUN: %clang -target x86_64-linux-gnu -fsanitize=address -fsanitize-minimal-runtime %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-MINIMAL
// CHECK-ASAN-MINIMAL: error: invalid argument '-fsanitize-minimal-runtime' not allowed with '-fsanitize=address'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=thread -fsanitize-minimal-runtime %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-MINIMAL
// CHECK-TSAN-MINIMAL: error: invalid argument '-fsanitize-minimal-runtime' not allowed with '-fsanitize=thread'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=undefined -fsanitize-minimal-runtime %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UBSAN-MINIMAL
// CHECK-UBSAN-MINIMAL: "-fsanitize={{((signed-integer-overflow|integer-divide-by-zero|float-divide-by-zero|shift-base|shift-exponent|unreachable|return|vla-bound|alignment|null|pointer-overflow|float-cast-overflow|array-bounds|enum|bool|builtin|returns-nonnull-attribute|nonnull-attribute|function),?){19}"}}
// CHECK-UBSAN-MINIMAL: "-fsanitize-minimal-runtime"

// RUN: %clang -target x86_64-linux-gnu -fsanitize=undefined -fsanitize=vptr -fsanitize-minimal-runtime %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UBSAN-VPTR-MINIMAL
// CHECK-UBSAN-VPTR-MINIMAL: error: invalid argument '-fsanitize=vptr' not allowed with '-fsanitize-minimal-runtime'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=address -fsanitize-minimal-runtime -fsanitize=undefined %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-UBSAN-MINIMAL
// CHECK-ASAN-UBSAN-MINIMAL: error: invalid argument '-fsanitize-minimal-runtime' not allowed with '-fsanitize=address'
// RUN: %clang -target x86_64-linux-gnu -fsanitize=hwaddress -fsanitize-minimal-runtime %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-HWASAN-MINIMAL
// CHECK-HWASAN-MINIMAL: error: invalid argument '-fsanitize-minimal-runtime' not allowed with '-fsanitize=hwaddress'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=cfi -flto -fvisibility=hidden -fsanitize-minimal-runtime %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-MINIMAL
// CHECK-CFI-MINIMAL: "-fsanitize=cfi-derived-cast,cfi-icall,cfi-mfcall,cfi-unrelated-cast,cfi-nvcall,cfi-vcall"
// CHECK-CFI-MINIMAL: "-fsanitize-trap=cfi-derived-cast,cfi-icall,cfi-mfcall,cfi-unrelated-cast,cfi-nvcall,cfi-vcall"
// CHECK-CFI-MINIMAL: "-fsanitize-minimal-runtime"

// RUN: %clang -target x86_64-linux-gnu -fsanitize=cfi -fno-sanitize-trap=cfi-icall -flto -fvisibility=hidden -fsanitize-minimal-runtime %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-NOTRAP-MINIMAL
// CHECK-CFI-NOTRAP-MINIMAL: error: invalid argument 'fsanitize-minimal-runtime' only allowed with 'fsanitize-trap=cfi'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=cfi -fno-sanitize-trap=cfi-icall -fno-sanitize=cfi-icall -flto -fvisibility=hidden -fsanitize-minimal-runtime %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-NOICALL-MINIMAL
// CHECK-CFI-NOICALL-MINIMAL: "-fsanitize=cfi-derived-cast,cfi-mfcall,cfi-unrelated-cast,cfi-nvcall,cfi-vcall"
// CHECK-CFI-NOICALL-MINIMAL: "-fsanitize-trap=cfi-derived-cast,cfi-mfcall,cfi-unrelated-cast,cfi-nvcall,cfi-vcall"
// CHECK-CFI-NOICALL-MINIMAL: "-fsanitize-minimal-runtime"

// RUN: %clang -target aarch64-linux-gnu -fsanitize=scudo %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SCUDO
// RUN: %clang -target arm-linux-androideabi -fsanitize=scudo %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SCUDO
// RUN: %clang -target x86_64-linux-gnu -fsanitize=scudo %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SCUDO
// RUN: %clang -target i386-linux-gnu -fsanitize=scudo %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SCUDO
// RUN: %clang -target mips64-unknown-linux-gnu -fsanitize=scudo %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SCUDO
// RUN: %clang -target mips64el-unknown-linux-gnu -fsanitize=scudo %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SCUDO
// RUN: %clang -target mips-unknown-linux-gnu -fsanitize=scudo %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SCUDO
// RUN: %clang -target mipsel-unknown-linux-gnu -fsanitize=scudo %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SCUDO
// RUN: %clang -target powerpc64-unknown-linux -fsanitize=scudo %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SCUDO
// RUN: %clang -target powerpc64le-unknown-linux -fsanitize=scudo %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SCUDO
// CHECK-SCUDO: "-fsanitize=scudo"

// RUN: %clang -target x86_64-linux-gnu -fsanitize=scudo %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SCUDO-PIE
// RUN: %clang -target arm-linux-androideabi -fsanitize=scudo %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SCUDO-PIE
// CHECK-SCUDO-PIE: "-mrelocation-model" "pic" "-pic-level" "2" "-pic-is-pie"
// CHECK-SCUDO-PIE: "-pie"

// RUN: %clang -target x86_64-linux-gnu -fsanitize=scudo,undefined %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SCUDO-UBSAN
// CHECK-SCUDO-UBSAN: "-fsanitize={{.*}}scudo"

// RUN: %clang -target x86_64-linux-gnu -fsanitize=scudo -fsanitize-minimal-runtime %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SCUDO-MINIMAL
// CHECK-SCUDO-MINIMAL: "-fsanitize=scudo"
// CHECK-SCUDO-MINIMAL: "-fsanitize-minimal-runtime"

// RUN: %clang -target x86_64-linux-gnu -fsanitize=undefined,scudo -fsanitize-minimal-runtime %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SCUDO-UBSAN-MINIMAL
// CHECK-SCUDO-UBSAN-MINIMAL: "-fsanitize={{.*}}scudo"
// CHECK-SCUDO-UBSAN-MINIMAL: "-fsanitize-minimal-runtime"

// RUN: %clang -target powerpc-unknown-linux -fsanitize=scudo %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-SCUDO
// CHECK-NO-SCUDO: unsupported option

// RUN: %clang -target x86_64-linux-gnu -fsanitize=scudo,address  %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SCUDO-ASAN
// CHECK-SCUDO-ASAN: error: invalid argument '-fsanitize=scudo' not allowed with '-fsanitize=address'
// RUN: %clang -target x86_64-linux-gnu -fsanitize=scudo,leak  %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SCUDO-LSAN
// CHECK-SCUDO-LSAN: error: invalid argument '-fsanitize=scudo' not allowed with '-fsanitize=leak'
// RUN: %clang -target x86_64-linux-gnu -fsanitize=scudo,memory  %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SCUDO-MSAN
// CHECK-SCUDO-MSAN: error: invalid argument '-fsanitize=scudo' not allowed with '-fsanitize=memory'
// RUN: %clang -target x86_64-linux-gnu -fsanitize=scudo,thread  %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SCUDO-TSAN
// CHECK-SCUDO-TSAN: error: invalid argument '-fsanitize=scudo' not allowed with '-fsanitize=thread'
// RUN: %clang -target x86_64-linux-gnu -fsanitize=scudo,hwaddress  %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SCUDO-HWASAN
// CHECK-SCUDO-HWASAN: error: invalid argument '-fsanitize=scudo' not allowed with '-fsanitize=hwaddress'
