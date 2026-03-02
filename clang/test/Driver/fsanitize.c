// This file contains feature-unspecific common -fsanitize= driver tests.
// Where possible avoid adding new tests to this file, and instead group tests
// in feature-specific fsanitize-*.c test files.

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=all %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FSANITIZE-ALL
// CHECK-FSANITIZE-ALL: error: unsupported argument 'all' to option '-fsanitize='

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=address,undefined -fno-sanitize=all -fsanitize=thread %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FNO-SANITIZE-ALL
// CHECK-FNO-SANITIZE-ALL: "-fsanitize=thread"

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=address,thread -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANA-SANT
// CHECK-SANA-SANT: '-fsanitize=address' not allowed with '-fsanitize=thread'

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=address,memory -pie -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANA-SANM
// CHECK-SANA-SANM: '-fsanitize=address' not allowed with '-fsanitize=memory'

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=thread,memory -pie -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANT-SANM
// CHECK-SANT-SANM: '-fsanitize=thread' not allowed with '-fsanitize=memory'

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=memory,thread -pie -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANM-SANT
// CHECK-SANM-SANT: '-fsanitize=thread' not allowed with '-fsanitize=memory'

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=leak,thread -pie -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANL-SANT
// CHECK-SANL-SANT: '-fsanitize=leak' not allowed with '-fsanitize=thread'

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=leak,memory -pie -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANL-SANM
// CHECK-SANL-SANM: '-fsanitize=leak' not allowed with '-fsanitize=memory'

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=kernel-memory,address -pie -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-KMSAN-ASAN
// CHECK-KMSAN-ASAN: '-fsanitize=kernel-memory' not allowed with '-fsanitize=address'

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=kernel-memory,hwaddress -pie -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-KMSAN-HWASAN
// CHECK-KMSAN-HWASAN: '-fsanitize=kernel-memory' not allowed with '-fsanitize=hwaddress'

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=kernel-memory,leak -pie -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-KMSAN-LSAN
// CHECK-KMSAN-LSAN: '-fsanitize=kernel-memory' not allowed with '-fsanitize=leak'

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=kernel-memory,thread -pie -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-KMSAN-TSAN
// CHECK-KMSAN-TSAN: '-fsanitize=kernel-memory' not allowed with '-fsanitize=thread'

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=kernel-memory,kernel-address -pie -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-KMSAN-KASAN
// CHECK-KMSAN-KASAN: '-fsanitize=kernel-memory' not allowed with '-fsanitize=kernel-address'

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=kernel-memory,memory -pie -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-KMSAN-MSAN
// CHECK-KMSAN-MSAN: '-fsanitize=kernel-memory' not allowed with '-fsanitize=memory'

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=kernel-memory,safe-stack -pie -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-KMSAN-SAFESTACK
// CHECK-KMSAN-SAFESTACK: '-fsanitize=kernel-memory' not allowed with '-fsanitize=safe-stack'

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=kernel-address,thread -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANKA-SANT
// CHECK-SANKA-SANT: '-fsanitize=kernel-address' not allowed with '-fsanitize=thread'

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=kernel-address,memory -pie -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANKA-SANM
// CHECK-SANKA-SANM: '-fsanitize=kernel-address' not allowed with '-fsanitize=memory'

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=kernel-address,address -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANKA-SANA
// CHECK-SANKA-SANA: '-fsanitize=kernel-address' not allowed with '-fsanitize=address'

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=kernel-address,leak -pie -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANKA-SANL
// CHECK-SANKA-SANL: '-fsanitize=kernel-address' not allowed with '-fsanitize=leak'

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=kernel-hwaddress,thread -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANKHA-SANT
// CHECK-SANKHA-SANT: '-fsanitize=kernel-hwaddress' not allowed with '-fsanitize=thread'

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=kernel-hwaddress,memory -pie -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANKHA-SANM
// CHECK-SANKHA-SANM: '-fsanitize=kernel-hwaddress' not allowed with '-fsanitize=memory'

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=kernel-hwaddress,address -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANKHA-SANA
// CHECK-SANKHA-SANA: '-fsanitize=kernel-hwaddress' not allowed with '-fsanitize=address'

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=kernel-hwaddress,leak -pie -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANKHA-SANL
// CHECK-SANKHA-SANL: '-fsanitize=kernel-hwaddress' not allowed with '-fsanitize=leak'

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=kernel-hwaddress,hwaddress -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANKHA-SANHA
// CHECK-SANKHA-SANHA: '-fsanitize=kernel-hwaddress' not allowed with '-fsanitize=hwaddress'

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=kernel-hwaddress,kernel-address -pie -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANKHA-SANKA
// CHECK-SANKHA-SANKA: '-fsanitize=kernel-hwaddress' not allowed with '-fsanitize=kernel-address'

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=hwaddress,thread -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANHA-SANT
// CHECK-SANHA-SANT: '-fsanitize=hwaddress' not allowed with '-fsanitize=thread'

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=hwaddress,memory -pie -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANHA-SANM
// CHECK-SANHA-SANM: '-fsanitize=hwaddress' not allowed with '-fsanitize=memory'

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=hwaddress,address -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANHA-SANA
// CHECK-SANHA-SANA: '-fsanitize=hwaddress' not allowed with '-fsanitize=address'

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=address -fsanitize-address-use-after-scope %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-USE-AFTER-SCOPE
// RUN: %clang_cl --target=x86_64-windows -fsanitize=address -fsanitize-address-use-after-scope -### -- %s 2>&1 | FileCheck %s --check-prefix=CHECK-USE-AFTER-SCOPE
// CHECK-USE-AFTER-SCOPE: -cc1{{.*}}-fsanitize-address-use-after-scope

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=address -fno-sanitize-address-use-after-scope %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-USE-AFTER-SCOPE-OFF
// RUN: %clang_cl --target=x86_64-windows -fsanitize=address -fno-sanitize-address-use-after-scope -### -- %s 2>&1 | FileCheck %s --check-prefix=CHECK-USE-AFTER-SCOPE-OFF
// CHECK-USE-AFTER-SCOPE-OFF-NOT: -cc1{{.*}}address-use-after-scope

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=address -fno-sanitize-address-use-after-scope -fsanitize-address-use-after-scope %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-USE-AFTER-SCOPE-BOTH
// RUN: %clang_cl --target=x86_64-windows -fsanitize=address -fno-sanitize-address-use-after-scope -fsanitize-address-use-after-scope -### -- %s 2>&1 | FileCheck %s --check-prefix=CHECK-USE-AFTER-SCOPE-BOTH
// CHECK-USE-AFTER-SCOPE-BOTH: -cc1{{.*}}-fsanitize-address-use-after-scope

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=address -fsanitize-address-use-after-scope -fno-sanitize-address-use-after-scope %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-USE-AFTER-SCOPE-BOTH-OFF
// CHECK-USE-AFTER-SCOPE-BOTH-OFF-NOT: -cc1{{.*}}address-use-after-scope

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=address %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-WITHOUT-USE-AFTER-SCOPE
// CHECK-ASAN-WITHOUT-USE-AFTER-SCOPE: -cc1{{.*}}address-use-after-scope

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=address -fsanitize-address-poison-custom-array-cookie %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-POISON-CUSTOM-ARRAY-NEW-COOKIE
// RUN: %clang_cl --target=x86_64-windows -fsanitize=address -fsanitize-address-poison-custom-array-cookie -### -- %s 2>&1 | FileCheck %s --check-prefix=CHECK-POISON-CUSTOM-ARRAY-NEW-COOKIE
// CHECK-POISON-CUSTOM-ARRAY-NEW-COOKIE: -cc1{{.*}}-fsanitize-address-poison-custom-array-cookie

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=address -fno-sanitize-address-poison-custom-array-cookie %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-POISON-CUSTOM-ARRAY-NEW-COOKIE-OFF
// RUN: %clang_cl --target=x86_64-windows -fsanitize=address -fno-sanitize-address-poison-custom-array-cookie -### -- %s 2>&1 | FileCheck %s --check-prefix=CHECK-POISON-CUSTOM-ARRAY-NEW-COOKIE-OFF
// CHECK-POISON-CUSTOM-ARRAY-NEW-COOKIE-OFF-NOT: -cc1{{.*}}address-poison-custom-array-cookie

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=address -fno-sanitize-address-poison-custom-array-cookie -fsanitize-address-poison-custom-array-cookie %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-POISON-CUSTOM-ARRAY-NEW-COOKIE-BOTH
// RUN: %clang_cl --target=x86_64-windows -fsanitize=address -fno-sanitize-address-poison-custom-array-cookie -fsanitize-address-poison-custom-array-cookie -### -- %s 2>&1 | FileCheck %s --check-prefix=CHECK-POISON-CUSTOM-ARRAY-NEW-COOKIE-BOTH
// CHECK-POISON-CUSTOM-ARRAY-NEW-COOKIE-BOTH: -cc1{{.*}}-fsanitize-address-poison-custom-array-cookie

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=address -fsanitize-address-poison-custom-array-cookie -fno-sanitize-address-poison-custom-array-cookie %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-POISON-CUSTOM-ARRAY-NEW-COOKIE-BOTH-OFF
// CHECK-POISON-CUSTOM-ARRAY-NEW-COOKIE-BOTH-OFF-NOT: -cc1{{.*}}address-poison-custom-array-cookie

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=address %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-WITHOUT-POISON-CUSTOM-ARRAY-NEW-COOKIE
// CHECK-ASAN-WITHOUT-POISON-CUSTOM-ARRAY-NEW-COOKIE-NOT: -cc1{{.*}}address-poison-custom-array-cookie

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=address -fsanitize-address-globals-dead-stripping %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-GLOBALS
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=address %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-GLOBALS
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=address -fsanitize-address-globals-dead-stripping -fno-sanitize-address-globals-dead-stripping %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ASAN-GLOBALS
// RUN: %clang_cl --target=x86_64-windows-msvc -fsanitize=address -fsanitize-address-globals-dead-stripping -### -- %s 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-GLOBALS
// RUN: %clang_cl --target=x86_64-windows-msvc -fsanitize=address -### -- %s 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-GLOBALS
// RUN: %clang --target=x86_64-scei-ps4  -fsanitize=address %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-GLOBALS
// RUN: %clang --target=x86_64-sie-ps5  -fsanitize=address %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-GLOBALS
// CHECK-ASAN-GLOBALS: -cc1{{.*}}-fsanitize-address-globals-dead-stripping
// CHECK-NO-ASAN-GLOBALS-NOT: -cc1{{.*}}-fsanitize-address-globals-dead-stripping

// RUN: %clang --target=x86_64-linux-gnu -fsanitize-address-outline-instrumentation %s -### 2>&1 | \
// RUN:     FileCheck %s --check-prefix=CHECK-ASAN-OUTLINE-WARN
// CHECK-ASAN-OUTLINE-WARN: warning: argument unused during compilation: '-fsanitize-address-outline-instrumentation'
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=address -fsanitize-address-outline-instrumentation %s -### 2>&1 | \
// RUN:     FileCheck %s --check-prefix=CHECK-ASAN-OUTLINE-OK
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=address -fno-sanitize-address-outline-instrumentation -fsanitize-address-outline-instrumentation %s -### 2>&1 | \
// RUN:     FileCheck %s --check-prefix=CHECK-ASAN-OUTLINE-OK
// CHECK-ASAN-OUTLINE-OK: "-mllvm" "-asan-instrumentation-with-call-threshold=0"
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=address -fno-sanitize-address-outline-instrumentation %s -### 2>&1 | \
// RUN:     FileCheck %s --check-prefix=CHECK-NO-CHECK-ASAN-CALLBACK
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=address -fsanitize-address-outline-instrumentation -fno-sanitize-address-outline-instrumentation %s -### 2>&1 | \
// RUN:     FileCheck %s --check-prefix=CHECK-NO-CHECK-ASAN-CALLBACK
// CHECK-NO-CHECK-ASAN-CALLBACK-NOT: "-mllvm" "-asan-instrumentation-with-call-threshold=0"

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=address %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-ODR-INDICATOR
// RUN: %clang_cl --target=x86_64-windows -fsanitize=address -### -- %s 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-ODR-INDICATOR-OFF
// CHECK-ASAN-ODR-INDICATOR-NOT: "-fsanitize-address-use-odr-indicator"

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=address -fno-sanitize-address-use-odr-indicator %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-ODR-INDICATOR-OFF
// RUN: %clang_cl --target=x86_64-windows -fsanitize=address -fno-sanitize-address-use-odr-indicator -### -- %s 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-ODR-INDICATOR-OFF
// CHECK-ASAN-ODR-INDICATOR-OFF: "-cc1" {{.*}} "-fno-sanitize-address-use-odr-indicator"

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=address -fno-sanitize-address-use-odr-indicator -fsanitize-address-use-odr-indicator %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-ODR-INDICATOR
// RUN: %clang_cl --target=x86_64-windows -fsanitize=address -fno-sanitize-address-use-odr-indicator -fsanitize-address-use-odr-indicator -### -- %s 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-ODR-INDICATOR

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=address -fsanitize-address-use-odr-indicator -fno-sanitize-address-use-odr-indicator %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-ODR-INDICATOR-OFF

// RUN: %clang --target=x86_64-linux-gnu -fsanitize-memory-track-origins -pie %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ONLY-TRACK-ORIGINS
// CHECK-ONLY-TRACK-ORIGINS: warning: argument unused during compilation: '-fsanitize-memory-track-origins'

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=memory -fno-sanitize=memory -fsanitize-memory-track-origins -pie %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TRACK-ORIGINS-DISABLED-MSAN
// CHECK-TRACK-ORIGINS-DISABLED-MSAN-NOT: warning: argument unused

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=address %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-EXTRA-TRACK-ORIGINS
// CHECK-NO-EXTRA-TRACK-ORIGINS-NOT: "-fsanitize-memory-track-origins"

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=address -fsanitize=alignment -fsanitize=vptr -fno-sanitize=vptr %s -### 2>&1
// OK

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=memory -pie %s -### 2>&1
// OK

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=memory -fsanitize-memory-track-origins -pie %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TRACK-ORIGINS-2
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=memory -fsanitize-memory-track-origins=1 -pie %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TRACK-ORIGINS-1
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=memory -fsanitize-memory-track-origins=1 -fsanitize-memory-track-origins -pie %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TRACK-ORIGINS-2
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=memory -fsanitize-memory-track-origins=2 -fsanitize-memory-track-origins -pie %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TRACK-ORIGINS-2
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=memory -fno-sanitize-memory-track-origins -fsanitize-memory-track-origins -pie %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TRACK-ORIGINS-2
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=memory -fsanitize-memory-track-origins=0 -fsanitize-memory-track-origins=1 -pie %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TRACK-ORIGINS-1
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=memory -fsanitize-memory-track-origins=0 -fsanitize-memory-track-origins -pie %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TRACK-ORIGINS-2

// CHECK-TRACK-ORIGINS-1: -fsanitize-memory-track-origins=1

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=memory -fno-sanitize-memory-track-origins -pie %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-TRACK-ORIGINS
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=memory -fsanitize-memory-track-origins=0 -pie %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-TRACK-ORIGINS
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=memory -fsanitize-memory-track-origins -fno-sanitize-memory-track-origins -pie %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-TRACK-ORIGINS
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=memory -fsanitize-memory-track-origins -fsanitize-memory-track-origins=0 -pie %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-TRACK-ORIGINS
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=memory -fsanitize-memory-track-origins=2 -fno-sanitize-memory-track-origins -pie %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-TRACK-ORIGINS
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=memory -fsanitize-memory-track-origins=2 -fsanitize-memory-track-origins=0 -pie %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-TRACK-ORIGINS
// CHECK-NO-TRACK-ORIGINS-NOT: sanitize-memory-track-origins

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=memory -fsanitize-memory-track-origins=2 -pie %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TRACK-ORIGINS-2
// CHECK-TRACK-ORIGINS-2: -fsanitize-memory-track-origins=2

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=memory -fsanitize-memory-track-origins=3 -pie %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TRACK-ORIGINS-3
// CHECK-TRACK-ORIGINS-3: error: invalid value '3' in '-fsanitize-memory-track-origins=3'

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=memory -fsanitize-memory-use-after-dtor %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-USE-AFTER-DTOR
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=memory -fno-sanitize-memory-use-after-dtor -fsanitize-memory-use-after-dtor %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-USE-AFTER-DTOR
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=memory %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-USE-AFTER-DTOR
// CHECK-USE-AFTER-DTOR: -cc1{{.*}}-fsanitize-memory-use-after-dtor

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=memory -fno-sanitize-memory-use-after-dtor %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-USE-AFTER-DTOR-OFF
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=memory -fsanitize-memory-use-after-dtor -fno-sanitize-memory-use-after-dtor %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-USE-AFTER-DTOR-OFF
// CHECK-USE-AFTER-DTOR-OFF-NOT: -cc1{{.*}}memory-use-after-dtor

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=address -fsanitize-address-field-padding=0 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-FIELD-PADDING-0
// CHECK-ASAN-FIELD-PADDING-0-NOT: -fsanitize-address-field-padding
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=address -fsanitize-address-field-padding=1 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-FIELD-PADDING-1
// CHECK-ASAN-FIELD-PADDING-1: -fsanitize-address-field-padding=1
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=address -fsanitize-address-field-padding=2 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-FIELD-PADDING-2
// CHECK-ASAN-FIELD-PADDING-2: -fsanitize-address-field-padding=2
// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=address -fsanitize-address-field-padding=3 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-FIELD-PADDING-3
// CHECK-ASAN-FIELD-PADDING-3: error: invalid value '3' in '-fsanitize-address-field-padding=3'
// RUN: %clang --target=x86_64-linux-gnu -fsanitize-address-field-padding=2 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-FIELD-PADDING-NO-ASAN
// CHECK-ASAN-FIELD-PADDING-NO-ASAN: warning: argument unused during compilation: '-fsanitize-address-field-padding=2'
// RUN: %clang --target=x86_64-linux-gnu -fsanitize-address-field-padding=2 -fsanitize=address -fno-sanitize=address %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-FIELD-PADDING-DISABLED-ASAN
// CHECK-ASAN-FIELD-PADDING-DISABLED-ASAN-NOT: warning: argument unused


// RUN: %clang --target=x86_64-linux-gnu -fsanitize=vptr -fno-sanitize=vptr -fsanitize=undefined,address %s -### 2>&1
// OK

// RUN: %clang --target=arm-linux-androideabi %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ANDROID-NO-ASAN
// CHECK-ANDROID-NO-ASAN: "-mrelocation-model" "pic"

// RUN: not %clang --target=aarch64-linux-android -fsanitize=memory %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-MSAN-ANDROID
// RUN: not %clang --target=i386-linux-android -fsanitize=memory %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-MSAN-ANDROID
// RUN: not %clang --target=x86_64-linux-android -fsanitize=memory %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-MSAN-ANDROID
// CHECK-MSAN-ANDROID: unsupported option '-fsanitize=memory' for target

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=memory %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-MSAN
// CHECK-MSAN: "-fno-assume-sane-operator-new"
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=address %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN
// CHECK-ASAN: "-fno-assume-sane-operator-new"

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=zzz %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-DIAG1
// CHECK-DIAG1: unsupported argument 'zzz' to option '-fsanitize='
// CHECK-DIAG1-NOT: unsupported argument 'zzz' to option '-fsanitize='

// RUN: not %clang --target=x86_64-apple-darwin10 -fsanitize=memory %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-MSAN-DARWIN
// CHECK-MSAN-DARWIN: unsupported option '-fsanitize=memory' for target 'x86_64-apple-darwin10'

// RUN: %clang --target=x86_64-apple-darwin10 -fsanitize=memory -fno-sanitize=memory %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-MSAN-NOMSAN-DARWIN
// CHECK-MSAN-NOMSAN-DARWIN-NOT: unsupported option

// RUN: not %clang --target=x86_64-apple-darwin10 -fsanitize=memory -fsanitize=thread,memory %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-MSAN-TSAN-MSAN-DARWIN
// CHECK-MSAN-TSAN-MSAN-DARWIN: unsupported option '-fsanitize=memory' for target 'x86_64-apple-darwin10'
// CHECK-MSAN-TSAN-MSAN-DARWIN-NOT: unsupported option

// RUN: not %clang --target=x86_64-apple-darwin10 -fsanitize=thread,memory -fsanitize=memory %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-MSAN-MSAN-DARWIN
// CHECK-TSAN-MSAN-MSAN-DARWIN: unsupported option '-fsanitize=memory' for target 'x86_64-apple-darwin10'
// CHECK-TSAN-MSAN-MSAN-DARWIN-NOT: unsupported option

// RUN: %clang --target=x86_64-apple-darwin -fsanitize=thread %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-X86-64-DARWIN
// CHECK-TSAN-X86-64-DARWIN-NOT: unsupported option
// RUN: %clang --target=x86_64-apple-macos -fsanitize=thread %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-X86-64-MACOS
// CHECK-TSAN-X86-64-MACOS-NOT: unsupported option
// RUN: %clang --target=arm64-apple-macos -fsanitize=thread %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-ARM64-MACOS
// CHECK-TSAN-ARM64-MACOS-NOT: unsupported option

// RUN: %clang --target=arm64-apple-ios-simulator -fsanitize=thread %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-ARM64-IOSSIMULATOR
// CHECK-TSAN-ARM64-IOSSIMULATOR-NOT: unsupported option

// RUN: %clang --target=arm64-apple-watchos-simulator -fsanitize=thread %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-ARM64-WATCHOSSIMULATOR
// CHECK-TSAN-ARM64-WATCHOSSIMULATOR-NOT: unsupported option

// RUN: %clang --target=arm64-apple-tvos-simulator -fsanitize=thread %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-ARM64-TVOSSIMULATOR
// CHECK-TSAN-ARM64-TVOSSIMULATOR-NOT: unsupported option

// RUN: %clang --target=x86_64-apple-ios-simulator -fsanitize=thread %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-X86-64-IOSSIMULATOR
// CHECK-TSAN-X86-64-IOSSIMULATOR-NOT: unsupported option

// RUN: %clang --target=x86_64-apple-watchos-simulator -fsanitize=thread %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-X86-64-WATCHOSSIMULATOR
// CHECK-TSAN-X86-64-WATCHOSSIMULATOR-NOT: unsupported option

// RUN: %clang --target=x86_64-apple-tvos-simulator -fsanitize=thread %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-X86-64-TVOSSIMULATOR
// CHECK-TSAN-X86-64-TVOSSIMULATOR-NOT: unsupported option

// RUN: not %clang --target=i386-apple-darwin -fsanitize=thread %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-I386-DARWIN
// CHECK-TSAN-I386-DARWIN: unsupported option '-fsanitize=thread' for target 'i386-apple-darwin'

// RUN: not %clang --target=arm-apple-ios -fsanitize=thread %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-ARM-IOS
// CHECK-TSAN-ARM-IOS: unsupported option '-fsanitize=thread' for target 'arm-apple-ios'

// RUN: not %clang --target=i386-apple-ios-simulator -fsanitize=thread %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-I386-IOSSIMULATOR
// CHECK-TSAN-I386-IOSSIMULATOR: unsupported option '-fsanitize=thread' for target 'i386-apple-ios-simulator'

// RUN: not %clang --target=i386-apple-tvos-simulator -fsanitize=thread %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-I386-TVOSSIMULATOR
// CHECK-TSAN-I386-TVOSSIMULATOR: unsupported option '-fsanitize=thread' for target 'i386-apple-tvos-simulator'

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=thread -fsanitize-thread-memory-access %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-MEMORY-ACCESS
// CHECK-TSAN-MEMORY-ACCESS-NOT: -cc1{{.*}}tsan-instrument-memory-accesses=0
// CHECK-TSAN-MEMORY-ACCESS-NOT: -cc1{{.*}}tsan-instrument-memintrinsics=0
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=thread -fno-sanitize-thread-memory-access %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-MEMORY-ACCESS-OFF
// CHECK-TSAN-MEMORY-ACCESS-OFF: -cc1{{.*}}tsan-instrument-memory-accesses=0{{.*}}tsan-instrument-memintrinsics=0
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=thread -fno-sanitize-thread-memory-access -fsanitize-thread-memory-access %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-MEMORY-ACCESS-BOTH
// CHECK-TSAN-MEMORY-ACCESS-BOTH-NOT: -cc1{{.*}}tsan-instrument-memory-accesses=0
// CHECK-TSAN-MEMORY-ACCESS-BOTH-NOT: -cc1{{.*}}tsan-instrument-memintrinsics=0
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=thread -fsanitize-thread-memory-access -fno-sanitize-thread-memory-access %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-MEMORY-ACCESS-BOTH-OFF
// CHECK-TSAN-MEMORY-ACCESS-BOTH-OFF: -cc1{{.*}}tsan-instrument-memory-accesses=0{{.*}}tsan-instrument-memintrinsics=0

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=thread -fsanitize-thread-func-entry-exit %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-FUNC-ENTRY-EXIT
// CHECK-TSAN-FUNC-ENTRY-EXIT-NOT: -cc1{{.*}}tsan-instrument-func-entry-exit=0
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=thread -fno-sanitize-thread-func-entry-exit %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-FUNC-ENTRY-EXIT-OFF
// CHECK-TSAN-FUNC-ENTRY-EXIT-OFF: -cc1{{.*}}tsan-instrument-func-entry-exit=0
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=thread -fno-sanitize-thread-func-entry-exit -fsanitize-thread-func-entry-exit %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-FUNC-ENTRY-EXIT-BOTH
// CHECK-TSAN-FUNC-ENTRY-EXIT-BOTH-NOT: -cc1{{.*}}tsan-instrument-func-entry-exit=0
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=thread -fsanitize-thread-func-entry-exit -fno-sanitize-thread-func-entry-exit %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-FUNC-ENTRY-EXIT-BOTH-OFF
// CHECK-TSAN-FUNC-ENTRY-EXIT-BOTH-OFF: -cc1{{.*}}tsan-instrument-func-entry-exit=0

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=thread -fsanitize-thread-atomics %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-ATOMICS
// CHECK-TSAN-ATOMICS-NOT: -cc1{{.*}}tsan-instrument-atomics=0
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=thread -fno-sanitize-thread-atomics %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-ATOMICS-OFF
// CHECK-TSAN-ATOMICS-OFF: -cc1{{.*}}tsan-instrument-atomics=0
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=thread -fno-sanitize-thread-atomics -fsanitize-thread-atomics %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-ATOMICS-BOTH
// CHECK-TSAN-ATOMICS-BOTH-NOT: -cc1{{.*}}tsan-instrument-atomics=0
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=thread -fsanitize-thread-atomics -fno-sanitize-thread-atomics %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-ATOMICS-BOTH-OFF
// CHECK-TSAN-ATOMICS-BOTH-OFF: -cc1{{.*}}tsan-instrument-atomics=0

// RUN: not %clang --target=x86_64-apple-darwin10 -mmacos-version-min=10.8 -fsanitize=vptr %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-VPTR-DARWIN-OLD
// CHECK-VPTR-DARWIN-OLD: unsupported option '-fsanitize=vptr' for target 'x86_64-apple-darwin10'

// RUN: not %clang --target=arm-apple-ios4 -fsanitize=vptr %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-VPTR-IOS-OLD
// CHECK-VPTR-IOS-OLD: unsupported option '-fsanitize=vptr' for target 'arm-apple-ios4'

// RUN: %clang --target=aarch64-apple-darwin15.0.0 -fsanitize=vptr %s -### 2>&1
// OK

// RUN: %clang --target=x86_64-apple-darwin15.0.0-simulator -fsanitize=vptr %s -### 2>&1
// OK

// RUN: %clang --target=x86_64-apple-darwin10 -mmacos-version-min=10.9 -fsanitize=alignment,vptr %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-VPTR-DARWIN-NEW
// CHECK-VPTR-DARWIN-NEW: -fsanitize=alignment,vptr

// RUN: %clang --target=armv7-apple-ios7 -miphoneos-version-min=7.0 -fsanitize=address %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-IOS
// CHECK-ASAN-IOS: -fsanitize=address

// RUN: %clang --target=i386-pc-openbsd -fsanitize=undefined %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UBSAN-OPENBSD
// CHECK-UBSAN-OPENBSD: "-fsanitize={{((signed-integer-overflow|integer-divide-by-zero|function|shift-base|shift-exponent|unreachable|return|vla-bound|alignment|null|pointer-overflow|float-cast-overflow|array-bounds|enum|bool|builtin|returns-nonnull-attribute|nonnull-attribute),?){18}"}}

// RUN: not %clang --target=i386-pc-openbsd -fsanitize=address %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-OPENBSD
// CHECK-ASAN-OPENBSD: unsupported option '-fsanitize=address' for target 'i386-pc-openbsd'

// RUN: not %clang --target=i386-pc-openbsd -fsanitize=leak %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-LSAN-OPENBSD
// CHECK-LSAN-OPENBSD: unsupported option '-fsanitize=leak' for target 'i386-pc-openbsd'

// RUN: not %clang --target=i386-pc-openbsd -fsanitize=thread %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-OPENBSD
// CHECK-TSAN-OPENBSD: unsupported option '-fsanitize=thread' for target 'i386-pc-openbsd'

// RUN: not %clang --target=i386-pc-openbsd -fsanitize=memory %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-MSAN-OPENBSD
// CHECK-MSAN-OPENBSD: unsupported option '-fsanitize=memory' for target 'i386-pc-openbsd'

// RUN: %clang --target=x86_64-apple-darwin -fsanitize=leak %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-LSAN-X86-64-DARWIN
// CHECK-LSAN-X86-64-DARWIN-NOT: unsupported option

// RUN: %clang --target=x86_64-apple-ios-simulator -fsanitize=leak %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-LSAN-X86-64-IOSSIMULATOR
// CHECK-LSAN-X86-64-IOSSIMULATOR-NOT: unsupported option

// RUN: %clang --target=x86_64-apple-tvos-simulator -fsanitize=leak %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-LSAN-X86-64-TVOSSIMULATOR
// CHECK-LSAN-X86-64-TVOSSIMULATOR-NOT: unsupported option

// RUN: %clang --target=i386-apple-darwin -fsanitize=leak %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-LSAN-I386-DARWIN
// CHECK-LSAN-I386-DARWIN-NOT: unsupported option

// RUN: %clang --target=arm-apple-ios -fsanitize=leak %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-LSAN-ARM-IOS
// CHECK-LSAN-ARM-IOS-NOT: unsupported option

// RUN: %clang --target=i386-apple-ios-simulator -fsanitize=leak %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-LSAN-I386-IOSSIMULATOR
// CHECK-LSAN-I386-IOSSIMULATOR-NOT: unsupported option

// RUN: %clang --target=i386-apple-tvos-simulator -fsanitize=leak %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-LSAN-I386-TVOSSIMULATOR
// CHECK-LSAN-I386-TVOSSIMULATOR-NOT: unsupported option


// RUN: not %clang --target=x86_64-linux-gnu -fsanitize-trap=address -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-TRAP
// CHECK-ASAN-TRAP: error: unsupported argument 'address' to option '-fsanitize-trap='

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize-trap=hwaddress -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-HWASAN-TRAP
// CHECK-HWASAN-TRAP: error: unsupported argument 'hwaddress' to option '-fsanitize-trap='

// RUN: not %clang_cl -fsanitize=address -c -MDd -### -- %s 2>&1 | FileCheck %s -check-prefix=CHECK-ASAN-DEBUGRTL
// RUN: not %clang_cl -fsanitize=address -c -MTd -### -- %s 2>&1 | FileCheck %s -check-prefix=CHECK-ASAN-DEBUGRTL
// RUN: not %clang_cl -fsanitize=address -c -LDd -### -- %s 2>&1 | FileCheck %s -check-prefix=CHECK-ASAN-DEBUGRTL
// RUN: not %clang_cl -fsanitize=address -c -MD -MDd -### -- %s 2>&1 | FileCheck %s -check-prefix=CHECK-ASAN-DEBUGRTL
// RUN: not %clang_cl -fsanitize=address -c -MT -MTd -### -- %s 2>&1 | FileCheck %s -check-prefix=CHECK-ASAN-DEBUGRTL
// RUN: not %clang_cl -fsanitize=address -c -LD -LDd -### -- %s 2>&1 | FileCheck %s -check-prefix=CHECK-ASAN-DEBUGRTL
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

// RUN: %clang --target=powerpc64-unknown-linux-gnu -fsanitize=memory %s -### 2>&1 | FileCheck %s -check-prefix=CHECK-SANM
// RUN: %clang --target=powerpc64le-unknown-linux-gnu -fsanitize=memory %s -### 2>&1 | FileCheck %s -check-prefix=CHECK-SANM
// CHECK-SANM: "-fsanitize=memory"

// RUN: %clang --target=x86_64--freebsd -fsanitize=kernel-address %s -### 2>&1 | FileCheck %s -check-prefix=KERNEL-ADDRESS-FREEBSD
// RUN: %clang --target=aarch64--freebsd -fsanitize=kernel-address %s -### 2>&1 | FileCheck %s -check-prefix=KERNEL-ADDRESS-FREEBSD
// KERNEL-ADDRESS-FREEBSD: "-fsanitize=kernel-address"

// RUN: %clang --target=x86_64--freebsd -fsanitize=kernel-memory %s -### 2>&1 | FileCheck %s -check-prefix=KERNEL-MEMORY-FREEBSD
// RUN: %clang --target=aarch64--freebsd -fsanitize=kernel-memory %s -### 2>&1 | FileCheck %s -check-prefix=KERNEL-MEMORY-FREEBSD
// KERNEL-MEMORY-FREEBSD: "-fsanitize=kernel-memory"

// * NetBSD; please keep ordered as in Sanitizers.def *

// RUN: %clang --target=i386--netbsd -fsanitize=address %s -### 2>&1 | FileCheck %s -check-prefix=ADDRESS-NETBSD
// RUN: %clang --target=x86_64--netbsd -fsanitize=address %s -### 2>&1 | FileCheck %s -check-prefix=ADDRESS-NETBSD
// ADDRESS-NETBSD: "-fsanitize=address"

// RUN: %clang --target=x86_64--netbsd -fsanitize=kernel-address %s -### 2>&1 | FileCheck %s -check-prefix=KERNEL-ADDRESS-NETBSD
// KERNEL-ADDRESS-NETBSD: "-fsanitize=kernel-address"

// RUN: %clang --target=x86_64--netbsd -fsanitize=hwaddress %s -### 2>&1 | FileCheck %s -check-prefix=HWADDRESS-NETBSD
// HWADDRESS-NETBSD: "-fsanitize=hwaddress"

// RUN: %clang --target=x86_64--netbsd -fsanitize=kernel-hwaddress %s -### 2>&1 | FileCheck %s -check-prefix=KERNEL-HWADDRESS-NETBSD
// KERNEL-HWADDRESS-NETBSD: "-fsanitize=kernel-hwaddress"

// RUN: %clang --target=x86_64--netbsd -fsanitize=memory %s -### 2>&1 | FileCheck %s -check-prefix=MEMORY-NETBSD
// MEMORY-NETBSD: "-fsanitize=memory"

// RUN: %clang --target=x86_64--netbsd -fsanitize=kernel-memory %s -### 2>&1 | FileCheck %s -check-prefix=KERNEL-MEMORY-NETBSD
// KERNEL-MEMORY-NETBSD: "-fsanitize=kernel-memory"

// RUN: %clang --target=x86_64--netbsd -fsanitize=thread %s -### 2>&1 | FileCheck %s -check-prefix=THREAD-NETBSD
// THREAD-NETBSD: "-fsanitize=thread"

// RUN: %clang --target=i386--netbsd -fsanitize=leak %s -### 2>&1 | FileCheck %s -check-prefix=LEAK-NETBSD
// RUN: %clang --target=x86_64--netbsd -fsanitize=leak %s -### 2>&1 | FileCheck %s -check-prefix=LEAK-NETBSD
// LEAK-NETBSD: "-fsanitize=leak"

// RUN: %clang --target=i386--netbsd -fsanitize=function %s -### 2>&1 | FileCheck %s -check-prefix=FUNCTION-NETBSD
// RUN: %clang --target=x86_64--netbsd -fsanitize=function %s -### 2>&1 | FileCheck %s -check-prefix=FUNCTION-NETBSD
// FUNCTION-NETBSD: "-fsanitize=function"

// RUN: %clang --target=i386--netbsd -fsanitize=vptr %s -### 2>&1 | FileCheck %s -check-prefix=VPTR-NETBSD
// RUN: %clang --target=x86_64--netbsd -fsanitize=vptr %s -### 2>&1 | FileCheck %s -check-prefix=VPTR-NETBSD
// VPTR-NETBSD: "-fsanitize=vptr"

// RUN: %clang --target=x86_64--netbsd -fsanitize=dataflow %s -### 2>&1 | FileCheck %s -check-prefix=DATAFLOW-NETBSD
// DATAFLOW-NETBSD: "-fsanitize=dataflow"

// RUN: %clang --target=i386--netbsd -fsanitize=cfi -flto -fvisibility=hidden -resource-dir=%S/Inputs/resource_dir -c %s -### 2>&1 | FileCheck %s -check-prefix=CFI-NETBSD
// RUN: %clang --target=x86_64--netbsd -fsanitize=cfi -flto -fvisibility=hidden -resource-dir=%S/Inputs/resource_dir -c %s -### 2>&1 | FileCheck %s -check-prefix=CFI-NETBSD
// CFI-NETBSD: "-fsanitize=cfi-derived-cast,cfi-icall,cfi-mfcall,cfi-unrelated-cast,cfi-nvcall,cfi-vcall"

// RUN: %clang --target=i386--netbsd -fsanitize=safe-stack %s -### 2>&1 | FileCheck %s -check-prefix=SAFESTACK-NETBSD
// RUN: %clang --target=x86_64--netbsd -fsanitize=safe-stack %s -### 2>&1 | FileCheck %s -check-prefix=SAFESTACK-NETBSD
// SAFESTACK-NETBSD: "-fsanitize=safe-stack"

// RUN: %clang --target=x86_64--netbsd -fsanitize=shadow-call-stack %s -### 2>&1 | FileCheck %s -check-prefix=SHADOW-CALL-STACK-NETBSD
// SHADOW-CALL-STACK-NETBSD: "-fsanitize=shadow-call-stack"

// RUN: %clang --target=i386--netbsd -fsanitize=undefined %s -### 2>&1 | FileCheck %s -check-prefix=UNDEFINED-NETBSD
// RUN: %clang --target=x86_64--netbsd -fsanitize=undefined %s -### 2>&1 | FileCheck %s -check-prefix=UNDEFINED-NETBSD
// UNDEFINED-NETBSD: "-fsanitize={{.*}}

// RUN: %clang --target=i386--netbsd -fsanitize=scudo %s -### 2>&1 | FileCheck %s -check-prefix=SCUDO-NETBSD
// RUN: %clang --target=x86_64--netbsd -fsanitize=scudo %s -### 2>&1 | FileCheck %s -check-prefix=SCUDO-NETBSD
// SCUDO-NETBSD: "-fsanitize=scudo"

// RUN: not %clang --target=x86_64-scei-ps4 -fsanitize=dataflow %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-DFSAN-PS4
// CHECK-DFSAN-PS4: unsupported option '-fsanitize=dataflow' for target 'x86_64-scei-ps4'
// RUN: not %clang --target=x86_64-scei-ps4 -fsanitize=leak %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-LSAN-PS4
// CHECK-LSAN-PS4: unsupported option '-fsanitize=leak' for target 'x86_64-scei-ps4'
// RUN: not %clang --target=x86_64-scei-ps4 -fsanitize=memory %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-MSAN-PS4
// CHECK-MSAN-PS4: unsupported option '-fsanitize=memory' for target 'x86_64-scei-ps4'
// RUN: not %clang --target=x86_64-scei-ps4 -fsanitize=thread %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-PS4
// CHECK-TSAN-PS4: unsupported option '-fsanitize=thread' for target 'x86_64-scei-ps4'
// RUN: %clang --target=x86_64-scei-ps4 -fsanitize=address %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-PS4
// Make sure there are no *.{o,bc} or -l passed before the ASan library.
// CHECK-ASAN-PS4-NOT: {{(\.(o|bc)"? |-l).*-lSceDbgAddressSanitizer_stub_weak}}
// CHECK-ASAN-PS4: --dependent-lib=libSceDbgAddressSanitizer_stub_weak.a
// CHECK-ASAN-PS4-NOT: {{(\.(o|bc)"? |-l).*-lSceDbgAddressSanitizer_stub_weak}}
// CHECK-ASAN-PS4: -lSceDbgAddressSanitizer_stub_weak
// RUN: %clang --target=x86_64-scei-ps4 -fsanitize=address -nostdlib %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-NOLIB-PS4
// RUN: %clang --target=x86_64-scei-ps4 -fsanitize=address -nodefaultlibs %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-NOLIB-PS4
// RUN: %clang --target=x86_64-scei-ps4 -fsanitize=address -nodefaultlibs -nostdlib  %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-NOLIB-PS4
// CHECK-ASAN-NOLIB-PS4-NOT: SceDbgAddressSanitizer_stub_weak

// RUN: %clang --target=x86_64-sie-ps5 -fsanitize=address %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-PS5
// Make sure there are no *.{o,bc} or -l passed before the ASan library.
// CHECK-ASAN-PS5-NOT: {{(\.(o|bc)"? |-l).*-lSceAddressSanitizer_nosubmission_stub_weak}}
// CHECK-ASAN-PS5: --dependent-lib=libSceAddressSanitizer_nosubmission_stub_weak.a
// CHECK-ASAN-PS5-NOT: {{(\.(o|bc)"? |-l).*-lSceAddressSanitizer_nosubmission_stub_weak}}
// CHECK-ASAN-PS5: -lSceAddressSanitizer_nosubmission_stub_weak
// RUN: %clang --target=x86_64-sie-ps5 -fsanitize=address -nostdlib %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-NOLIB-PS5
// RUN: %clang --target=x86_64-sie-ps5 -fsanitize=address -nodefaultlibs %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-NOLIB-PS5
// RUN: %clang --target=x86_64-sie-ps5 -fsanitize=address -nodefaultlibs -nostdlib  %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-NOLIB-PS5
// CHECK-ASAN-NOLIB-PS5-NOT: SceAddressSanitizer_nosubmission_stub_weak

// RUN: %clang --target=x86_64-sie-ps5 -fsanitize=thread %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-PS5
// Make sure there are no *.{o,bc} or -l passed before the TSan library.
// CHECK-TSAN-PS5-NOT: {{(\.(o|bc)"? |-l).*-lSceThreadSanitizer_nosubmission_stub_weak}}
// CHECK-TSAN-PS5: --dependent-lib=libSceThreadSanitizer_nosubmission_stub_weak.a
// CHECK-TSAN-PS5-NOT: {{(\.(o|bc)"? |-l).*-lSceThreadSanitizer_nosubmission_stub_weak}}
// CHECK-TSAN-PS5: -lSceThreadSanitizer_nosubmission_stub_weak
// RUN: %clang --target=x86_64-sie-ps5 -fsanitize=thread -nostdlib %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-NOLIB-PS5
// RUN: %clang --target=x86_64-sie-ps5 -fsanitize=thread -nodefaultlibs %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-NOLIB-PS5
// RUN: %clang --target=x86_64-sie-ps5 -fsanitize=thread -nodefaultlibs -nostdlib  %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-NOLIB-PS5
// CHECK-TSAN-NOLIB-PS5-NOT: SceThreadSanitizer_nosubmission_stub_weak


// RUN: %clang --target=aarch64 -fsanitize=shadow-call-stack -ffixed-x18 %s -### 2>&1 | FileCheck %s --check-prefix=AARCH64-SCS
// RUN: %clang --target=aarch64_be -fsanitize=shadow-call-stack -ffixed-x18 %s -### 2>&1 | FileCheck %s --check-prefix=AARCH64-SCS
// AARCH64-SCS: "-fsanitize=shadow-call-stack"

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=hwaddress %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-HWASAN-INTERCEPTOR-ABI
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=hwaddress -fsanitize-hwaddress-abi=interceptor %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-HWASAN-INTERCEPTOR-ABI
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=hwaddress -fsanitize-hwaddress-abi=platform %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-HWASAN-PLATFORM-ABI
// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=hwaddress -fsanitize-hwaddress-abi=foo %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-HWASAN-FOO-ABI
// CHECK-HWASAN-INTERCEPTOR-ABI: "-default-function-attr" "hwasan-abi=interceptor"
// CHECK-HWASAN-PLATFORM-ABI: "-default-function-attr" "hwasan-abi=platform"
// CHECK-HWASAN-FOO-ABI: error: invalid value 'foo' in '-fsanitize-hwaddress-abi=foo'

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=hwaddress -fsanitize-hwaddress-experimental-aliasing %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-HWASAN-ALIAS
// CHECK-HWASAN-ALIAS: "-mllvm" "-hwasan-experimental-use-page-aliases=1"

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=address,pointer-compare,pointer-subtract %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-POINTER-ALL
// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=pointer-compare %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-POINTER-CMP-NEEDS-ADDRESS
// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=pointer-subtract %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-POINTER-SUB-NEEDS-ADDRESS
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=pointer-subtract -fno-sanitize=pointer-subtract %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-POINTER-SUB
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=pointer-compare -fno-sanitize=pointer-compare %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-POINTER-CMP
// CHECK-POINTER-ALL: -cc1{{.*}}-fsanitize={{[^"]*}}pointer-compare,pointer-subtract{{.*}}" {{.*}} "-mllvm" "-asan-detect-invalid-pointer-cmp" {{.*}}"-mllvm" "-asan-detect-invalid-pointer-sub"
// CHECK-POINTER-CMP-NEEDS-ADDRESS: error: invalid argument '-fsanitize=pointer-compare' only allowed with '-fsanitize=address'
// CHECK-POINTER-SUB-NEEDS-ADDRESS: error: invalid argument '-fsanitize=pointer-subtract' only allowed with '-fsanitize=address'
// CHECK-NO-POINTER-SUB-NOT: {{.*}}asan-detect-invalid-pointer{{.*}}
// CHECK-NO-POINTER-CMP-NOT: {{.*}}asan-detect-invalid-pointer{{.*}}

// RUN: not %clang --target=x86_64-sie-ps5 -fsanitize=function %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UBSAN-FUNCTION-TARGET
// RUN: not %clang --target=x86_64-sie-ps5 -fsanitize=undefined -fsanitize=function %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UBSAN-FUNCTION-TARGET
// RUN: not %clang --target=x86_64-sie-ps5 -fsanitize=kcfi %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UBSAN-KCFI-TARGET
// RUN: not %clang --target=x86_64-sie-ps5 -fsanitize=function -fsanitize=kcfi %s -### 2>&1 | FileCheck %s  --check-prefix=CHECK-UBSAN-KCFI-TARGET --check-prefix=CHECK-UBSAN-FUNCTION-TARGET
// RUN: %clang --target=x86_64-sie-ps5 -fsanitize=undefined %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UBSAN-UNDEFINED

// RUN: not %clang --target=armv6t2-eabi -mexecute-only -fsanitize=function %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UBSAN-FUNCTION-MEXECUTE-ONLY
// RUN: not %clang --target=armv6t2-eabi -mpure-code -fsanitize=function %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UBSAN-FUNCTION-MPURE-CODE
// RUN: not %clang --target=armv6t2-eabi -mexecute-only -fsanitize=kcfi %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UBSAN-KCFI-MEXECUTE-ONLY
// RUN: not %clang --target=armv6t2-eabi -mpure-code -fsanitize=kcfi %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UBSAN-KCFI-MPURE-CODE
// RUN: %clang --target=armv6t2-eabi -mexecute-only -fsanitize=undefined %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UBSAN-UNDEFINED

// RUN: not %clang --target=aarch64 -mexecute-only -fsanitize=function %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UBSAN-FUNCTION-MEXECUTE-ONLY
// RUN: not %clang --target=aarch64 -mpure-code -fsanitize=function %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UBSAN-FUNCTION-MPURE-CODE
// RUN: not %clang --target=aarch64 -mexecute-only -fsanitize=kcfi %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UBSAN-KCFI-MEXECUTE-ONLY
// RUN: not %clang --target=aarch64 -mpure-code -fsanitize=kcfi %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UBSAN-KCFI-MPURE-CODE
// RUN: %clang --target=aarch64 -mexecute-only -fsanitize=undefined %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UBSAN-UNDEFINED

// CHECK-UBSAN-KCFI-TARGET-DAG: error: unsupported option '-fsanitize=kcfi' for target 'x86_64-sie-ps5'
// CHECK-UBSAN-KCFI-MEXECUTE-ONLY-DAG: error: invalid argument '-fsanitize=kcfi' not allowed with '-mexecute-only'
// CHECK-UBSAN-KCFI-MPURE-CODE-DAG: error: invalid argument '-fsanitize=kcfi' not allowed with '-mpure-code'
// CHECK-UBSAN-FUNCTION-TARGET-DAG: error: unsupported option '-fsanitize=function' for target 'x86_64-sie-ps5'
// CHECK-UBSAN-FUNCTION-MEXECUTE-ONLY-DAG: error: invalid argument '-fsanitize=function' not allowed with '-mexecute-only'
// CHECK-UBSAN-FUNCTION-MPURE-CODE-DAG: error: invalid argument '-fsanitize=function' not allowed with '-mpure-code'
// CHECK-UBSAN-UNDEFINED: "-fsanitize={{((alignment|array-bounds|bool|builtin|enum|float-cast-overflow|integer-divide-by-zero|nonnull-attribute|null|pointer-overflow|return|returns-nonnull-attribute|shift-base|shift-exponent|signed-integer-overflow|unreachable|vla-bound),?){17}"}}
