// XFAIL: !sanitizer-ignorelists

// Check that the default ignorelist is not added as an extra dependency.
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=address -resource-dir=%S/Inputs/resource_dir %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-DEFAULT-IGNORELIST-ASAN --implicit-check-not=fdepfile-entry --implicit-check-not=-fsanitize-ignorelist=
// CHECK-DEFAULT-IGNORELIST-ASAN: -fsanitize-system-ignorelist={{.*[^w]}}asan_ignorelist.txt
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=hwaddress -resource-dir=%S/Inputs/resource_dir %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-DEFAULT-IGNORELIST-HWASAN --implicit-check-not=fdepfile-entry --implicit-check-not=-fsanitize-ignorelist=
// CHECK-DEFAULT-IGNORELIST-HWASAN: -fsanitize-system-ignorelist={{.*}}hwasan_ignorelist.txt

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=integer -resource-dir=%S/Inputs/resource_dir %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-DEFAULT-UBSAN-IGNORELIST --implicit-check-not=fdepfile-entry --implicit-check-not=-fsanitize-ignorelist=
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=nullability -resource-dir=%S/Inputs/resource_dir %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-DEFAULT-UBSAN-IGNORELIST --implicit-check-not=fdepfile-entry --implicit-check-not=-fsanitize-ignorelist=
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=undefined -resource-dir=%S/Inputs/resource_dir %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-DEFAULT-UBSAN-IGNORELIST --implicit-check-not=fdepfile-entry --implicit-check-not=-fsanitize-ignorelist=
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=alignment -resource-dir=%S/Inputs/resource_dir %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-DEFAULT-UBSAN-IGNORELIST --implicit-check-not=fdepfile-entry --implicit-check-not=-fsanitize-ignorelist=
// Pass `--no-xcselect` to prevent the driver injecting -fdepfile-entry=<sysroot>/SDKSettings.json when xcselect is active.
// RUN: %clang --target=%itanium_abi_triple --no-xcselect -fsanitize=float-divide-by-zero -resource-dir=%S/Inputs/resource_dir %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-DEFAULT-UBSAN-IGNORELIST --implicit-check-not=fdepfile-entry --implicit-check-not=-fsanitize-ignorelist=
// CHECK-DEFAULT-UBSAN-IGNORELIST: -fsanitize-system-ignorelist={{.*}}ubsan_ignorelist.txt

// Check that combining ubsan and another sanitizer results in both ignorelists being used.
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=undefined,address -resource-dir=%S/Inputs/resource_dir %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-DEFAULT-UBSAN-IGNORELIST --check-prefix=CHECK-DEFAULT-IGNORELIST-ASAN --implicit-check-not=fdepfile-entry --implicit-check-not=-fsanitize-ignorelist=

// If cfi_ignorelist.txt cannot be found in the resource dir, driver should fail.
// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=cfi -flto -fvisibility=default -resource-dir=/dev/null %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-MISSING-CFI-IGNORELIST
// CHECK-MISSING-CFI-IGNORELIST: error: missing sanitizer ignorelist: '{{.*}}cfi_ignorelist.txt'

// -fno-sanitize-ignorelist disables checking for cfi_ignorelist.txt in the resource dir.
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=cfi -flto -fvisibility=default -fno-sanitize-ignorelist -resource-dir=/dev/null %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-MISSING-CFI-NO-IGNORELIST
// CHECK-MISSING-CFI-NO-IGNORELIST-NOT: error: no such file or directory: '{{.*}}cfi_ignorelist.txt'
