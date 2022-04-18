// General ignorelist usage.

// PR12920

// Make sure we don't match the -NOT lines with the linker invocation.
// Delimiters match the start of the cc1 and the start of the linker lines
// for fragile tests.
// DELIMITERS: {{^ (\(in-process\)|")}}

// RUN: echo "fun:foo" > %t.good
// RUN: echo "fun:bar" > %t.second
// RUN: echo "badline" > %t.bad

// RUN: %clang -target x86_64-linux-gnu -fsanitize=address -fsanitize-ignorelist=%t.good -fsanitize-ignorelist=%t.second %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-IGNORELIST
// RUN: %clang -target aarch64-linux-gnu -fsanitize=hwaddress -fsanitize-ignorelist=%t.good -fsanitize-ignorelist=%t.second %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-IGNORELIST
// CHECK-IGNORELIST: -fsanitize-ignorelist={{.*}}.good" "-fsanitize-ignorelist={{.*}}.second

// Check that the default ignorelist is not added as an extra dependency.
// RUN: %clang -target x86_64-linux-gnu -fsanitize=address -resource-dir=%S/Inputs/resource_dir %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-DEFAULT-IGNORELIST-ASAN --implicit-check-not=fdepfile-entry --implicit-check-not=-fsanitize-ignorelist=
// CHECK-DEFAULT-IGNORELIST-ASAN: -fsanitize-system-ignorelist={{.*[^w]}}asan_ignorelist.txt
// RUN: %clang -target x86_64-linux-gnu -fsanitize=hwaddress -resource-dir=%S/Inputs/resource_dir %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-DEFAULT-IGNORELIST-HWASAN --implicit-check-not=fdepfile-entry --implicit-check-not=-fsanitize-ignorelist=
// CHECK-DEFAULT-IGNORELIST-HWASAN: -fsanitize-system-ignorelist={{.*}}hwasan_ignorelist.txt

// RUN: %clang -target x86_64-linux-gnu -fsanitize=integer -resource-dir=%S/Inputs/resource_dir %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-DEFAULT-UBSAN-IGNORELIST --implicit-check-not=fdepfile-entry --implicit-check-not=-fsanitize-ignorelist=
// RUN: %clang -target x86_64-linux-gnu -fsanitize=nullability -resource-dir=%S/Inputs/resource_dir %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-DEFAULT-UBSAN-IGNORELIST --implicit-check-not=fdepfile-entry --implicit-check-not=-fsanitize-ignorelist=
// RUN: %clang -target x86_64-linux-gnu -fsanitize=undefined -resource-dir=%S/Inputs/resource_dir %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-DEFAULT-UBSAN-IGNORELIST --implicit-check-not=fdepfile-entry --implicit-check-not=-fsanitize-ignorelist=
// RUN: %clang -target x86_64-linux-gnu -fsanitize=alignment -resource-dir=%S/Inputs/resource_dir %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-DEFAULT-UBSAN-IGNORELIST --implicit-check-not=fdepfile-entry --implicit-check-not=-fsanitize-ignorelist=
// RUN: %clang -target %itanium_abi_triple -fsanitize=float-divide-by-zero -resource-dir=%S/Inputs/resource_dir %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-DEFAULT-UBSAN-IGNORELIST --implicit-check-not=fdepfile-entry --implicit-check-not=-fsanitize-ignorelist=
// CHECK-DEFAULT-UBSAN-IGNORELIST: -fsanitize-system-ignorelist={{.*}}ubsan_ignorelist.txt

// Check that combining ubsan and another sanitizer results in both ignorelists being used.
// RUN: %clang -target x86_64-linux-gnu -fsanitize=undefined,address -resource-dir=%S/Inputs/resource_dir %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-DEFAULT-UBSAN-IGNORELIST --check-prefix=CHECK-DEFAULT-IGNORELIST-ASAN --implicit-check-not=fdepfile-entry --implicit-check-not=-fsanitize-ignorelist=

// Ignore -fsanitize-ignorelist flag if there is no -fsanitize flag.
// RUN: %clang -target x86_64-linux-gnu -fsanitize-ignorelist=%t.good %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-SANITIZE --check-prefix=DELIMITERS
// CHECK-NO-SANITIZE-NOT: -fsanitize-ignorelist

// Ignore -fsanitize-ignorelist flag if there is no -fsanitize flag.
// Now, check for the absence of -fdepfile-entry flags.
// RUN: %clang -target x86_64-linux-gnu -fsanitize-ignorelist=%t.good %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-SANITIZE2 --check-prefix=DELIMITERS
// CHECK-NO-SANITIZE2-NOT: -fdepfile-entry

// Flag -fno-sanitize-ignorelist wins if it is specified later.
// RUN: %clang -target x86_64-linux-gnu -fsanitize=address -fsanitize-ignorelist=%t.good -fno-sanitize-ignorelist %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-IGNORELIST --check-prefix=DELIMITERS
// CHECK-NO-IGNORELIST-NOT: -fsanitize-ignorelist

// Driver barks on unexisting ignorelist files.
// RUN: %clang -target x86_64-linux-gnu -fno-sanitize-ignorelist -fsanitize-ignorelist=unexisting.txt %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-SUCH-FILE
// CHECK-NO-SUCH-FILE: error: no such file or directory: 'unexisting.txt'

// Driver properly reports malformed ignorelist files.
// RUN: %clang -target x86_64-linux-gnu -fsanitize=address -fsanitize-ignorelist=%t.second -fsanitize-ignorelist=%t.bad -fsanitize-ignorelist=%t.good %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-BAD-IGNORELIST
// CHECK-BAD-IGNORELIST: error: malformed sanitizer ignorelist: 'error parsing file '{{.*}}.bad': malformed line 1: 'badline''

// -fno-sanitize-ignorelist disables all ignorelists specified earlier.
// RUN: %clang -target x86_64-linux-gnu -fsanitize=address -fsanitize-ignorelist=%t.good -fno-sanitize-ignorelist -fsanitize-ignorelist=%t.second %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ONLY-FIRST-DISABLED --implicit-check-not=-fsanitize-ignorelist=
// CHECK-ONLY_FIRST-DISABLED-NOT: good
// CHECK-ONLY-FIRST-DISABLED: -fsanitize-ignorelist={{.*}}.second
// CHECK-ONLY_FIRST-DISABLED-NOT: good

// -fno-sanitize-ignorelist disables the system ignorelists.
// RUN: %clang -target x86_64-linux-gnu -fsanitize=address -fno-sanitize-ignorelist %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-DISABLED-SYSTEM --check-prefix=DELIMITERS
// CHECK-DISABLED-SYSTEM-NOT: -fsanitize-system-ignorelist

// If cfi_ignorelist.txt cannot be found in the resource dir, driver should fail.
// RUN: %clang -target x86_64-linux-gnu -fsanitize=cfi -flto -fvisibility=default -resource-dir=/dev/null %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-MISSING-CFI-IGNORELIST
// CHECK-MISSING-CFI-IGNORELIST: error: no such file or directory: '{{.*}}cfi_ignorelist.txt'

// -fno-sanitize-ignorelist disables checking for cfi_ignorelist.txt in the resource dir.
// RUN: %clang -target x86_64-linux-gnu -fsanitize=cfi -flto -fvisibility=default -fno-sanitize-ignorelist -resource-dir=/dev/null %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-MISSING-CFI-NO-IGNORELIST
// CHECK-MISSING-CFI-NO-IGNORELIST-NOT: error: no such file or directory: '{{.*}}cfi_ignorelist.txt'

// DELIMITERS: {{^ *"}}
