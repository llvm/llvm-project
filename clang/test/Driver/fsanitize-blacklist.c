// General blacklist usage.

// PR12920
// REQUIRES: clang-driver

// Make sure we don't match the -NOT lines with the linker invocation.
// Delimiters match the start of the cc1 and the start of the linker lines
// for fragile tests.
// DELIMITERS: {{^ *"}}

// RUN: echo "fun:foo" > %t.good
// RUN: echo "fun:bar" > %t.second
// RUN: echo "badline" > %t.bad

// RUN: %clang -target x86_64-linux-gnu -fsanitize=address -fsanitize-blacklist=%t.good -fsanitize-blacklist=%t.second %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-BLACKLIST
// RUN: %clang -target aarch64-linux-gnu -fsanitize=hwaddress -fsanitize-blacklist=%t.good -fsanitize-blacklist=%t.second %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-BLACKLIST
// CHECK-BLACKLIST: -fsanitize-blacklist={{.*}}.good" "-fsanitize-blacklist={{.*}}.second

// Now, check for -fdepfile-entry flags.
// RUN: %clang -target x86_64-linux-gnu -fsanitize=address -fsanitize-blacklist=%t.good -fsanitize-blacklist=%t.second %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-BLACKLIST2
// CHECK-BLACKLIST2: -fdepfile-entry={{.*}}.good" "-fdepfile-entry={{.*}}.second

// Check that the default blacklist is not added as an extra dependency.
// RUN: %clang -target x86_64-linux-gnu -fsanitize=address -resource-dir=%S/Inputs/resource_dir %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-DEFAULT-BLACKLIST-ASAN --implicit-check-not=fdepfile-entry --implicit-check-not=-fsanitize-blacklist=
// CHECK-DEFAULT-BLACKLIST-ASAN: -fsanitize-blacklist={{.*[^w]}}asan_blacklist.txt
// RUN: %clang -target aarch64-linux-gnu -fsanitize=hwaddress -resource-dir=%S/Inputs/resource_dir %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-DEFAULT-BLACKLIST-HWASAN --implicit-check-not=fdepfile-entry --implicit-check-not=-fsanitize-blacklist=
// CHECK-DEFAULT-BLACKLIST-HWASAN: -fsanitize-blacklist={{.*}}hwasan_blacklist.txt

// RUN: %clang -target x86_64-linux-gnu -fsanitize=integer -resource-dir=%S/Inputs/resource_dir %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-DEFAULT-UBSAN-BLACKLIST --implicit-check-not=fdepfile-entry --implicit-check-not=-fsanitize-blacklist=
// RUN: %clang -target x86_64-linux-gnu -fsanitize=nullability -resource-dir=%S/Inputs/resource_dir %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-DEFAULT-UBSAN-BLACKLIST --implicit-check-not=fdepfile-entry --implicit-check-not=-fsanitize-blacklist=
// RUN: %clang -target x86_64-linux-gnu -fsanitize=undefined -resource-dir=%S/Inputs/resource_dir %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-DEFAULT-UBSAN-BLACKLIST --implicit-check-not=fdepfile-entry --implicit-check-not=-fsanitize-blacklist=
// RUN: %clang -target x86_64-linux-gnu -fsanitize=alignment -resource-dir=%S/Inputs/resource_dir %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-DEFAULT-UBSAN-BLACKLIST --implicit-check-not=fdepfile-entry --implicit-check-not=-fsanitize-blacklist=
// CHECK-DEFAULT-UBSAN-BLACKLIST: -fsanitize-blacklist={{.*}}ubsan_blacklist.txt

// Check that combining ubsan and another sanitizer results in both blacklists being used.
// RUN: %clang -target x86_64-linux-gnu -fsanitize=undefined,address -resource-dir=%S/Inputs/resource_dir %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-DEFAULT-UBSAN-BLACKLIST --check-prefix=CHECK-DEFAULT-ASAN-BLACKLIST --implicit-check-not=fdepfile-entry --implicit-check-not=-fsanitize-blacklist=

// Ignore -fsanitize-blacklist flag if there is no -fsanitize flag.
// RUN: %clang -target x86_64-linux-gnu -fsanitize-blacklist=%t.good %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-SANITIZE --check-prefix=DELIMITERS
// CHECK-NO-SANITIZE-NOT: -fsanitize-blacklist

// Ignore -fsanitize-blacklist flag if there is no -fsanitize flag.
// Now, check for the absense of -fdepfile-entry flags.
// RUN: %clang -target x86_64-linux-gnu -fsanitize-blacklist=%t.good %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-SANITIZE2 --check-prefix=DELIMITERS
// CHECK-NO-SANITIZE2-NOT: -fdepfile-entry

// Flag -fno-sanitize-blacklist wins if it is specified later.
// RUN: %clang -target x86_64-linux-gnu -fsanitize=address -fsanitize-blacklist=%t.good -fno-sanitize-blacklist %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-BLACKLIST --check-prefix=DELIMITERS
// CHECK-NO-BLACKLIST-NOT: -fsanitize-blacklist

// Driver barks on unexisting blacklist files.
// RUN: %clang -target x86_64-linux-gnu -fno-sanitize-blacklist -fsanitize-blacklist=unexisting.txt %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-SUCH-FILE
// CHECK-NO-SUCH-FILE: error: no such file or directory: 'unexisting.txt'

// Driver properly reports malformed blacklist files.
// RUN: %clang -target x86_64-linux-gnu -fsanitize=address -fsanitize-blacklist=%t.second -fsanitize-blacklist=%t.bad -fsanitize-blacklist=%t.good %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-BAD-BLACKLIST
// CHECK-BAD-BLACKLIST: error: malformed sanitizer blacklist: 'error parsing file '{{.*}}.bad': malformed line 1: 'badline''

// -fno-sanitize-blacklist disables all blacklists specified earlier.
// RUN: %clang -target x86_64-linux-gnu -fsanitize=address -fsanitize-blacklist=%t.good -fno-sanitize-blacklist -fsanitize-blacklist=%t.second %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ONLY-FIRST-DISABLED --implicit-check-not=-fsanitize-blacklist=
// CHECK-ONLY_FIRST-DISABLED-NOT: good
// CHECK-ONLY-FIRST-DISABLED: -fsanitize-blacklist={{.*}}.second
// CHECK-ONLY_FIRST-DISABLED-NOT: good

// DELIMITERS: {{^ *"}}
