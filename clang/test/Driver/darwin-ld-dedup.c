// REQUIRES: system-darwin

// -no_deduplicate is only present from ld64 version 262 and later.
// RUN: %clang -target x86_64-apple-darwin10 -### %s \
// RUN:   -mlinker-version=261 -O0 2>&1 | FileCheck -check-prefix=LINK_DEDUP %s

// Add -no_deduplicate when either -O0 or -O1 is explicitly specified
// RUN: %clang -target x86_64-apple-darwin10 -### %s \
// RUN:   -mlinker-version=262 -O0 2>&1 | FileCheck -check-prefix=LINK_NODEDUP %s
// RUN: %clang -target x86_64-apple-darwin10 -### %s \
// RUN:   -mlinker-version=262 -O1 2>&1 | FileCheck -check-prefix=LINK_NODEDUP %s

// RUN: %clang -target x86_64-apple-darwin10 -### %s \
// RUN:   -mlinker-version=262 -O2 2>&1 | FileCheck -check-prefix=LINK_DEDUP %s
// RUN: %clang -target x86_64-apple-darwin10 -### %s \
// RUN:   -mlinker-version=262 -O3 2>&1 | FileCheck -check-prefix=LINK_DEDUP %s
// RUN: %clang -target x86_64-apple-darwin10 -### %s \
// RUN:   -mlinker-version=262 -Os 2>&1 | FileCheck -check-prefix=LINK_DEDUP %s
// RUN: %clang -target x86_64-apple-darwin10 -### %s \
// RUN:   -mlinker-version=262 -O4 2>&1 | FileCheck -check-prefix=LINK_DEDUP %s
// RUN: %clang -target x86_64-apple-darwin10 -### %s \
// RUN:   -mlinker-version=262 -Ofast 2>&1 | FileCheck -check-prefix=LINK_DEDUP %s

// Add -no_deduplicate when no -O option is specified *and* this is a compile+link
// (implicit -O0)
// RUN: %clang -target x86_64-apple-darwin10 -### %s \
// RUN:   -mlinker-version=262 2>&1 | FileCheck -check-prefix=LINK_NODEDUP %s

// Do *not* add -no_deduplicate when no -O option is specified and this is just a link
// (since we can't imply -O0)
// RUN: rm -f %t.o %t.bin
// RUN: yaml2obj %S/Inputs/empty-x86_64-apple-darwin.yaml -o %t.o
// RUN: %clang -target x86_64-apple-darwin10 %t.o -### -mlinker-version=262 \
// RUN:   -o %t.bin 2>&1 | FileCheck -check-prefix=LINK_DEDUP %s
// RUN: %clang -target x86_64-apple-darwin10 %t.o -### -mlinker-version=262 \
// RUN:   -O0 -o %t.bin 2>&1 | FileCheck -check-prefix=LINK_NODEDUP %s
// RUN: %clang -target x86_64-apple-darwin10 %t.o -### -mlinker-version=262 \
// RUN:   -O1 -o %t.bin 2>&1 | FileCheck -check-prefix=LINK_NODEDUP %s
// RUN: %clang -target x86_64-apple-darwin10 %t.o -### -mlinker-version=262 \
// RUN:   -O2 -o %t.bin 2>&1 | FileCheck -check-prefix=LINK_DEDUP %s

// LINK_NODEDUP: {{ld(.exe)?"}}
// LINK_NODEDUP: "-no_deduplicate"

// LINK_DEDUP: {{ld(.exe)?"}}
// LINK_DEDUP-NOT: "-no_deduplicate"
