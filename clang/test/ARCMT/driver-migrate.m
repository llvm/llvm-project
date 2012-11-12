// RUN: %clang -### -ccc-arcmt-migrate /foo/bar -fsyntax-only %s 2>&1 | FileCheck %s

// CHECK: "-arcmt-migrate" "-mt-migrate-directory" "{{[^"]*}}/foo/bar"

// RUN: touch %t.o
// RUN: %clang -ccc-arcmt-check -target i386-apple-darwin9 -### %t.o 2> %t.log
// RUN: FileCheck -check-prefix=LINK %s < %t.log
// RUN: %clang -ccc-arcmt-migrate /foo/bar -target i386-apple-darwin9 -### %t.o 2> %t.log
// RUN: FileCheck -check-prefix=LINK %s < %t.log

// LINK-NOT: {{ld(.exe)?"}}
// LINK: {{touch(.exe)?"}}
