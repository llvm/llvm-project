// RUN: touch %t.o
// RUN: %clang -target arm64-apple-darwin13 -### \
// RUN: -reexport_framework Foo -reexport-lBar -reexport_library Baz %t.o 2> %t.log

// Check older spellings also work.
// RUN: %clang -target arm64-apple-darwin13 -### \
// RUN: -Xlinker -reexport_framework -Xlinker Forest \
// RUN: -Xlinker -reexport-lBranch \
// RUN: -Xlinker -reexport_library -Xlinker Flower %t.o 2>> %t.log
// RUN: FileCheck -check-prefix=LINK_REEXPORT %s < %t.log

// LINK_REEXPORT: {{ld(.exe)?"}}
// LINK_REEXPORT: "-reexport_framework" "Foo"
// LINK_REEXPORT: "-reexport-lBar"
// LINK_REEXPORT: "-reexport_library" "Baz"
// LINK_REEXPORT: "-reexport_framework" "Forest"
// LINK_REEXPORT: "-reexport-lBranch"
// LINK_REEXPORT: "-reexport_library" "Flower"

// Make sure arguments are not repeated.
// LINK_REEXPORT-NOT: "-reexport
