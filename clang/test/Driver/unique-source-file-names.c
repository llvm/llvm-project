// RUN: %clang -funique-source-file-names -### %s 2> %t
// RUN: FileCheck --check-prefix=SRC < %t %s

// SRC: "-cc1"
// SRC: "-funique-source-file-identifier={{.*}}unique-source-file-names.c"

// RUN: %clang -funique-source-file-names -funique-source-file-identifier=foo -### %s 2> %t
// RUN: FileCheck --check-prefix=ID < %t %s

// ID: "-cc1"
// ID: "-funique-source-file-identifier=foo"

// RUN: %clang -funique-source-file-names -funique-source-file-output-paths -o out.o -c -### %s 2> %t
// RUN: FileCheck --check-prefix=OUTPUT < %t %s

// OUTPUT: "-cc1"
// OUTPUT: "-funique-source-file-identifier=out.o"
