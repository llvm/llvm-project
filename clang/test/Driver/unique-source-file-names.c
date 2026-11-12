// RUN: %clang -funique-source-file-names -### %s 2> %t
// RUN: FileCheck --check-prefix=SRC < %t %s

// SRC: "-cc1"
// SRC: "-funique-source-file-identifier={{.*}}unique-source-file-names.c"

// RUN: %clang -funique-source-file-names -funique-source-file-identifier=foo -### %s 2> %t
// RUN: FileCheck --check-prefix=ID < %t %s

// ID: "-cc1"
// ID: "-funique-source-file-identifier=foo"
