// RUN: %clang --autocomplete=-fsyn | FileCheck %s -check-prefix=FSYN
// FSYN: -fsyntax-only
// RUN: %clang --autocomplete=-s | FileCheck %s -check-prefix=STD
// STD: -std={{.*}}-stdlib=
// RUN: %clang --autocomplete=foo | not FileCheck %s -check-prefix=NONE
// NONE: foo
