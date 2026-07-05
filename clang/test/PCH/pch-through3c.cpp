// RUN: %clang_cc1 -I %S -emit-pch \
// RUN:   -include Inputs/pch-through3c.h \
// RUN:   -pch-through-header=Inputs/pch-through3c.h -o %t.3c %s

// Checks that no warnings appear for this successful use.
// RUN: %clang_cc1 -verify -I %S -include-pch %t.3c \
// RUN:   -include Inputs/pch-through3c.h \
// RUN:   -pch-through-header=Inputs/pch-through3c.h \
// RUN:   %S/Inputs/pch-through-use3c.cpp 2>&1
