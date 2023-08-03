// RUN: not c-index-test core --scan-deps %S -output-dir=%t -- \
// RUN:   %clang -c %s -o %t/t.o 2> %t.err.txt
// RUN: FileCheck -input-file=%t.err.txt %s

// CHECK: [[@LINE+1]]:10: fatal error: 'not-existent.h' file not found
#include "not-existent.h"
