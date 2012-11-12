// RUN: %clang -fno-ms-extensions %s -E | grep 'stddef.h.*3'
#include <stddef.h>
