// RUN: clang-tidy %s -checks=-*,bugprone-macro-condition -- -I %S | count 0

#define CROSS_FILE_MACRO 1
#include "Inputs/macro-condition-cross-file.h"

#if CROSS_FILE_MACRO
#endif

