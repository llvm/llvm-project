// RUN: rm -rf %t
// RUN: %clang_cc1 %s -I %S/Inputs -index-store-path %t/idx
// RUN: %clang_cc1 %s -I %S/Inputs -index-store-path %t/idx -DUNDEF
// RUN: find %t/idx/*/records -name "macro.h*" | count 1
// RUN: find %t/idx/*/records -name "macro-only-guards.h*" | count 0
// RUN: find %t/idx/*/records -name "record-hash-macros-header*" | count 2

// Changing the macro location in the header should create not only a new record
// for the header, but also a new record for the main file since the location
// of user-defined macros is part of their USR.
// RUN: %clang_cc1 %s -I %S/Inputs -index-store-path %t/idx -DMACRO_1 -DUNDEF
// RUN: find %t/idx/*/records -name "macro.h*" | count 2
// RUN: find %t/idx/*/records -name "record-hash-macros-header*" | count 3

#include "macro.h"
#include "macro-only-guards.h"

#define MACRO_FROM_MAIN_FILE 2

#ifdef UNDEF
#undef MACRO_FROM_HEADER
#define MACRO_FROM_HEADER 3
#endif
