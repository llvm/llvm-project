// RUN: rm -rf %t
// RUN: %clang_cc1 %s -index-store-path %t/idx
// RUN: %clang_cc1 %s -index-store-path %t/idx -DMACRO_1
// RUN: find %t/idx/*/records -name "record-hash-macros*" | count 2

#ifdef MACRO_1
#define MACRO "hi"
#else
#define MACRO "hi"
#endif
