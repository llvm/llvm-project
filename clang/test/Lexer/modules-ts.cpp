// RUN: %clang_cc1 -fsyntax-only %s
// RUN: %clang_cc1 -fmodules-ts -DMODULES -fsyntax-only %s

#ifdef MODULES
#define MODULES_KEYWORD(NAME) _Static_assert(!__is_identifier(NAME), #NAME)
#else
#define MODULES_KEYWORD(NAME) _Static_assert(__is_identifier(NAME), #NAME)
#endif

MODULES_KEYWORD(import);
MODULES_KEYWORD(module);
