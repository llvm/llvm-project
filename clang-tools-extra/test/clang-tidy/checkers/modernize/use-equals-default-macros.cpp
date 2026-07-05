// RUN: %check_clang_tidy %s modernize-use-equals-default %t -- \
// RUN:   -config="{CheckOptions: {modernize-use-equals-default.IgnoreMacros: false}}"

#define STRUCT_WITH_DEFAULT(_base, _type) \
  struct _type {                          \
    _type() {}                            \
    _base value;                          \
  };

STRUCT_WITH_DEFAULT(unsigned char, InMacro)
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use '= default' to define a trivial default constructor
// CHECK-MESSAGES: :[[@LINE-6]]:13: note:
