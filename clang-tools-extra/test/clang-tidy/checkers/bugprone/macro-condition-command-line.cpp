// RUN: %check_clang_tidy -check-suffix=DEFINED %s \
// RUN:   bugprone-macro-condition %t -- -- -DCOMMAND_LINE_MACRO=0
// RUN: clang-tidy %s -checks=-*,bugprone-macro-condition -- \
// RUN:   -UCOMMAND_LINE_MACRO | count 0

// With -UCOMMAND_LINE_MACRO, these conditions are equivalent to:
//
// #undef COMMAND_LINE_MACRO

// With -DCOMMAND_LINE_MACRO=0, they are equivalent to:
//
// #define COMMAND_LINE_MACRO 0
#ifdef COMMAND_LINE_MACRO
#endif

#if COMMAND_LINE_MACRO
// CHECK-MESSAGES-DEFINED: :[[@LINE-1]]:5: warning: Macro 'COMMAND_LINE_MACRO' checked here for value after being checked for definition
// CHECK-MESSAGES-DEFINED: :[[@LINE-5]]:2: note: Macro 'COMMAND_LINE_MACRO' first checked here for definition
#endif
