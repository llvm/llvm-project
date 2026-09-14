// RUN: %check_clang_tidy %s readability-function-cognitive-complexity %t -- \
// RUN:   -- -target x86_64-unknown-linux-gnu

extern "C" int function_alias_target() { return 42; }
extern "C" int function_alias()
    __attribute__((alias("function_alias_target")));
