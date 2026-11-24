// clang/test/Interpreter/delayed-template-parsing.cpp
// REQUIRES: host-supports-jit
// RUN: cat %s | clang-repl -Xcc -fdelayed-template-parsing 2>&1
// see ISSUE 217073 and PR 218571

namespace GH217073{ template <typename T> int f(T) { return 0; } }
GH217073::f(1);
