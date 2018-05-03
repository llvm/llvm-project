// RUN: clang-tidy %s -checks=-*,modernize-use-nullptr -- | count 0

#if !defined(__clang_analyzer__)
#error __clang_analyzer__ is not defined
#endif
