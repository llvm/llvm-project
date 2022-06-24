// Check that clang is able to process response files with extra whitespace.
// We generate a dos-style file with \r\n for line endings, and then split
// some joined arguments (like "-x c") across lines to ensure that regular
// clang (not clang-cl) can process it correctly.

// RUN: printf " -x\r\nc\r\n-DTEST\r\n" > %t.0.txt
// RUN: %clang -E @%t.0.txt %s -v 2>&1 | FileCheck %s -check-prefix=SHORT
// SHORT: extern int it_works;

#ifdef TEST
extern int it_works;
#endif
