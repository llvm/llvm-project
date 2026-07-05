// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify %s

// There should still be diagnostics within included files.
#include "source1.cpp"
#include "source2.cpp"
