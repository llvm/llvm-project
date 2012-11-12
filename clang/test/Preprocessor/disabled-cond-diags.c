// RUN: %clang_cc1 -E %s 2>&1 | not grep "warning\|error"

#if 0

// Shouldn't get warnings here.
??( ??)

// Should not get an error here.
` ` ` `
#endif
