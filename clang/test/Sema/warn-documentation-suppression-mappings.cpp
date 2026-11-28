// RUN: %clang_cc1 -fsyntax-only -Wdocumentation -verify=mapped \
// RUN:   --warning-suppression-mappings=%S/Inputs/documentation-suppression-mapping.txt %s
// RUN: %clang_cc1 -fsyntax-only -Wdocumentation -verify=unrelated,mapped \
// RUN:   --warning-suppression-mappings=%S/Inputs/documentation-suppression-mapping-unrelated.txt %s

// Whether a documentation comment is checked is cached per diagnostic state.
// But a suppression mapping for documentation diagnostics makes the result
// file path dependent. In that case, the cache must not be used. Test that.

/// \param x The x.
void doc1(int y); // unrelated-warning@-1 {{parameter 'x' not found in the function declaration}} \
                  // unrelated-note@-1 {{did you mean 'y'?}}

#line 100 "foo/bar.h"
/// \param x The x.
void doc2(int y); // mapped-warning@-1 {{parameter 'x' not found in the function declaration}} \
                  // mapped-note@-1 {{did you mean 'y'?}}
