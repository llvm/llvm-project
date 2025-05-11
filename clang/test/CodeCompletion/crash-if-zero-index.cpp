// RUN: not %clang_cc1 -fsyntax-only -code-completion-at=%s:0:1 %s -o -
// RUN: not %clang_cc1 -fsyntax-only -code-completion-at=%s:1:0 %s -o -

// Related to #139375
// Clang uses 1-based indexing for source locations given from the command-line.
// Verify Clang doesn’t crash when 0 is given as line or column number.
