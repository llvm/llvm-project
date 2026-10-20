// RUN: %check_clang_tidy %s bugprone-tagged-union-member-count %t \
// RUN:   -config='{CheckOptions: { \
// RUN:       bugprone-tagged-union-member-count.EnableCountingEnumHeuristic: false, \
// RUN:       bugprone-tagged-union-member-count.CountingEnumSuffixes: "count", \
// RUN:       bugprone-tagged-union-member-count.CountingEnumPrefixes: "last", \
// RUN:   }}'

// Warn when the heuristic is disabled and a suffix or a prefix is set explicitly.

// CHECK-MESSAGES: warning: bugprone-tagged-union-member-count: Counting enum heuristic is disabled but CountingEnumPrefixes is set
// CHECK-MESSAGES: warning: bugprone-tagged-union-member-count: Counting enum heuristic is disabled but CountingEnumSuffixes is set
