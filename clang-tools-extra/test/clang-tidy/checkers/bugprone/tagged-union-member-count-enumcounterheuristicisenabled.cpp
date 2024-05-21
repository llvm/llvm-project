// RUN: %check_clang_tidy -std=c++98-or-later %s bugprone-tagged-union-member-count %t \
// RUN:   -config='{CheckOptions: { \
// RUN:     bugprone-tagged-union-member-count.StrictMode: 1, \
// RUN:     bugprone-tagged-union-member-count.EnumCounterHeuristicIsEnabled: 0, \
// RUN:     bugprone-tagged-union-member-count.EnumCounterSuffix: "count", \
// RUN:  }}' --

// Without the heuristic the tags and the data members match
struct taggedUnion1 { // No warnings expected
	enum {
		tags1,
		tags2,
		tags3,
		tags_count,
	} tags;
	union {
		int a;
		int b;
		int c;
		int d;
	} data;
};

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: Tagged union has fewer data members (3) than tags (4)! [bugprone-tagged-union-member-count]
struct taggedUnion2 {
	enum {
		tags11,
		tags22,
		tags33,
		tags_count,
	} tags;
	union {
		int a;
		int b;
		int c;
	} data;
};

