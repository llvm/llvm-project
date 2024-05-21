// RUN: %check_clang_tidy -std=c++98-or-later %s bugprone-tagged-union-member-count %t \
// RUN:   -config='{CheckOptions: { \
// RUN:     bugprone-tagged-union-member-count.StrictMode: 0, \
// RUN:     bugprone-tagged-union-member-count.EnumCounterHeuristicIsEnabled: 1, \
// RUN:     bugprone-tagged-union-member-count.EnumCounterSuffix: "size", \
// RUN:  }}' --

typedef union union3 {
	short *shorts;
	int *ints;
	float *floats;
} union3;

typedef union union4 {
	short *shorts;
	double *doubles;
	int *ints;
	float *floats;
} union4;

enum tag_with_counter_count {
	twc1,
	twc2,
	twc3,
	twc_count,
};

enum tag_with_counter_size {
	twc11,
	twc22,
	twc33,
	twc_size,
};

// The heuristic is configured with the "size" suffix so
// twc_count will not be considered a counting enum constant
struct taggedUnion1 { // No warning expected
	enum tag_with_counter_count tag;
	union union4 data;
};

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: Tagged union has more data members (4) than tags (3)! [bugprone-tagged-union-member-count]
struct taggedUnion2 { 
	enum tag_with_counter_size tag;
	union union4 data;
};

