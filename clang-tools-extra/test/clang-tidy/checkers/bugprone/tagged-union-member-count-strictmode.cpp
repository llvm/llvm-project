// RUN: %check_clang_tidy -std=c++98-or-later %s bugprone-tagged-union-member-count %t \
// RUN:   -config='{CheckOptions: { \
// RUN:     bugprone-tagged-union-member-count.StrictMode: 1, \
// RUN:     bugprone-tagged-union-member-count.EnumCounterHeuristicIsEnabled: 0, \
// RUN:     bugprone-tagged-union-member-count.EnumCounterSuffix: "count", \
// RUN:  }}' --

typedef enum tags3 {
	tags3_1,
	tags3_2,
	tags3_3,
} tags3;

typedef enum tags4 {
	tags4_1,
	tags4_2,
	tags4_3,
	tags4_4,
} tags4;

typedef enum tags5 {
	tags5_1,
	tags5_2,
	tags5_3,
	tags5_4,
	tags5_5,
} tags5;

enum class classtags3 {
	classtags3_1,
	classtags3_2,
	classtags3_3,
};

enum class classtags4 {
	classtags4_1,
	classtags4_2,
	classtags4_3,
	classtags4_4,
};

enum class classtags5 {
	classtags5_1,
	classtags5_2,
	classtags5_3,
	classtags5_4,
	classtags5_5,
};

enum class typedtags3 : unsigned int {
	typedtags3_1,
	typedtags3_2,
	typedtags3_3,
};

enum class typedtags4 : long {
	typedtags4_1,
	typedtags4_2,
	typedtags4_3,
	typedtags4_4,
};

enum class typedtags5 {
	typedtags5_1,
	typedtags5_2,
	typedtags5_3,
	typedtags5_4,
	typedtags5_5,
};

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

// Technically this means that every enum value is defined from 0-256 and therefore a warning is given.
enum mycolor {
	mycolor_black = 0x00,
	mycolor_gray  = 0xcc,
	mycolor_white = 0xff,
};

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: Tagged union has fewer data members (3) than tags (256)! [bugprone-tagged-union-member-count]
struct taggedunion9 { 
	enum mycolor tag;
	union {
		int a;
		float b;
		struct {
			double re;
			double im;
		} complex;
	} data;
};

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: Tagged union has fewer data members (3) than tags (4)! [bugprone-tagged-union-member-count]
struct withanonymous { 
	enum tags4 tag;
	union {
		int a;
		float b;
		char *c;
	};
}; 

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: Tagged union has fewer data members (3) than tags (4)! [bugprone-tagged-union-member-count]
struct withTypedef1 { 
	tags4 tag;
	union3 data;
}; 

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: Tagged union has fewer data members (3) than tags (4)! [bugprone-tagged-union-member-count]
struct withEnumClass1 {
	enum classtags4 tag;
	union3 data;
};

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: Tagged union has fewer data members (3) than tags (4)! [bugprone-tagged-union-member-count]
struct withTypedEnum2 {
	typedtags4 tag;
	union3 data;
};

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: Tagged union has fewer data members (2) than tags (3)! [bugprone-tagged-union-member-count]
struct anonymous2 {
	tags3 tag;
	union {
		int a;
		float f;
	};
};

// CHECK-MESSAGES: :[[@LINE+2]]:8: warning: Tagged union has fewer data members (3) than tags (5)! [bugprone-tagged-union-member-count]
template <typename Union, typename Tag>
struct templated {
	Tag tag;
	Union data;
};

templated<union3, tags3> t1; // No warning expected
templated<union3, tags5> t2;

