// RUN: %check_clang_tidy -std=c++98-or-later %s bugprone-tagged-union-member-count %t
// Test cases for the default configuration

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

// It is not obvious which enum is the tag for the union.
struct taggedunion2 { // No warnings expected.
	enum tags3 tagA;
	enum tags4 tagB;
	union union4 data;
};

// It is not obvious which union does the tag belong to.
struct taggedunion4 { // No warnings expected.
	enum tags3 tag;
	union union3 dataB;
	union union3 dataA;
};

// It is not obvious which union does the tag belong to.
struct taggedunion6 { // No warnings expected.
	enum tags3 tag;
	union {
		int i1;
		int i2;
		int i3;
	};
	union {
		float f1;
		float f2;
		float f3;
	};
};

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: Tagged union has more data members (4) than tags (3)! [bugprone-tagged-union-member-count]
struct taggedunion1 {
	enum tags3 tag;
    union union4 data;
};

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: Tagged union has more data members (4) than tags (3)! [bugprone-tagged-union-member-count]
struct taggedunion5 {
	enum tags3 tag;
    union {
		int *ints;
		char characters[13];
		struct {
			double re;
			double im;
		} complex;
		long l;
    } data;
};

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: Tagged union has more data members (4) than tags (3)! [bugprone-tagged-union-member-count]
struct taggedunion7 { 
	enum {
		tag1,
		tag2,
		tag3,
	} tag;
	union union4 data;
};

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: Tagged union has more data members (4) than tags (3)! [bugprone-tagged-union-member-count]
struct taggedunion8 { 
	enum {
		tag1,
		tag2,
		tag3,
	} tag;
	union {
		int *ints;
		char characters[13];
		struct {
			double re;
			double im;
		} complex;
		long l;
	} data;
};

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: Tagged union has more data members (4) than tags (3)! [bugprone-tagged-union-member-count]
struct nested4 { 
	enum tags3 tag;
	union {
		float f;
		int i;
		long l;
		// CHECK-MESSAGES: :[[@LINE+1]]:10: warning: Tagged union has more data members (4) than tags (3)! [bugprone-tagged-union-member-count]
		struct innerdecl { 
			enum tags3 tag;
			union union4 data;
		} inner; 
	} data;
};

enum tag_with_counter_lowercase {
	node_type_loop,
	node_type_branch,
	node_type_function,
	node_type_count,
};

enum tag_with_counter_uppercase {
	NODE_TYPE_LOOP,
	NODE_TYPE_BRANCH,
	NODE_TYPE_FUNCTION,
	NODE_TYPE_COUNT,
};

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: Tagged union has more data members (4) than tags (3)! [bugprone-tagged-union-member-count]
struct taggedunion10 { 
	enum tag_with_counter_lowercase tag;
	union union4 data;
};

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: Tagged union has more data members (4) than tags (3)! [bugprone-tagged-union-member-count]
struct taggedunion11 { 
	enum tag_with_counter_uppercase tag;
	union union4 data;
};

// Without the counter enum constant the number of tags
// and the number data members are equal.
struct taggedunion12 { // No warnings expected.
	enum tag_with_counter_uppercase tag;
	union union3 data;
};

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: Tagged union has more data members (4) than tags (3)! [bugprone-tagged-union-member-count]
struct withTypedef2 { 
	tags3 tag;
	union4 data;
}; 

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: Tagged union has more data members (4) than tags (3)! [bugprone-tagged-union-member-count]
struct withEnumClass2 { 
	enum classtags3 tag;
	union4 data;
}; 

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: Tagged union has more data members (4) than tags (3)! [bugprone-tagged-union-member-count]
struct withTypedEnum1 {
	typedtags3 tag;
	union4 data;
};

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: Tagged union has more data members (4) than tags (3)! [bugprone-tagged-union-member-count]
struct anonymous1 {
	tags3 tag;
	union {
		int a;
		int b;
		int c;
		int d;
	};
};

// CHECK-MESSAGES: :[[@LINE+2]]:8: warning: Tagged union has more data members (4) than tags (3)! [bugprone-tagged-union-member-count]
template <typename Union, typename Tag>
struct templated {
	Tag tag;
	Union data;
};

templated<union3, tags3> t1; // No warning expected
templated<union4, tags3> t3;

