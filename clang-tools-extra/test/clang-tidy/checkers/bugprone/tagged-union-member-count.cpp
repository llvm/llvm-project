// RUN: %check_clang_tidy %s bugprone-tagged-union-member-count %t

enum tags3 {
	tags3_1,
	tags3_2,
	tags3_3,
};

enum tags4 {
	tags4_1,
	tags4_2,
	tags4_3,
	tags4_4,
};

enum tags5 {
	tags5_1,
	tags5_2,
	tags5_3,
	tags5_4,
	tags5_5,
};

union union3 {
	short *shorts;
	int *ints;
	float *floats;
};

union union4 {
	short *shorts;
	double *doubles;
	int *ints;
	float *floats;
};

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

struct taggedunion1 { // CHECK-MESSAGES: :[[@LINE]]:8: warning: Tagged union has more data members than tags! Data members: 4 Tags: 3 [bugprone-tagged-union-member-count]
	enum tags3 tag;
    union union4 data;
};

struct taggedunion5 { // CHECK-MESSAGES: :[[@LINE]]:8: warning: Tagged union has more data members than tags! Data members: 4 Tags: 3 [bugprone-tagged-union-member-count]
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

struct taggedunion7 { // CHECK-MESSAGES: :[[@LINE]]:8: warning: Tagged union has more data members than tags! Data members: 4 Tags: 3 [bugprone-tagged-union-member-count]
	enum {
		tag1,
		tag2,
		tag3,
	} tag;
	union union4 data;
};

struct taggedunion8 { // CHECK-MESSAGES: :[[@LINE]]:8: warning: Tagged union has more data members than tags! Data members: 4 Tags: 3 [bugprone-tagged-union-member-count]
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

struct nested1 { // CHECK-MESSAGES: :[[@LINE]]:8: warning: Tagged union has more data members than tags! Data members: 4 Tags: 3 [bugprone-tagged-union-member-count]
	enum tags3 tag;
	union {
		char c;
		short s;
		int i;
		struct { // CHECK-MESSAGES: :[[@LINE]]:3: warning: Tagged union has fewer data members than tags! Data members: 4 Tags: 5 [bugprone-tagged-union-member-count]
			enum tags5 tag;
			union union4 data;
		} inner;
	} data;
};

struct nested2 {
	enum tags3 tag;
	union {
		float f;
		int i;
		struct { // CHECK-MESSAGES: :[[@LINE]]:3: warning: Tagged union has more data members than tags! Data members: 4 Tags: 3 [bugprone-tagged-union-member-count]
			enum tags3 tag;
			union union4 data;
		} inner;
	} data;
};

struct nested3 { // CHECK-MESSAGES: :[[@LINE]]:8: warning: Tagged union has fewer data members than tags! Data members: 2 Tags: 3 [bugprone-tagged-union-member-count]
	enum tags3 tag;
	union {
		float f;
		int i;
		struct innerdecl { // CHECK-MESSAGES: :[[@LINE]]:10: warning: Tagged union has more data members than tags! Data members: 4 Tags: 3 [bugprone-tagged-union-member-count]
			enum tags3 tag;
			union union4 data;
		}; 
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

struct taggedunion10 { // CHECK-MESSAGES: :[[@LINE]]:8: warning: Tagged union has more data members than tags! Data members: 4 Tags: 3 [bugprone-tagged-union-member-count]
	enum tag_with_counter_lowercase tag;
	union union4 data;
};

struct taggedunion11 { // CHECK-MESSAGES: :[[@LINE]]:8: warning: Tagged union has more data members than tags! Data members: 4 Tags: 3 [bugprone-tagged-union-member-count]
	enum tag_with_counter_uppercase tag;
	union union4 data;
};

// Without the counter enum constant the number of tags
// and the number data members are equal.
struct taggedunion12 { // No warnings expected.
	enum tag_with_counter_uppercase tag;
	union union3 data;
};

// Technically this means that every enum value is defined from 0-256 and therefore a warning is given.
enum mycolor {
	mycolor_black = 0x00,
	mycolor_gray  = 0xcc,
	mycolor_white = 0xff,
};

struct taggedunion9 { // CHECK-MESSAGES: :[[@LINE]]:8: warning: Tagged union has fewer data members than tags! Data members: 3 Tags: 256 [bugprone-tagged-union-member-count]
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

