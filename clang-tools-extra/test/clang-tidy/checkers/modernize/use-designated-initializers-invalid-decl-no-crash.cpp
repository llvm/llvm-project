// RUN: %check_clang_tidy -std=c++20-or-later -expect-clang-tidy-error %s modernize-use-designated-initializers %t -- -header-filter=.*

struct S1 {int a1;};

struct S2 : S1 {
	S2(const S1& a);
};

struct S3 {
	S1 a;
	S2 b;
};

struct S4 {
	S3 c;
};

S4 s41{0, {0}};
// CHECK-MESSAGES: :[[@LINE-1]]:11: error: no matching constructor for initialization of 'S2' [clang-diagnostic-error]
