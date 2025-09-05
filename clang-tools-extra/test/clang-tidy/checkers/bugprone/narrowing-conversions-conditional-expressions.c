// RUN: %check_clang_tidy %s bugprone-narrowing-conversions %t -- \
// RUN: -- -target x86_64-unknown-linux

char test_char(int cond, char c) {
	char ret = cond > 0 ? ':' : c;
	return ret;
}

short test_short(int cond, short s) {
	short ret = cond > 0 ? ':' : s;
	return ret;
}

int test_int(int cond, int i) {
	int ret = cond > 0 ? ':' : i;
	return ret;
}

void test(int cond, int i) {
  char x = cond > 0 ? ':' : i;
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: narrowing conversion from 'int' to signed type 'char' is implementation-defined [bugprone-narrowing-conversions]
}
