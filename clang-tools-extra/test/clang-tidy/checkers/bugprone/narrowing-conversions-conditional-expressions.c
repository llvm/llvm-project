// RUN: %check_clang_tidy %s bugprone-narrowing-conversions %t -- --

char test(int cond, char c) {
	char ret = cond > 0 ? ':' : c;
	return ret;
}
