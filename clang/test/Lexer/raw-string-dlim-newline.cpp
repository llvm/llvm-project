// RUN: not  %clang_cc1 -E %s 2>&1 | grep 'error: invalid newline character in raw string delimiter; use PREFIX( )PREFIX to delimit raw string'

// Introduced new error code err_invalid_nexline_raw_delim for code which has \n as delimiter.
char const* str1 = R"
";
