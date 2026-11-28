// Test header for __has_include_next with absolute path
// When this header is found via absolute path (not through search directories),
// __has_include_next should return false instead of searching from the start
// of the include path.

#if __has_include_next(<nonexistent_header.h>)
#error "__has_include_next should return false for nonexistent header"
#endif

#define TEST_HEADER_INCLUDED 1
