// RUN: %clang_cc1 -fsyntax-only -Wno-all -Wunsafe-buffer-usage -verify %s -std=c++20
// RUN: %clang_cc1 -fsyntax-only -Wno-all -Wunsafe-buffer-usage -verify %s -x c
// expected-no-diagnostics

typedef struct {} FILE;
int fprintf( FILE* stream, const char* format, ... );
FILE * stderr;

#define DEBUG_ASSERT_MESSAGE(name, assertion, label, message, file, line, value) \
  fprintf(stderr, "AssertMacros: %s, %s file: %s, line: %d, value: %lld\n",      \
          assertion, (message!=0) ? message : "", file, line, (long long) (value));


#define Require(assertion, exceptionLabel)                              \
  do                                                                    \
    {                                                                   \
      if ( __builtin_expect(!(assertion), 0) ) {                        \
        DEBUG_ASSERT_MESSAGE(                                           \
	  "DEBUG_ASSERT_COMPONENT_NAME_STRING",                         \
	  #assertion, #exceptionLabel, 0, __FILE__, __LINE__,  0);      \
	goto exceptionLabel;                                            \
      }									\
    } while ( 0 )


void f(int x, int y) {
  Require(x == y, L1);
 L1:
  return;
}

