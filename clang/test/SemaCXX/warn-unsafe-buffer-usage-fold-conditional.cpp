// RUN: %clang_cc1 -fsyntax-only -Wno-all -Wunsafe-buffer-usage -verify=cxx %s -std=c++20
// RUN: %clang_cc1 -fsyntax-only -Wno-all -Wunsafe-buffer-usage -verify=c-only %s -x c
// c-only-no-diagnostics

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

// Test nested conditional expressions:
void testNested(char * message) {
  fprintf(stderr, "AssertMacros: %s", (0!=0) ? message : ((0!=0) ? message : ""));
}

// If the conditional cannot be constant-folded, try analyze both branches:
void testConditionalAnalysis(char * message, int x) {
  fprintf(stderr, "AssertMacros: %s", (x!=0) ? "hello" : "world");
  fprintf(stderr, "AssertMacros: %s", (0!=0) ? message : ((x!=0) ? "hello" : "world"));
  fprintf(stderr, "AssertMacros: %s", (x!=0) ? (((x!=0) ? "hello" : "world")) : ((x!=0) ? "hello" : "world"));
  fprintf(stderr, "AssertMacros: %s", (x!=0) ? (((x!=0) ? "hello" : "world")) : ((x!=0) ? "hello" : message));
  // cxx-warning@-1 {{function 'fprintf' is unsafe}}
  // cxx-note@-2 {{string argument is not guaranteed to be null-terminated}}											       
}

// Test that the analysis will not crash when a conditional expression
// appears in dependent context:
#ifdef __cplusplus
struct Foo {
  static void static_method(int);
};
void conditional_inside_dependent_context(void) {
  auto lambda = [](auto result) { // opens a dependent context
    Foo::static_method(result ? 1 : 2); // no-crash
  };
  (void)lambda;
}
#endif // __cplusplus
