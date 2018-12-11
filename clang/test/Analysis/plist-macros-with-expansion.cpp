// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify %s
//
// RUN: %clang_analyze_cc1 -analyzer-checker=core %s  \
// RUN:   -analyzer-output=plist -o %t.plist \
// RUN:   -analyzer-config expand-macros=true
//
// Check the actual plist output.
//   RUN: cat %t.plist | %diff_plist \
//   RUN:   %S/Inputs/expected-plists/plist-macros-with-expansion.cpp.plist
//
// Check the macro expansions from the plist output here, to make the test more
// understandable.
//   RUN: FileCheck --input-file=%t.plist %s

void print(const void*);

//===----------------------------------------------------------------------===//
// Tests for non-function-like macro expansions.
//===----------------------------------------------------------------------===//

#define SET_PTR_VAR_TO_NULL \
  ptr = 0

void nonFunctionLikeMacroTest() {
  int *ptr;
  SET_PTR_VAR_TO_NULL;
  *ptr = 5; // expected-warning{{Dereference of null pointer}}
}

// CHECK: <key>name</key><string>SET_PTR_VAR_TO_NULL</string>
// CHECK-NEXT: <key>expansion</key><string>ptr = 0</string>

#define NULL 0
#define SET_PTR_VAR_TO_NULL_WITH_NESTED_MACRO \
  ptr = NULL

void nonFunctionLikeNestedMacroTest() {
  int *ptr;
  SET_PTR_VAR_TO_NULL_WITH_NESTED_MACRO;
  *ptr = 5; // expected-warning{{Dereference of null pointer}}
}

// CHECK: <key>name</key><string>SET_PTR_VAR_TO_NULL_WITH_NESTED_MACRO</string>
// CHECK-NEXT: <key>expansion</key><string>ptr =0</string>

//===----------------------------------------------------------------------===//
// Tests for function-like macro expansions.
//===----------------------------------------------------------------------===//

void setToNull(int **vptr) {
  *vptr = nullptr;
}

#define TO_NULL(x) \
  setToNull(x)

void functionLikeMacroTest() {
  int *ptr;
  TO_NULL(&ptr);
  *ptr = 5; // expected-warning{{Dereference of null pointer}}
}

// CHECK: <key>name</key><string>TO_NULL</string>
// CHECK-NEXT: <key>expansion</key><string>setToNull(&amp;ptr)</string>

#define DOES_NOTHING(x) \
  {                     \
    int b;              \
    b = 5;              \
  }                     \
  print(x)

#define DEREF(x)   \
  DOES_NOTHING(x); \
  *x

void functionLikeNestedMacroTest() {
  int *a;
  TO_NULL(&a);
  DEREF(a) = 5; // expected-warning{{Dereference of null pointer}}
}

// CHECK: <key>name</key><string>TO_NULL</string>
// CHECK-NEXT: <key>expansion</key><string>setToNull(&amp;a)</string>

// CHECK: <key>name</key><string>DEREF</string>
// CHECK-NEXT: <key>expansion</key><string>{ int b; b = 5; } print(a); *a</string>

//===----------------------------------------------------------------------===//
// Tests for undefining and/or redifining macros.
//===----------------------------------------------------------------------===//

#define WILL_UNDEF_SET_NULL_TO_PTR(ptr) \
  ptr = nullptr;

void undefinedMacroByTheEndOfParsingTest() {
  int *ptr;
  WILL_UNDEF_SET_NULL_TO_PTR(ptr);
  *ptr = 5; // expected-warning{{Dereference of null pointer}}
}

#undef WILL_UNDEF_SET_NULL_TO_PTR

// CHECK: <key>name</key><string>WILL_UNDEF_SET_NULL_TO_PTR</string>
// CHECK-NEXT: <key>expansion</key><string>ptr = nullptr;</string>

#define WILL_REDIFINE_MULTIPLE_TIMES_SET_TO_NULL(ptr) \
  /* Nothing */
#undef WILL_REDIFINE_MULTIPLE_TIMES_SET_TO_NULL
#define WILL_REDIFINE_MULTIPLE_TIMES_SET_TO_NULL(ptr) \
  ptr = nullptr;

void macroRedefinedMultipleTimesTest() {
  int *ptr;
  WILL_REDIFINE_MULTIPLE_TIMES_SET_TO_NULL(ptr)
  *ptr = 5; // expected-warning{{Dereference of null pointer}}
}

#undef WILL_REDIFINE_MULTIPLE_TIMES_SET_TO_NULL
#define WILL_REDIFINE_MULTIPLE_TIMES_SET_TO_NULL(ptr)                      \
  print("This string shouldn't be in the plist file at all. Or anywhere, " \
        "but here.");

// CHECK: <key>name</key><string>WILL_REDIFINE_MULTIPLE_TIMES_SET_TO_NULL</string>
// CHECK-NEXT: <key>expansion</key><string>ptr = nullptr;</string>

#define WILL_UNDEF_SET_NULL_TO_PTR_2(ptr) \
  ptr = nullptr;

#define PASS_PTR_TO_MACRO_THAT_WILL_BE_UNDEFD(ptr) \
  WILL_UNDEF_SET_NULL_TO_PTR_2(ptr)

void undefinedMacroInsideAnotherMacroTest() {
  int *ptr;
  PASS_PTR_TO_MACRO_THAT_WILL_BE_UNDEFD(ptr);
  *ptr = 5; // expected-warning{{Dereference of null pointer}}
}

// TODO: Expand arguments.
// CHECK: <key>name</key><string>PASS_PTR_TO_MACRO_THAT_WILL_BE_UNDEFD</string>
// CHECK-NEXT: <key>expansion</key><string>ptr = nullptr;</string>

#undef WILL_UNDEF_SET_NULL_TO_PTR_2

//===----------------------------------------------------------------------===//
// Tests for macro arguments containing commas and parantheses.
//
// As of writing these tests, the algorithm expands macro arguments by lexing
// the macro's expansion location, and relies on finding tok::comma and
// tok::l_paren/tok::r_paren.
//===----------------------------------------------------------------------===//

// Note that this commas, parantheses in strings aren't parsed as tok::comma or
// tok::l_paren/tok::r_paren, but why not test them.

#define TO_NULL_AND_PRINT(x, str) \
  x = 0; \
  print(str)

void macroArgContainsCommaInStringTest() {
  int *a;
  TO_NULL_AND_PRINT(a, "Will this , cause a crash?");
  *a = 5; // expected-warning{{Dereference of null pointer}}
}

// CHECK: <key>name</key><string>TO_NULL_AND_PRINT</string>
// CHECK-NEXT: <key>expansion</key><string>a = 0; print( &quot;Will this , cause a crash?&quot;)</string>

void macroArgContainsLParenInStringTest() {
  int *a;
  TO_NULL_AND_PRINT(a, "Will this ( cause a crash?");
  *a = 5; // expected-warning{{Dereference of null pointer}}
}

// CHECK: <key>name</key><string>TO_NULL_AND_PRINT</string>
// CHECK-NEXT: <key>expansion</key><string>a = 0; print( &quot;Will this ( cause a crash?&quot;)</string>

void macroArgContainsRParenInStringTest() {
  int *a;
  TO_NULL_AND_PRINT(a, "Will this ) cause a crash?");
  *a = 5; // expected-warning{{Dereference of null pointer}}
}

// CHECK: <key>name</key><string>TO_NULL_AND_PRINT</string>
// CHECK-NEXT: <key>expansion</key><string>a = 0; print( &quot;Will this ) cause a crash?&quot;)</string>

#define CALL_FUNCTION(funcCall)   \
  funcCall

// Function calls do contain both tok::comma and tok::l_paren/tok::r_paren.

void macroArgContainsLParenRParenTest() {
  int *a;
  CALL_FUNCTION(setToNull(&a));
  *a = 5; // expected-warning{{Dereference of null pointer}}
}

// CHECK: <key>name</key><string>CALL_FUNCTION</string>
// CHECK-NEXT: <key>expansion</key><string>setToNull(&amp;a)</string>

void setToNullAndPrint(int **vptr, const char *str) {
  setToNull(vptr);
  print(str);
}

void macroArgContainsCommaLParenRParenTest() {
  int *a;
  CALL_FUNCTION(setToNullAndPrint(&a, "Hello!"));
  *a = 5; // expected-warning{{Dereference of null pointer}}
}

// CHECK: <key>name</key><string>CALL_FUNCTION</string>
// CHECK-NEXT: <key>expansion</key><string>setToNullAndPrint(&amp;a, &quot;Hello!&quot;)</string>

#define CALL_FUNCTION_WITH_TWO_PARAMS(funcCall, param1, param2) \
  funcCall(param1, param2)

void macroArgContainsCommaLParenRParenTest2() {
  int *a;
  CALL_FUNCTION_WITH_TWO_PARAMS(setToNullAndPrint, &a, "Hello!");
  *a = 5; // expected-warning{{Dereference of null pointer}}
}

// CHECK: <key>name</key><string>CALL_FUNCTION_WITH_TWO_PARAMS</string>
// CHECK-NEXT: <key>expansion</key><string>setToNullAndPrint( &amp;a, &quot;Hello!&quot;)</string>

#define CALL_LAMBDA(l) \
  l()

void commaInBracketsTest() {
  int *ptr;
  const char str[] = "Hello!";
  // You need to add parantheses around a lambda expression to compile this,
  // else the comma in the capture will be parsed as divider of macro args.
  CALL_LAMBDA(([&ptr, str] () mutable { TO_NULL(&ptr); }));
  *ptr = 5; // expected-warning{{Dereference of null pointer}}
}

// CHECK: <key>name</key><string>CALL_LAMBDA</string>
// CHECK-NEXT: <key>expansion</key><string>([&amp;ptr, str] () mutable { setToNull(&amp;ptr); })()</string>

#define PASTE_CODE(code) \
  code

void commaInBracesTest() {
  PASTE_CODE({ // expected-warning{{Dereference of null pointer}}
    // NOTE: If we were to add a new variable here after a comma, we'd get a
    // compilation error, so this test is mainly here to show that this was also
    // investigated.

    // int *ptr = nullptr, a;
    int *ptr = nullptr;
    *ptr = 5;
  })
}

// CHECK: <key>name</key><string>PASTE_CODE</string>
// CHECK-NEXT: <key>expansion</key><string>{ int *ptr = nullptr; *ptr = 5; }</string>

// Example taken from
// https://gcc.gnu.org/onlinedocs/cpp/Macro-Arguments.html#Macro-Arguments.

#define POTENTIALLY_EMPTY_PARAM(x, y) \
  x;                                  \
  y = nullptr

void emptyParamTest() {
  int *ptr;

  POTENTIALLY_EMPTY_PARAM(,ptr);
  *ptr = 5; // expected-warning{{Dereference of null pointer}}
}

// CHECK: <key>name</key><string>POTENTIALLY_EMPTY_PARAM</string>
// CHECK-NEXT: <key>expansion</key><string>;ptr = nullptr</string>

#define NESTED_EMPTY_PARAM(a, b) \
  POTENTIALLY_EMPTY_PARAM(a, b);


void nestedEmptyParamTest() {
  int *ptr;

  NESTED_EMPTY_PARAM(, ptr);
  *ptr = 5; // expected-warning{{Dereference of null pointer}}
}

// CHECK: <key>name</key><string>NESTED_EMPTY_PARAM</string>
// CHECK-NEXT: <key>expansion</key><string>; ptr = nullptr;</string>

#define CALL_FUNCTION_WITH_ONE_PARAM_THROUGH_MACRO(func, param) \
  CALL_FUNCTION(func(param))

void lParenRParenInNestedMacro() {
  int *ptr;
  CALL_FUNCTION_WITH_ONE_PARAM_THROUGH_MACRO(setToNull, &ptr);
  *ptr = 5; // expected-warning{{Dereference of null pointer}}
}

// CHECK: <key>name</key><string>CALL_FUNCTION_WITH_ONE_PARAM_THROUGH_MACRO</string>
// CHECK-NEXT: <key>expansion</key><string>setToNull( &amp;ptr)</string>

//===----------------------------------------------------------------------===//
// Tests for variadic macro arguments.
//===----------------------------------------------------------------------===//

template <typename ...Args>
void variadicFunc(Args ...args);

#define VARIADIC_SET_TO_NULL(ptr, ...) \
  ptr = nullptr;                       \
  variadicFunc(__VA_ARGS__)

void variadicMacroArgumentTest() {
  int *ptr;
  VARIADIC_SET_TO_NULL(ptr, 1, 5, "haha!");
  *ptr = 5; // expected-warning{{Dereference of null pointer}}
}

// CHECK: <key>name</key><string>VARIADIC_SET_TO_NULL</string>
// CHECK-NEXT: <key>expansion</key><string>ptr = nullptr; variadicFunc( 1, 5, &quot;haha!&quot;)</string>

void variadicMacroArgumentWithoutAnyArgumentTest() {
  int *ptr;
  // Not adding a single parameter to ... is silly (and also causes a
  // preprocessor warning), but is not an excuse to crash on it.
  VARIADIC_SET_TO_NULL(ptr);
  *ptr = 5; // expected-warning{{Dereference of null pointer}}
}

// CHECK: <key>name</key><string>VARIADIC_SET_TO_NULL</string>
// CHECK-NEXT: <key>expansion</key><string>ptr = nullptr; variadicFunc()</string>

//===----------------------------------------------------------------------===//
// Tests for # and ##.
//===----------------------------------------------------------------------===//

#define DECLARE_FUNC_AND_SET_TO_NULL(funcName, ptr) \
  void generated_##funcName();                      \
  ptr = nullptr;

void hashHashOperatorTest() {
  int *ptr;
  DECLARE_FUNC_AND_SET_TO_NULL(whatever, ptr);
  *ptr = 5; // expected-warning{{Dereference of null pointer}}
}

// CHECK: <key>name</key><string>DECLARE_FUNC_AND_SET_TO_NULL</string>
// CHECK-NEXT: <key>expansion</key><string>void generated_whatever(); ptr = nullptr;</string>

void macroArgContainsHashHashInStringTest() {
  int *a;
  TO_NULL_AND_PRINT(a, "Will this ## cause a crash?");
  *a = 5; // expected-warning{{Dereference of null pointer}}
}

// CHECK: <key>name</key><string>TO_NULL_AND_PRINT</string>
// CHECK-NEXT: <key>expansion</key><string>a = 0; print( &quot;Will this ## cause a crash?&quot;)</string>

#define PRINT_STR(str, ptr) \
  print(#str);              \
  ptr = nullptr

void hashOperatorTest() {
  int *ptr;
  PRINT_STR(Hello, ptr);
  *ptr = 5; // expected-warning{{Dereference of null pointer}}
}

// CHECK: <key>name</key><string>PRINT_STR</string>
// CHECK-NEXT: <key>expansion</key><string>print(&quot;Hello&quot;); ptr = nullptr</string>

void macroArgContainsHashInStringTest() {
  int *a;
  TO_NULL_AND_PRINT(a, "Will this # cause a crash?");
  *a = 5; // expected-warning{{Dereference of null pointer}}
}

// CHECK: <key>name</key><string>TO_NULL_AND_PRINT</string>
// CHECK-NEXT: <key>expansion</key><string>a = 0; print( &quot;Will this # cause a crash?&quot;)</string>

//===----------------------------------------------------------------------===//
// Tests for more complex macro expansions.
//
// We won't cover anything that wasn't covered up to this point, but rather
// show more complex, macros with deeper nesting, more arguments (some unused)
// and so on.
//===----------------------------------------------------------------------===//

#define IF(Condition) \
  if ( Condition )

#define L_BRACE {
#define R_BRACE }
#define LESS <
#define GREATER >
#define EQUALS =
#define SEMICOLON ;
#define NEGATIVE -
#define RETURN return
#define ZERO 0

#define EUCLIDEAN_ALGORITHM(A, B)                                              \
  IF(A LESS ZERO) L_BRACE                                                      \
    A EQUALS NEGATIVE A SEMICOLON                                              \
  R_BRACE                                                                      \
  IF(B LESS ZERO) L_BRACE                                                      \
    B EQUALS NEGATIVE B SEMICOLON                                              \
  R_BRACE                                                                      \
                                                                               \
  /* This is where a while loop would be, but that seems to be too complex */  \
  /* for the analyzer just yet. Let's just pretend that this algorithm     */  \
  /* works.                                                                */  \
                                                                               \
  RETURN B / (B - B) SEMICOLON

int getLowestCommonDenominator(int A, int B) {
  EUCLIDEAN_ALGORITHM(A, B) // expected-warning{{Division by zero}}
}

void testVeryComplexAlgorithm() {
  int tmp = 8 / (getLowestCommonDenominator(5, 7) - 1);
  print(&tmp);
}
// CHECK: <key>name</key><string>EUCLIDEAN_ALGORITHM</string>
// CHECK-NEXT: <key>expansion</key><string>if (A&lt;0 ){A=-A;} if ( B&lt;0 ){ B=- B;}return B / ( B - B);</string>

#define YET_ANOTHER_SET_TO_NULL(x, y, z)   \
  print((void *) x);                       \
  print((void *) y);                       \
  z = nullptr;

#define DO_NOTHING(str) str
#define DO_NOTHING2(str2) DO_NOTHING(str2)

void test() {
  int *ptr;
  YET_ANOTHER_SET_TO_NULL(5, DO_NOTHING2("Remember the Vasa"), ptr);
  *ptr = 5; // expected-warning{{Dereference of null pointer}}
}
// CHECK: <key>name</key><string>YET_ANOTHER_SET_TO_NULL</string>
// CHECK-NEXT: <key>expansion</key><string>print((void *)5); print((void *)&quot;Remember the Vasa&quot;); ptr = nullptr;</string>
