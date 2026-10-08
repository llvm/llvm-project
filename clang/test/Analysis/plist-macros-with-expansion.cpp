// RUN: %clang_analyze_cc1 -std=c++14 -analyzer-checker=core %s  \
// RUN:   -analyzer-output=plist -o %t.plist \
// RUN:   -analyzer-config expand-macros=true -verify
//
// RUN: FileCheck --input-file=%t.plist %s

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

// CHECK:      <key>macro_expansions</key>
// CHECK-NEXT: <array>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>18</integer>
// CHECK-NEXT:    <key>col</key><integer>3</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <key>name</key><string>SET_PTR_VAR_TO_NULL</string>
// CHECK-NEXT:   <key>expansion</key><string>ptr =0</string>
// CHECK-NEXT:  </dict>
// CHECK-NEXT: </array>

#define NULL 0
#define SET_PTR_VAR_TO_NULL_WITH_NESTED_MACRO \
  ptr = NULL

void nonFunctionLikeNestedMacroTest() {
  int *ptr;
  SET_PTR_VAR_TO_NULL_WITH_NESTED_MACRO;
  *ptr = 5; // expected-warning{{Dereference of null pointer}}
}

// CHECK:      <key>macro_expansions</key>
// CHECK-NEXT: <array>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>42</integer>
// CHECK-NEXT:    <key>col</key><integer>3</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:  <key>name</key><string>SET_PTR_VAR_TO_NULL_WITH_NESTED_MACRO</string>
// CHECK-NEXT:  <key>expansion</key><string>ptr =0</string>
// CHECK-NEXT:  </dict>
// CHECK-NEXT: </array>

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

// CHECK:      <key>macro_expansions</key>
// CHECK-NEXT: <array>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>73</integer>
// CHECK-NEXT:    <key>col</key><integer>3</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:  <key>name</key><string>TO_NULL(&amp;ptr)</string>
// CHECK-NEXT:  <key>expansion</key><string>setToNull (&amp;ptr )</string>
// CHECK-NEXT:  </dict>
// CHECK-NEXT: </array>

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

// CHECK:      <key>macro_expansions</key>
// CHECK-NEXT: <array>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>104</integer>
// CHECK-NEXT:    <key>col</key><integer>3</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <key>name</key><string>TO_NULL(&amp;a)</string>
// CHECK-NEXT:   <key>expansion</key><string>setToNull (&amp;a )</string>
// CHECK-NEXT:  </dict>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>105</integer>
// CHECK-NEXT:    <key>col</key><integer>3</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <key>name</key><string>DEREF(a)</string>
// CHECK-NEXT:   <key>expansion</key><string>{int b ;b =5;}print (a );*a </string>
// CHECK-NEXT:  </dict>
// CHECK-NEXT: </array>

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

// CHECK:      <key>macro_expansions</key>
// CHECK-NEXT: <array>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>141</integer>
// CHECK-NEXT:    <key>col</key><integer>3</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <key>name</key><string>WILL_UNDEF_SET_NULL_TO_PTR(ptr)</string>
// CHECK-NEXT:   <key>expansion</key><string>ptr =nullptr ;</string>
// CHECK-NEXT:  </dict>
// CHECK-NEXT: </array>

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

// CHECK:      <key>macro_expansions</key>
// CHECK-NEXT: <array>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>169</integer>
// CHECK-NEXT:    <key>col</key><integer>3</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <key>name</key><string>WILL_REDIFINE_MULTIPLE_TIMES_SET_TO_NULL(ptr)</string>
// CHECK-NEXT:   <key>expansion</key><string>ptr =nullptr ;</string>
// CHECK-NEXT:  </dict>
// CHECK-NEXT: </array>

#define WILL_UNDEF_SET_NULL_TO_PTR_2(ptr) \
  ptr = nullptr;

#define PASS_PTR_TO_MACRO_THAT_WILL_BE_UNDEFD(ptr) \
  WILL_UNDEF_SET_NULL_TO_PTR_2(ptr)

void undefinedMacroInsideAnotherMacroTest() {
  int *ptr;
  PASS_PTR_TO_MACRO_THAT_WILL_BE_UNDEFD(ptr);
  *ptr = 5; // expected-warning{{Dereference of null pointer}}
}

// CHECK:      <key>macro_expansions</key>
// CHECK-NEXT: <array>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>200</integer>
// CHECK-NEXT:    <key>col</key><integer>3</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <key>name</key><string>PASS_PTR_TO_MACRO_THAT_WILL_BE_UNDEFD(ptr)</string>
// CHECK-NEXT:   <key>expansion</key><string>ptr =nullptr ;</string>
// CHECK-NEXT:  </dict>
// CHECK-NEXT: </array>

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

// CHECK:      <key>macro_expansions</key>
// CHECK-NEXT: <array>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>237</integer>
// CHECK-NEXT:    <key>col</key><integer>3</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <key>name</key><string>TO_NULL_AND_PRINT(a, &quot;Will this , cause a crash?&quot;)</string>
// CHECK-NEXT:   <key>expansion</key><string>a =0;print (&quot;Will this , cause a crash?&quot;)</string>
// CHECK-NEXT:  </dict>
// CHECK-NEXT: </array>

void macroArgContainsLParenInStringTest() {
  int *a;
  TO_NULL_AND_PRINT(a, "Will this ( cause a crash?");
  *a = 5; // expected-warning{{Dereference of null pointer}}
}

// CHECK:      <key>macro_expansions</key>
// CHECK-NEXT: <array>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>257</integer>
// CHECK-NEXT:    <key>col</key><integer>3</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <key>name</key><string>TO_NULL_AND_PRINT(a, &quot;Will this ( cause a crash?&quot;)</string>
// CHECK-NEXT:   <key>expansion</key><string>a =0;print (&quot;Will this ( cause a crash?&quot;)</string>
// CHECK-NEXT:  </dict>
// CHECK-NEXT: </array>

void macroArgContainsRParenInStringTest() {
  int *a;
  TO_NULL_AND_PRINT(a, "Will this ) cause a crash?");
  *a = 5; // expected-warning{{Dereference of null pointer}}
}

// CHECK:      <key>macro_expansions</key>
// CHECK-NEXT: <array>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>277</integer>
// CHECK-NEXT:    <key>col</key><integer>3</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <key>name</key><string>TO_NULL_AND_PRINT(a, &quot;Will this ) cause a crash?&quot;)</string>
// CHECK-NEXT:   <key>expansion</key><string>a =0;print (&quot;Will this ) cause a crash?&quot;)</string>
// CHECK-NEXT:  </dict>
// CHECK-NEXT: </array>

#define CALL_FUNCTION(funcCall)   \
  funcCall

// Function calls do contain both tok::comma and tok::l_paren/tok::r_paren.

void macroArgContainsLParenRParenTest() {
  int *a;
  CALL_FUNCTION(setToNull(&a));
  *a = 5; // expected-warning{{Dereference of null pointer}}
}

// CHECK:      <key>macro_expansions</key>
// CHECK-NEXT: <array>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>302</integer>
// CHECK-NEXT:    <key>col</key><integer>3</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <key>name</key><string>CALL_FUNCTION(setToNull(&amp;a))</string>
// CHECK-NEXT:   <key>expansion</key><string>setToNull (&amp;a )</string>
// CHECK-NEXT:  </dict>
// CHECK-NEXT: </array>

void setToNullAndPrint(int **vptr, const char *str) {
  setToNull(vptr);
  print(str);
}

void macroArgContainsCommaLParenRParenTest() {
  int *a;
  CALL_FUNCTION(setToNullAndPrint(&a, "Hello!"));
  *a = 5; // expected-warning{{Dereference of null pointer}}
}

// CHECK:      <key>macro_expansions</key>
// CHECK-NEXT: <array>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>327</integer>
// CHECK-NEXT:    <key>col</key><integer>3</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <key>name</key><string>CALL_FUNCTION(setToNullAndPrint(&amp;a, &quot;Hello!&quot;))</string>
// CHECK-NEXT:   <key>expansion</key><string>setToNullAndPrint (&amp;a ,&quot;Hello!&quot;)</string>
// CHECK-NEXT:  </dict>
// CHECK-NEXT: </array>

#define CALL_FUNCTION_WITH_TWO_PARAMS(funcCall, param1, param2) \
  funcCall(param1, param2)

void macroArgContainsCommaLParenRParenTest2() {
  int *a;
  CALL_FUNCTION_WITH_TWO_PARAMS(setToNullAndPrint, &a, "Hello!");
  *a = 5; // expected-warning{{Dereference of null pointer}}
}

// CHECK:      <key>macro_expansions</key>
// CHECK-NEXT: <array>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>350</integer>
// CHECK-NEXT:    <key>col</key><integer>3</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <key>name</key><string>CALL_FUNCTION_WITH_TWO_PARAMS(setToNullAndPrint, &amp;a, &quot;Hello!&quot;)</string>
// CHECK-NEXT:   <key>expansion</key><string>setToNullAndPrint (&amp;a ,&quot;Hello!&quot;)</string>
// CHECK-NEXT:  </dict>
// CHECK-NEXT: </array>

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
// FIXME: Why does the expansion appear twice?
// CHECK:      <key>macro_expansions</key>
// CHECK-NEXT: <array>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>376</integer>
// CHECK-NEXT:    <key>col</key><integer>3</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <key>name</key><string>CALL_LAMBDA(([&amp;ptr, str] () mutable { TO_NULL(&amp;ptr); }))</string>
// CHECK-NEXT:   <key>expansion</key><string>([&amp;ptr ,str ]()mutable {setToNull (&amp;ptr );})()</string>
// CHECK-NEXT:  </dict>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>376</integer>
// CHECK-NEXT:    <key>col</key><integer>3</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <key>name</key><string>CALL_LAMBDA(([&amp;ptr, str] () mutable { TO_NULL(&amp;ptr); }))</string>
// CHECK-NEXT:   <key>expansion</key><string>([&amp;ptr ,str ]()mutable {setToNull (&amp;ptr );})()</string>
// CHECK-NEXT:  </dict>
// CHECK-NEXT: </array>

#define PASTE_CODE(code) \
  code

void commaInBracesTest() {
  PASTE_CODE({ // expected-warning{{Dereference of null pointer}}
    // NOTE: If we were to add a new variable here after a comma, we'd get a
    // compilation error, so this test is mainly here to show that this was also
    // investigated.
    //
    // int *ptr = nullptr, a;
    int *ptr = nullptr;
    *ptr = 5;
  })
}

// CHECK:        <key>macro_expansions</key>
// CHECK-NEXT:   <array>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>location</key>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>line</key><integer>408</integer>
// CHECK-NEXT:      <key>col</key><integer>3</integer>
// CHECK-NEXT:      <key>file</key><integer>0</integer>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <key>name</key><string>PASTE_CODE({ // expected-
// CHECK-NEXT:    // NOTE: If we were to add a new variable here after a comma, we&apos;d get a
// CHECK-NEXT:    // compilation error, so this test is mainly here to show that this was also
// CHECK-NEXT:    // investigated.
// CHECK-NEXT:    //
// CHECK-NEXT:    // int *ptr = nullptr, a;
// CHECK-NEXT:    int *ptr = nullptr;
// CHECK-NEXT:    *ptr = 5;
// CHECK-NEXT:  })</string>
// CHECK-NEXT:     <key>expansion</key><string>{int *ptr =nullptr ;*ptr =5;}</string>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:   </array>

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

// CHECK:      <key>macro_expansions</key>
// CHECK-NEXT: <array>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>451</integer>
// CHECK-NEXT:    <key>col</key><integer>3</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <key>name</key><string>POTENTIALLY_EMPTY_PARAM(,ptr)</string>
// CHECK-NEXT:   <key>expansion</key><string>;ptr =nullptr </string>
// CHECK-NEXT:  </dict>
// CHECK-NEXT: </array>

#define NESTED_EMPTY_PARAM(a, b) \
  POTENTIALLY_EMPTY_PARAM(a, b);


void nestedEmptyParamTest() {
  int *ptr;

  NESTED_EMPTY_PARAM(, ptr);
  *ptr = 5; // expected-warning{{Dereference of null pointer}}
}

// CHECK:      <key>macro_expansions</key>
// CHECK-NEXT: <array>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>476</integer>
// CHECK-NEXT:    <key>col</key><integer>3</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <key>name</key><string>NESTED_EMPTY_PARAM(, ptr)</string>
// CHECK-NEXT:   <key>expansion</key><string>;ptr =nullptr ;</string>
// CHECK-NEXT:  </dict>
// CHECK-NEXT: </array>

#define CALL_FUNCTION_WITH_ONE_PARAM_THROUGH_MACRO(func, param) \
  CALL_FUNCTION(func(param))

void lParenRParenInNestedMacro() {
  int *ptr;
  CALL_FUNCTION_WITH_ONE_PARAM_THROUGH_MACRO(setToNull, &ptr);
  *ptr = 5; // expected-warning{{Dereference of null pointer}}
}

// CHECK:      <key>macro_expansions</key>
// CHECK-NEXT: <array>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>499</integer>
// CHECK-NEXT:    <key>col</key><integer>3</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <key>name</key><string>CALL_FUNCTION_WITH_ONE_PARAM_THROUGH_MACRO(setToNull, &amp;ptr)</string>
// CHECK-NEXT:   <key>expansion</key><string>setToNull (&amp;ptr )</string>
// CHECK-NEXT:  </dict>
// CHECK-NEXT: </array>

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

// CHECK:      <key>macro_expansions</key>
// CHECK-NEXT: <array>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>530</integer>
// CHECK-NEXT:    <key>col</key><integer>3</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <key>name</key><string>VARIADIC_SET_TO_NULL(ptr, 1, 5, &quot;haha!&quot;)</string>
// CHECK-NEXT:   <key>expansion</key><string>ptr =nullptr ;variadicFunc (1,5,&quot;haha!&quot;)</string>
// CHECK-NEXT:  </dict>
// CHECK-NEXT: </array>

void variadicMacroArgumentWithoutAnyArgumentTest() {
  int *ptr;
  // Not adding a single parameter to ... is silly (and also causes a
  // preprocessor warning), but is not an excuse to crash on it.
  VARIADIC_SET_TO_NULL(ptr);
  *ptr = 5; // expected-warning{{Dereference of null pointer}}
}

// CHECK:      <key>macro_expansions</key>
// CHECK-NEXT: <array>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>552</integer>
// CHECK-NEXT:    <key>col</key><integer>3</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <key>name</key><string>VARIADIC_SET_TO_NULL(ptr)</string>
// CHECK-NEXT:   <key>expansion</key><string>ptr =nullptr ;variadicFunc ()</string>
// CHECK-NEXT:  </dict>
// CHECK-NEXT: </array>

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

// CHECK:      <key>macro_expansions</key>
// CHECK-NEXT: <array>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>580</integer>
// CHECK-NEXT:    <key>col</key><integer>3</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <key>name</key><string>DECLARE_FUNC_AND_SET_TO_NULL(whatever, ptr)</string>
// CHECK-NEXT:   <key>expansion</key><string>void generated_whatever ();ptr =nullptr ;</string>
// CHECK-NEXT:  </dict>
// CHECK-NEXT: </array>

void macroArgContainsHashHashInStringTest() {
  int *a;
  TO_NULL_AND_PRINT(a, "Will this ## cause a crash?");
  *a = 5; // expected-warning{{Dereference of null pointer}}
}

// CHECK:      <key>macro_expansions</key>
// CHECK-NEXT: <array>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>600</integer>
// CHECK-NEXT:    <key>col</key><integer>3</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <key>name</key><string>TO_NULL_AND_PRINT(a, &quot;Will this ## cause a crash?&quot;)</string>
// CHECK-NEXT:   <key>expansion</key><string>a =0;print (&quot;Will this ## cause a crash?&quot;)</string>
// CHECK-NEXT:  </dict>
// CHECK-NEXT: </array>

#define PRINT_STR(str, ptr) \
  print(#str);              \
  ptr = nullptr

void hashOperatorTest() {
  int *ptr;
  PRINT_STR(Hello, ptr);
  *ptr = 5; // expected-warning{{Dereference of null pointer}}
}

// CHECK:      <key>macro_expansions</key>
// CHECK-NEXT: <array>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>624</integer>
// CHECK-NEXT:    <key>col</key><integer>3</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <key>name</key><string>PRINT_STR(Hello, ptr)</string>
// CHECK-NEXT:   <key>expansion</key><string>print (&quot;Hello&quot;);ptr =nullptr </string>
// CHECK-NEXT:  </dict>
// CHECK-NEXT: </array>

void macroArgContainsHashInStringTest() {
  int *a;
  TO_NULL_AND_PRINT(a, "Will this # cause a crash?");
  *a = 5; // expected-warning{{Dereference of null pointer}}
}

// CHECK:      <key>macro_expansions</key>
// CHECK-NEXT: <array>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>644</integer>
// CHECK-NEXT:    <key>col</key><integer>3</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <key>name</key><string>TO_NULL_AND_PRINT(a, &quot;Will this # cause a crash?&quot;)</string>
// CHECK-NEXT:   <key>expansion</key><string>a =0;print (&quot;Will this # cause a crash?&quot;)</string>
// CHECK-NEXT:  </dict>
// CHECK-NEXT: </array>

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

// CHECK:      <key>macro_expansions</key>
// CHECK-NEXT: <array>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>698</integer>
// CHECK-NEXT:    <key>col</key><integer>3</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <key>name</key><string>EUCLIDEAN_ALGORITHM(A, B)</string>
// CHECK-NEXT:   <key>expansion</key><string>if (A &lt;0){A =-A ;}if (B &lt;0){B =-B ;}return B /(B -B );</string>
// CHECK-NEXT:  </dict>
// CHECK-NEXT: </array>

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

// CHECK:      <key>macro_expansions</key>
// CHECK-NEXT: <array>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>730</integer>
// CHECK-NEXT:    <key>col</key><integer>3</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <key>name</key><string>YET_ANOTHER_SET_TO_NULL(5, DO_NOTHING2(&quot;Remember the Vasa&quot;), ptr)</string>
// CHECK-NEXT:   <key>expansion</key><string>print ((void *)5);print ((void *)&quot;Remember the Vasa&quot;);ptr =nullptr ;</string>
// CHECK-NEXT:  </dict>
// CHECK-NEXT: </array>

int garbage_value;

#define REC_MACRO_FUNC(REC_MACRO_PARAM) garbage_##REC_MACRO_PARAM
#define value REC_MACRO_FUNC(value)

void recursiveMacroUser() {
  if (value == 0)
    1 / value; // expected-warning{{Division by zero}}
               // expected-warning@-1{{expression result unused}}
}

// CHECK:      <key>macro_expansions</key>
// CHECK-NEXT: <array>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>754</integer>
// CHECK-NEXT:    <key>col</key><integer>7</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <key>name</key><string>value</string>
// CHECK-NEXT:   <key>expansion</key><string>garbage_value </string>
// CHECK-NEXT:  </dict>
// CHECK-NEXT: </array>

#define FOO(x) int foo() { return x; }
#define APPLY_ZERO1(function) function(0)

APPLY_ZERO1(FOO)
void useZeroApplier1() { (void)(1 / foo()); } // expected-warning{{Division by zero}}

// CHECK:      <key>macro_expansions</key>
// CHECK-NEXT: <array>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>776</integer>
// CHECK-NEXT:    <key>col</key><integer>1</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <key>name</key><string>APPLY_ZERO1(FOO)</string>
// CHECK-NEXT:   <key>expansion</key><string>int foo (){return 0;}</string>
// CHECK-NEXT:  </dict>
// CHECK-NEXT: </array>

#define BAR(x) int bar() { return x; }
#define APPLY_ZERO2 BAR(0)

APPLY_ZERO2
void useZeroApplier2() { (void)(1 / bar()); } // expected-warning{{Division by zero}}

// CHECK:      <key>macro_expansions</key>
// CHECK-NEXT: <array>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>796</integer>
// CHECK-NEXT:    <key>col</key><integer>1</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <key>name</key><string>APPLY_ZERO2</string>
// CHECK-NEXT:   <key>expansion</key><string>int bar (){return 0;}</string>
// CHECK-NEXT:  </dict>
// CHECK-NEXT: </array>

void foo(int &x, const char *str);

#define PARAMS_RESOLVE_TO_VA_ARGS(i, fmt) foo(i, fmt); \
  i = 0;
#define DISPATCH(...) PARAMS_RESOLVE_TO_VA_ARGS(__VA_ARGS__);

void mulitpleParamsResolveToVA_ARGS(void) {
  int x = 1;
  DISPATCH(x, "LF1M healer");
  (void)(10 / x); // expected-warning{{Division by zero}}
}

// CHECK:      <key>macro_expansions</key>
// CHECK-NEXT: <array>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>821</integer>
// CHECK-NEXT:    <key>col</key><integer>3</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <key>name</key><string>DISPATCH(x, &quot;LF1M healer&quot;)</string>
// CHECK-NEXT:   <key>expansion</key><string>foo (x ,&quot;LF1M healer&quot;);x =0;;</string>
// CHECK-NEXT:  </dict>
// CHECK-NEXT: </array>

void variadicCFunction(int &x, const char *str, ...);

#define CONCAT_VA_ARGS(i, fmt, ...) variadicCFunction(i, fmt, ##__VA_ARGS__); \
  i = 0;

void concatVA_ARGS(void) {
  int x = 1;
  CONCAT_VA_ARGS(x, "You need to construct additional pylons.", 'c', 9);
  (void)(10 / x); // expected-warning{{Division by zero}}
}

// CHECK:      <key>macro_expansions</key>
// CHECK-NEXT: <array>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>846</integer>
// CHECK-NEXT:    <key>col</key><integer>3</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <key>name</key><string>CONCAT_VA_ARGS(x, &quot;You need to construct additional pylons.&quot;, &apos;c&apos;, 9)</string>
// CHECK-NEXT:   <key>expansion</key><string>variadicCFunction (x ,&quot;You need to construct additional pylons.&quot;,&apos;c&apos;,9);x =0;</string>
// CHECK-NEXT:  </dict>
// CHECK-NEXT: </array>

void concatVA_ARGSEmpty(void) {
  int x = 1;
  CONCAT_VA_ARGS(x, "You need to construct");
  (void)(10 / x); // expected-warning{{Division by zero}}
}

// CHECK:      <key>macro_expansions</key>
// CHECK-NEXT: <array>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>866</integer>
// CHECK-NEXT:    <key>col</key><integer>3</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <key>name</key><string>CONCAT_VA_ARGS(x, &quot;You need to construct&quot;)</string>
// CHECK-NEXT:   <key>expansion</key><string>variadicCFunction (x ,&quot;You need to construct&quot;);x =0;</string>
// CHECK-NEXT:  </dict>
// CHECK-NEXT: </array>

#define STRINGIFIED_VA_ARGS(i, fmt, ...) variadicCFunction(i, fmt, #__VA_ARGS__); \
  i = 0;

void stringifyVA_ARGS(void) {
  int x = 1;
  STRINGIFIED_VA_ARGS(x, "Additional supply depots required.", 'a', 10);
  (void)(10 / x); // expected-warning{{Division by zero}}
}

// CHECK:      <key>macro_expansions</key>
// CHECK-NEXT: <array>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>889</integer>
// CHECK-NEXT:    <key>col</key><integer>3</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <key>name</key><string>STRINGIFIED_VA_ARGS(x, &quot;Additional supply depots required.&quot;, &apos;a&apos;, 10)</string>
// CHECK-NEXT:   <key>expansion</key><string>variadicCFunction (x ,&quot;Additional supply depots required.&quot;,&quot;&apos;a&apos;, 10&quot;);x =0;</string>
// CHECK-NEXT:  </dict>
// CHECK-NEXT: </array>

void stringifyVA_ARGSEmpty(void) {
  int x = 1;
  STRINGIFIED_VA_ARGS(x, "Additional supply depots required.");
  (void)(10 / x); // expected-warning{{Division by zero}}
}

// CHECK:      <key>macro_expansions</key>
// CHECK-NEXT: <array>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>909</integer>
// CHECK-NEXT:    <key>col</key><integer>3</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <key>name</key><string>STRINGIFIED_VA_ARGS(x, &quot;Additional supply depots required.&quot;)</string>
// CHECK-NEXT:   <key>expansion</key><string>variadicCFunction (x ,&quot;Additional supply depots required.&quot;,&quot;&quot;);x =0;</string>
// CHECK-NEXT:  </dict>
// CHECK-NEXT: </array>

// bz44493: Support GNU-style named variadic arguments in plister
#define BZ44493_GNUVA(i, args...)  --(i);

int bz44493(void) {
  int a = 2;
  BZ44493_GNUVA(a);
  BZ44493_GNUVA(a, "arg2");
  (void)(10 / a); // expected-warning{{Division by zero}}
  return 0;
}

// CHECK:      <key>macro_expansions</key>
// CHECK-NEXT: <array>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>933</integer>
// CHECK-NEXT:    <key>col</key><integer>3</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <key>name</key><string>BZ44493_GNUVA(a, &quot;arg2&quot;)</string>
// CHECK-NEXT:   <key>expansion</key><string>--(a );</string>
// CHECK-NEXT:  </dict>
// CHECK-NEXT: </array>
