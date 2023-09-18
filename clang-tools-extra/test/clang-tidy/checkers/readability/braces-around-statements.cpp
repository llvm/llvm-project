// RUN: %check_clang_tidy %s readability-braces-around-statements %t

void do_something(const char *) {}

bool cond(const char *) {
  return false;
}

#define EMPTY_MACRO
#define EMPTY_MACRO_FUN()

void test() {
  if (cond("if0") /*comment*/) do_something("same-line");
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: statement should be inside braces
  // CHECK-FIXES:   if (cond("if0") /*comment*/) { do_something("same-line");
  // CHECK-FIXES: }

  if (cond("if0.1"))
    do_something("single-line");
  // CHECK-MESSAGES: :[[@LINE-2]]:21: warning: statement should be inside braces
  // CHECK-FIXES: if (cond("if0.1")) {
  // CHECK-FIXES: }

  if (cond("if1") /*comment*/)
    // some comment
    do_something("if1");
  // CHECK-MESSAGES: :[[@LINE-3]]:31: warning: statement should be inside braces
  // CHECK-FIXES: if (cond("if1") /*comment*/) {
  // CHECK-FIXES: }
  if (cond("if2")) {
    do_something("if2");
  }
  if (cond("if3"))
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:19: warning: statement should be inside braces
  // CHECK-FIXES: if (cond("if3")) {
  // CHECK-FIXES: }

  if (cond("if-else1"))
    do_something("if-else1");
  // CHECK-MESSAGES: :[[@LINE-2]]:24: warning: statement should be inside braces
  // CHECK-FIXES: if (cond("if-else1")) {
  // CHECK-FIXES: } else {
  else
    do_something("if-else1 else");
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: statement should be inside braces
  // CHECK-FIXES: }
  if (cond("if-else2")) {
    do_something("if-else2");
  } else {
    do_something("if-else2 else");
  }

  if (cond("if-else if-else1"))
    do_something("if");
  // CHECK-MESSAGES: :[[@LINE-2]]:32: warning: statement should be inside braces
  // CHECK-FIXES: } else if (cond("else if1")) {
  else if (cond("else if1"))
    do_something("else if");
  // CHECK-MESSAGES: :[[@LINE-2]]:29: warning: statement should be inside braces
  else
    do_something("else");
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: statement should be inside braces
  // CHECK-FIXES: }
  if (cond("if-else if-else2")) {
    do_something("if");
  } else if (cond("else if2")) {
    do_something("else if");
  } else {
    do_something("else");
  }

  for (;;)
    do_something("for");
  // CHECK-MESSAGES: :[[@LINE-2]]:11: warning: statement should be inside braces
  // CHECK-FIXES: for (;;) {
  // CHECK-FIXES-NEXT: do_something("for");
  // CHECK-FIXES-NEXT: }

  for (;;) {
    do_something("for-ok");
  }
  for (;;)
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:11: warning: statement should be inside braces
  // CHECK-FIXES: for (;;) {
  // CHECK-FIXES-NEXT: ;
  // CHECK-FIXES-NEXT: }

  int arr[4] = {1, 2, 3, 4};
  for (int a : arr)
    do_something("for-range");
  // CHECK-MESSAGES: :[[@LINE-2]]:20: warning: statement should be inside braces
  // CHECK-FIXES: for (int a : arr) {
  // CHECK-FIXES-NEXT: do_something("for-range");
  // CHECK-FIXES-NEXT: }
  for (int &assign : arr)
    assign = 7;
  // CHECK-MESSAGES: :[[@LINE-2]]:26: warning: statement should be inside braces
  // CHECK-FIXES: for (int &assign : arr) {
  // CHECK-FIXES-NEXT: assign = 7;
  // CHECK-FIXES-NEXT: }
  for (int ok : arr) {
    do_something("for-range");
  }
  for (int NullStmt : arr)
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:27: warning: statement should be inside braces
  // CHECK-FIXES: for (int NullStmt : arr) {
  // CHECK-FIXES-NEXT: ;
  // CHECK-FIXES-NEXT: }

  while (cond("while1"))
    do_something("while");
  // CHECK-MESSAGES: :[[@LINE-2]]:25: warning: statement should be inside braces
  // CHECK-FIXES: while (cond("while1")) {
  // CHECK-FIXES: }
  while (cond("while2")) {
    do_something("while");
  }
  while (false)
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:16: warning: statement should be inside braces
  // CHECK-FIXES: while (false) {
  // CHECK-FIXES: }

  do
    do_something("do1");
  while (cond("do1"));
  // CHECK-MESSAGES: :[[@LINE-3]]:5: warning: statement should be inside braces
  // CHECK-FIXES: do {
  // CHECK-FIXES: } while (cond("do1"));
  do {
    do_something("do2");
  } while (cond("do2"));

  do
    ;
  while (false);
  // CHECK-MESSAGES: :[[@LINE-3]]:5: warning: statement should be inside braces
  // CHECK-FIXES: do {
  // CHECK-FIXES: } while (false);

  if (cond("ifif1"))
    // comment
    if (cond("ifif2"))
      // comment
      /*comment*/ ; // comment
  // CHECK-MESSAGES: :[[@LINE-5]]:21: warning: statement should be inside braces
  // CHECK-MESSAGES: :[[@LINE-4]]:23: warning: statement should be inside braces
  // CHECK-FIXES: if (cond("ifif1")) {
  // CHECK-FIXES: if (cond("ifif2")) {
  // CHECK-FIXES: }
  // CHECK-FIXES-NEXT: }

  if (cond("ifif3"))
    // comment1
    if (cond("ifif4")) {
      // comment2
      /*comment3*/; // comment4
    }
  // CHECK-MESSAGES: :[[@LINE-6]]:21: warning: statement should be inside braces
  // CHECK-FIXES: if (cond("ifif3")) {
  // CHECK-FIXES-NEXT: // comment1
  // CHECK-FIXES-NEXT: if (cond("ifif4")) {
  // CHECK-FIXES-NEXT: // comment2
  // CHECK-FIXES-NEXT: /*comment3*/; // comment4
  // CHECK-FIXES-NEXT: }
  // CHECK-FIXES-NEXT: }

  if (cond("ifif5"))
    ; /* multi-line
        comment */
  // CHECK-MESSAGES: :[[@LINE-3]]:21: warning: statement should be inside braces
  // CHECK-FIXES: if (cond("ifif5")) {
  // CHECK-FIXES: }/* multi-line

  if (1) while (2) if (3) for (;;) do ; while(false) /**/;/**/
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: statement should be inside braces
  // CHECK-MESSAGES: :[[@LINE-2]]:19: warning: statement should be inside braces
  // CHECK-MESSAGES: :[[@LINE-3]]:26: warning: statement should be inside braces
  // CHECK-MESSAGES: :[[@LINE-4]]:35: warning: statement should be inside braces
  // CHECK-MESSAGES: :[[@LINE-5]]:38: warning: statement should be inside braces
  // CHECK-FIXES: if (1) { while (2) { if (3) { for (;;) { do { ; } while(false) /**/;/**/
  // CHECK-FIXES-NEXT: }
  // CHECK-FIXES-NEXT: }
  // CHECK-FIXES-NEXT: }
  // CHECK-FIXES-NEXT: }

  int S;
  if (cond("assign with brackets"))
    S = {5};
  // CHECK-MESSAGES: :[[@LINE-2]]:36: warning: statement should be inside braces
  // CHECK-FIXES: if (cond("assign with brackets")) {
  // CHECK-FIXES-NEXT: S = {5};
  // CHECK-FIXES-NEXT: }

  if (cond("assign with brackets 2"))
    S = {  5  } /* comment1 */ ; /* comment2 */
  // CHECK-MESSAGES: :[[@LINE-2]]:38: warning: statement should be inside braces
  // CHECK-FIXES: if (cond("assign with brackets 2")) {
  // CHECK-FIXES-NEXT: S = {  5  } /* comment1 */ ; /* comment2 */
  // CHECK-FIXES-NEXT: }

  if (cond("return"))
    return;
  // CHECK-MESSAGES: :[[@LINE-2]]:22: warning: statement should be inside braces
  // CHECK-FIXES: if (cond("return")) {
  // CHECK-FIXES-NEXT: return;
  // CHECK-FIXES-NEXT: }

  while (cond("break and continue")) {
    // CHECK-FIXES: while (cond("break and continue")) {
    if (true)
      break;
    // CHECK-MESSAGES: :[[@LINE-2]]:14: warning: statement should be inside braces
    // CHECK-FIXES: {{^}}    if (true) {{{$}}
    // CHECK-FIXES-NEXT: {{^}}      break;{{$}}
    // CHECK-FIXES-NEXT: {{^ *}}}{{$}}
    if (false)
      continue;
    // CHECK-MESSAGES: :[[@LINE-2]]:15: warning: statement should be inside braces
    // CHECK-FIXES: {{^}}    if (false) {{{$}}
    // CHECK-FIXES-NEXT: {{^}}      continue;{{$}}
    // CHECK-FIXES-NEXT: {{^ *}}}{{$}}
  } //end
  // CHECK-FIXES: } //end

  if (cond("decl 1"))
    int s;
  else
    int t;
  // CHECK-MESSAGES: :[[@LINE-4]]:22: warning: statement should be inside braces
  // CHECK-MESSAGES: :[[@LINE-3]]:7: warning: statement should be inside braces
  // CHECK-FIXES: if (cond("decl 1")) {
  // CHECK-FIXES-NEXT: int s;
  // CHECK-FIXES-NEXT: } else {
  // CHECK-FIXES-NEXT: int t;
  // CHECK-FIXES-NEXT: }

  if (cond("decl 2"))
    int s = (5);
  else
    int t = (5);
  // CHECK-MESSAGES: :[[@LINE-4]]:22: warning: statement should be inside braces
  // CHECK-MESSAGES: :[[@LINE-3]]:7: warning: statement should be inside braces
  // CHECK-FIXES: if (cond("decl 2")) {
  // CHECK-FIXES-NEXT: int s = (5);
  // CHECK-FIXES-NEXT: } else {
  // CHECK-FIXES-NEXT: int t = (5);
  // CHECK-FIXES-NEXT: }

  if (cond("decl 3"))
    int s = {6};
  else
    int t = {6};
  // CHECK-MESSAGES: :[[@LINE-4]]:22: warning: statement should be inside braces
  // CHECK-MESSAGES: :[[@LINE-3]]:7: warning: statement should be inside braces
  // CHECK-FIXES: if (cond("decl 3")) {
  // CHECK-FIXES-NEXT: int s = {6};
  // CHECK-FIXES-NEXT: } else {
  // CHECK-FIXES-NEXT: int t = {6};
  // CHECK-FIXES-NEXT: }
}

void test_whitespace() {
  while(cond("preserve empty lines"))
    if(cond("using continue within if"))
      continue;


  test();

  // CHECK-MESSAGES: :[[@LINE-7]]:{{[0-9]+}}: warning: statement should be inside braces
  // CHECK-MESSAGES: :[[@LINE-7]]:{{[0-9]+}}: warning: statement should be inside braces
  // CHECK-FIXES: {{^}}  while(cond("preserve empty lines")) {{{$}}
  // CHECK-FIXES-NEXT: {{^}}    if(cond("using continue within if")) {{{$}}
  // CHECK-FIXES-NEXT: {{^      continue;$}}
  // The closing brace is added at beginning of line, clang-format can be
  // applied afterwards.
  // CHECK-FIXES-NEXT: {{^}$}}
  // CHECK-FIXES-NEXT: {{^}$}}
  // Following whitespace is assumed to not to belong to the else branch.
  // However the check is not possible with CHECK-FIXES-NEXT.
  // CHECK-FIXES: {{^}}  test();{{$}}

  if (cond("preserve empty lines"))
 
  
    int s;
   
    
  else
 
  
    int t;
   
    
  test();

  // CHECK-MESSAGES: :[[@LINE-14]]:{{[0-9]+}}: warning: statement should be inside braces
  // CHECK-MESSAGES: :[[@LINE-9]]:{{[0-9]+}}: warning: statement should be inside braces
  // CHECK-FIXES: {{^}}  if (cond("preserve empty lines")) {{{$}}
  // CHECK-FIXES-NEXT: {{^ $}}
  // CHECK-FIXES-NEXT: {{^  $}}
  // CHECK-FIXES-NEXT: {{^    int s;$}}
  // CHECK-FIXES-NEXT: {{^   $}}
  // CHECK-FIXES-NEXT: {{^    $}}
  // CHECK-FIXES-NEXT: {{^  } else {$}}
  // CHECK-FIXES-NEXT: {{^ $}}
  // CHECK-FIXES-NEXT: {{^  $}}
  // CHECK-FIXES-NEXT: {{^    int t;$}}
  // The closing brace is added at beginning of line, clang-format can be
  // applied afterwards.
  // CHECK-FIXES-NEXT: {{^}$}}
  // Following whitespace is assumed to not to belong to the else branch.
  // CHECK-FIXES-NEXT: {{^   $}}
  // CHECK-FIXES-NEXT: {{^    $}}
  // CHECK-FIXES-NEXT: {{^}}  test();{{$}}
}

int test_return_int() {
  if (cond("return5"))
    return 5;
  // CHECK-MESSAGES: :[[@LINE-2]]:23: warning: statement should be inside braces
  // CHECK-FIXES: if (cond("return5")) {
  // CHECK-FIXES-NEXT: return 5;
  // CHECK-FIXES-NEXT: }

  if (cond("return{6}"))
    return {6};
  // CHECK-MESSAGES: :[[@LINE-2]]:25: warning: statement should be inside braces
  // CHECK-FIXES: if (cond("return{6}")) {
  // CHECK-FIXES-NEXT: return {6};
  // CHECK-FIXES-NEXT: }

  // From https://bugs.llvm.org/show_bug.cgi?id=25970
  if (cond("25970")) return {25970};
  return {!25970};
  // CHECK-MESSAGES: :[[@LINE-2]]:21: warning: statement should be inside braces
  // CHECK-FIXES: if (cond("25970")) { return {25970};
  // CHECK-FIXES-NEXT: }
  // CHECK-FIXES-NEXT: return {!25970};
}

void f(const char *p) {
  if (!p)
    f("\
");
} // end of f
// CHECK-MESSAGES: :[[@LINE-4]]:10: warning: statement should be inside braces
// CHECK-FIXES:      {{^}}  if (!p) {{{$}}
// CHECK-FIXES-NEXT: {{^}}    f("\{{$}}
// CHECK-FIXES-NEXT: {{^}}");{{$}}
// CHECK-FIXES-NEXT: {{^}}}{{$}}
// CHECK-FIXES-NEXT: {{^}}} // end of f{{$}}

#define M(x) x

int test_macros(bool b) {
  if (b) {
    return 1;
  } else
    M(return 2);
  // CHECK-MESSAGES: :[[@LINE-2]]:9: warning: statement should be inside braces
  // CHECK-FIXES: } else {
  // CHECK-FIXES-NEXT:   M(return 2);
  // CHECK-FIXES-NEXT: }
  M(
    for (;;)
      ;
  );
  // CHECK-MESSAGES: :[[@LINE-3]]:13: warning: statement should be inside braces
  // CHECK-FIXES: {{^}}    for (;;) {{{$}}
  // CHECK-FIXES-NEXT: {{^      ;$}}
  // CHECK-FIXES-NEXT: {{^}$}}


  #define WRAP(X) { X; }
  // This is to ensure no other CHECK-FIXES matches the macro definition:
  // CHECK-FIXES: WRAP

  // Use-case: LLVM_DEBUG({ for(...) do_something(); });
  WRAP({
    for (;;)
      do_something("for in wrapping macro 1");
    });
  // CHECK-MESSAGES: :[[@LINE-3]]:13: warning: statement should be inside braces
  // CHECK-FIXES: for (;;) {
  // CHECK-FIXES-NEXT: do_something("for in wrapping macro 1");
  // CHECK-FIXES-NEXT: }

  // Use-case: LLVM_DEBUG( for(...) do_something(); );
  WRAP(
    for (;;)
      do_something("for in wrapping macro 2");
    );
  // CHECK-MESSAGES: :[[@LINE-3]]:13: warning: statement should be inside braces
  // CHECK-FIXES: for (;;) {
  // CHECK-FIXES-NEXT: do_something("for in wrapping macro 2");
  // CHECK-FIXES-NEXT: }

  // Use-case: LLVM_DEBUG( for(...) do_something() );
  // This is not supported and this test ensure it's correctly not changed.
  // We don't want to add the `}` into the Macro and there is no other way
  // to add it except for introduction of a NullStmt.
  WRAP(
    for (;;)
        do_something("for in wrapping macro 3")
    );
  // CHECK-MESSAGES: :[[@LINE-3]]:13: warning: statement should be inside braces
  // CHECK-FIXES: WRAP(
  // CHECK-FIXES-NEXT: for (;;)
  // CHECK-FIXES-NEXT: do_something("for in wrapping macro 3")
  // CHECK-FIXES-NEXT: );

  // Taken from https://bugs.llvm.org/show_bug.cgi?id=22785
  int i;
  #define MACRO_1 i++
  #define MACRO_2
  if( i % 3) i--;
  else if( i % 2) MACRO_1;
  else MACRO_2;
  // CHECK-MESSAGES: :[[@LINE-3]]:13: warning: statement should be inside braces
  // CHECK-MESSAGES: :[[@LINE-3]]:18: warning: statement should be inside braces
  // CHECK-MESSAGES: :[[@LINE-3]]:7: warning: statement should be inside braces
  // CHECK-FIXES: if( i % 3) { i--;
  // CHECK-FIXES-NEXT: } else if( i % 2) { MACRO_1;
  // CHECK-FIXES-NEXT: } else { MACRO_2;
  // CHECK-FIXES-NEXT: }

  // Taken from https://bugs.llvm.org/show_bug.cgi?id=22785
  #define M(x) x

  if (b)
    return 1;
  else
    return 2;
  // CHECK-MESSAGES: :[[@LINE-4]]:9: warning: statement should be inside braces
  // CHECK-MESSAGES: :[[@LINE-3]]:7: warning: statement should be inside braces
  // CHECK-FIXES: if (b) {
  // CHECK-FIXES-NEXT: return 1;
  // CHECK-FIXES-NEXT: } else {
  // CHECK-FIXES-NEXT: return 2;
  // CHECK-FIXES-NEXT: }

  if (b)
    return 1;
  else
    M(return 2);
  // CHECK-MESSAGES: :[[@LINE-4]]:9: warning: statement should be inside braces
  // CHECK-MESSAGES: :[[@LINE-3]]:7: warning: statement should be inside braces
  // CHECK-FIXES: if (b) {
  // CHECK-FIXES-NEXT: return 1;
  // CHECK-FIXES-NEXT: } else {
  // CHECK-FIXES-NEXT: M(return 2);
  // CHECK-FIXES-NEXT: }

}

template <bool A>
auto constexpr_lambda_1 = [] {
  if constexpr (A) {
    1;
  }
};

template <bool A>
auto constexpr_lambda_2 = [] {
  // CHECK-MESSAGES: :[[@LINE+1]]:19: warning: statement should be inside braces
  if constexpr (A)
    1;
  // CHECK-FIXES:if constexpr (A) {
  // CHECK-FIXES-NEXT:1;
  // CHECK-FIXES-NEXT:}
};

void test_constexpr() {
  constexpr_lambda_1<false>();
  constexpr_lambda_2<false>();
}
