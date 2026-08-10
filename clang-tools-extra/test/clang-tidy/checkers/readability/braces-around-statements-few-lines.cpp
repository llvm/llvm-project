// RUN: %check_clang_tidy %s readability-braces-around-statements %t -- -config="{CheckOptions: {readability-braces-around-statements.ShortStatementLines: 4}}" --

void do_something(const char *) {}

bool cond(const char *) {
  return false;
}

void test() {
  if (cond("if1") /*comment*/) do_something("same-line");

  if (cond("if2"))
    do_something("single-line");

  if (cond("if3") /*comment*/)
    // some comment
    do_something("three"
                 "lines");

  if (cond("if4") /*comment*/)
    // some comment
    do_something("many"
                 "many"
                 "many"
                 "many"
                 "lines");
  // CHECK-MESSAGES: :[[@LINE-7]]:31: warning: statement should be inside braces
  // CHECK-FIXES: if (cond("if4") /*comment*/) {
  // CHECK-FIXES: }
}
