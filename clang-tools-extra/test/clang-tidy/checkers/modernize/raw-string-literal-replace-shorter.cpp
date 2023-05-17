// RUN: %check_clang_tidy %s modernize-raw-string-literal %t

// Don't replace these, because the raw literal would be longer.
char const *const JustAQuote("quote:\'");
char const *const NeedDelimiter("\":)\"");

char const *const ManyQuotes("quotes:\'\'\'\'");
// CHECK-MESSAGES: :[[@LINE-1]]:30: warning: {{.*}} can be written as a raw string literal
// CHECK-FIXES: {{^}}char const *const ManyQuotes(R"(quotes:'''')");{{$}}

char const *const LongOctal("\042\072\051\042");
// CHECK-MESSAGES: :[[@LINE-1]]:29: warning: {{.*}} can be written as a raw string literal
// CHECK-FIXES: {{^}}char const *const LongOctal(R"lit(":)")lit");{{$}}
