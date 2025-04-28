// RUN: %check_clang_tidy %s modernize-raw-string-literal %t -- -config='{CheckOptions: {modernize-raw-string-literal.DelimiterStem: "str", modernize-raw-string-literal.ReplaceShorterLiterals: true}}'

char const *const ContainsSentinel{"who\\ops)\""};
// CHECK-MESSAGES: :[[@LINE-1]]:36: warning: {{.*}} can be written as a raw string literal
// CHECK-FIXES: {{^}}char const *const ContainsSentinel{R"str(who\ops)")str"};{{$}}

//char const *const ContainsDelim{"whoops)\")lit\""};
// CHECK-XMESSAGES: :[[@LINE-1]]:33: warning: {{.*}} can be written as a raw string literal
// CHECK-XFIXES: {{^}}char const *const ContainsDelim{R"lit1(whoops)")lit")lit1"};{{$}}
