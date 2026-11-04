// RUN: %check_clang_tidy %s modernize-use-using %t -- -config="{CheckOptions: {modernize-use-using.IgnoreExternC: false}}" -- -I %S/Input/use-using/

// Some Header
extern "C" {

typedef int NewInt;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef' [modernize-use-using]
// CHECK-FIXES: using NewInt = int;
}

extern "C++" {

typedef int InExternCPP;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef' [modernize-use-using]
// CHECK-FIXES: using InExternCPP = int;
}
