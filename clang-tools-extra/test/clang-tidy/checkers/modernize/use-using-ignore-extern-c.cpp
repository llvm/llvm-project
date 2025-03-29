// RUN: %check_clang_tidy %s modernize-use-using %t -- -config="{CheckOptions: {modernize-use-using.IgnoreExternC: true}}" -- -I %S/Input/use-using/

// Some Header
extern "C" {

typedef int NewInt;
}

extern "C++" {

typedef int InExternCPP;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef' [modernize-use-using]
// CHECK-FIXES: using InExternCPP = int;
}
