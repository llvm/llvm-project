// RUN: %check_clang_tidy %s google-objc-function-naming %t

typedef _Bool bool;

static bool ispositive(int a) { return a > 0; }
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: function name 'ispositive' not using function naming conventions described by Google Objective-C style guide
// CHECK-FIXES: static bool Ispositive(int a) { return a > 0; }

static bool is_positive(int a) { return a > 0; }
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: function name 'is_positive' not using function naming conventions described by Google Objective-C style guide
// CHECK-FIXES: static bool IsPositive(int a) { return a > 0; }

static bool isPositive(int a) { return a > 0; }
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: function name 'isPositive' not using function naming conventions described by Google Objective-C style guide
// CHECK-FIXES: static bool IsPositive(int a) { return a > 0; }

static bool Is_Positive(int a) { return a > 0; }
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: function name 'Is_Positive' not using function naming conventions described by Google Objective-C style guide
// CHECK-FIXES: static bool IsPositive(int a) { return a > 0; }

static bool IsPositive(int a) { return a > 0; }

bool ispalindrome(const char *str);
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function name 'ispalindrome' not using function naming conventions described by Google Objective-C style guide

static const char *md5(const char *str) { return 0; }
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: function name 'md5' not using function naming conventions described by Google Objective-C style guide
// CHECK-FIXES: static const char *Md5(const char *str) { return 0; }

static const char *MD5(const char *str) { return 0; }

static const char *URL(void) { return "https://clang.llvm.org/"; }

static const char *DEFURL(void) { return "https://clang.llvm.org/"; }

static const char *DEFFooURL(void) { return "https://clang.llvm.org/"; }

static const char *StringFromNSString(id str) { return ""; }

void ABLog_String(const char *str);
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function name 'ABLog_String' not using function naming conventions described by Google Objective-C style guide

void ABLogString(const char *str);

bool IsPrime(int a);
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function name 'IsPrime' not using function naming conventions described by Google Objective-C style guide

const char *ABURL(void) { return "https://clang.llvm.org/"; }

const char *ABFooURL(void) { return "https://clang.llvm.org/"; }

int main(int argc, const char **argv) { return 0; }
