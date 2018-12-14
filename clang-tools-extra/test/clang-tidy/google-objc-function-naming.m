// RUN: %check_clang_tidy %s google-objc-function-naming %t

typedef _Bool bool;

static bool ispositive(int a) { return a > 0; }
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: static function named 'ispositive'
// must be in Pascal case as required by Google Objective-C style guide
// CHECK-FIXES: static bool Ispositive(int a) { return a > 0; }

static bool is_positive(int a) { return a > 0; }
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: static function named 'is_positive'
// must be in Pascal case as required by Google Objective-C style guide
// CHECK-FIXES: static bool IsPositive(int a) { return a > 0; }

static bool isPositive(int a) { return a > 0; }
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: static function named 'isPositive'
// must be in Pascal case as required by Google Objective-C style guide
// CHECK-FIXES: static bool IsPositive(int a) { return a > 0; }

static bool Is_Positive(int a) { return a > 0; }
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: static function named 'Is_Positive'
// must be in Pascal case as required by Google Objective-C style guide
// CHECK-FIXES: static bool IsPositive(int a) { return a > 0; }

static bool IsPositive(int a) { return a > 0; }

bool ispalindrome(const char *str);
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function in global namespace named
// 'ispalindrome' must have an appropriate prefix followed by Pascal case as
// required by Google Objective-C style guide

static const char *md5(const char *str) { return 0; }
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: static function named 'md5' must be
// in Pascal case as required by Google Objective-C style guide
// CHECK-FIXES: static const char *Md5(const char *str) { return 0; }

static const char *MD5(const char *str) { return 0; }

static const char *URL(void) { return "https://clang.llvm.org/"; }

static const char *DEFURL(void) { return "https://clang.llvm.org/"; }

static const char *DEFFooURL(void) { return "https://clang.llvm.org/"; }

static const char *StringFromNSString(id str) { return ""; }

void ABLog_String(const char *str);
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function in global namespace named
// 'ABLog_String' must have an appropriate prefix followed by Pascal case as
// required by Google Objective-C style guide

void ABLogString(const char *str);

bool IsPrime(int a);
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function in global namespace named
// 'IsPrime' must have an appropriate prefix followed by Pascal case as required
// by Google Objective-C style guide

const char *ABURL(void) { return "https://clang.llvm.org/"; }

const char *ABFooURL(void) { return "https://clang.llvm.org/"; }

int main(int argc, const char **argv) { return 0; }
