// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s

// Nullability of const string-like globals, testing
// NonnullGlobalConstantsChecker.

void clang_analyzer_eval(bool);

@class NSString;
typedef const struct __CFString *CFStringRef;
typedef const struct __CFBoolean * CFBooleanRef;

// Global NSString* is non-null.
extern NSString *const StringConstGlobal;
void stringConstGlobal() {
  clang_analyzer_eval(StringConstGlobal); // expected-warning{{TRUE}}
}

// The logic does not apply to local variables though.
extern NSString *stringGetter();
void stringConstLocal() {
  NSString *const local = stringGetter();
  clang_analyzer_eval(local); // expected-warning{{UNKNOWN}}
}

// Global const CFStringRef's are also assumed to be non-null.
extern const CFStringRef CFStringConstGlobal;
void cfStringCheckGlobal() {
  clang_analyzer_eval(CFStringConstGlobal); // expected-warning{{TRUE}}
}

// But only "const" ones.
extern CFStringRef CFStringNonConstGlobal;
void cfStringCheckMutableGlobal() {
  clang_analyzer_eval(CFStringNonConstGlobal); // expected-warning{{UNKNOWN}}
}

// char* const is also assumed to be non-null.
extern const char *const ConstCharStarConst;
void constCharStarCheckGlobal() {
  clang_analyzer_eval(ConstCharStarConst); // expected-warning{{TRUE}}
}

// Pointer value can be mutable.
extern char *const CharStarConst;
void charStarCheckGlobal() {
  clang_analyzer_eval(CharStarConst); // expected-warning{{TRUE}}
}

// But the pointer itself should be immutable.
extern char *CharStar;
void charStartCheckMutableGlobal() {
  clang_analyzer_eval(CharStar); // expected-warning{{UNKNOWN}}
}

// Type definitions should also work across typedefs, for pointers:
typedef char *const str;
extern str globalStr;
void charStarCheckTypedef() {
  clang_analyzer_eval(globalStr); // expected-warning{{TRUE}}
}

// And for types.
typedef NSString *const NStr;
extern NStr globalNSString;
void NSStringCheckTypedef() {
  clang_analyzer_eval(globalNSString); // expected-warning{{TRUE}}
}

// Note that constness could be either inside
// the var declaration, or in a typedef.
typedef NSString *NStr2;
extern const NStr2 globalNSString2;
void NSStringCheckConstTypedef() {
  clang_analyzer_eval(globalNSString2); // expected-warning{{TRUE}}
}

// Nested typedefs should work as well.
typedef const CFStringRef str1;
typedef str1 str2;
extern str2 globalStr2;
void testNestedTypedefs() {
  clang_analyzer_eval(globalStr2); // expected-warning{{TRUE}}
}

// And for NSString *.
typedef NSString *const nstr1;
typedef nstr1 nstr2;
extern nstr2 nglobalStr2;
void testNestedTypedefsForNSString() {
  clang_analyzer_eval(nglobalStr2); // expected-warning{{TRUE}}
}

// And for CFBooleanRefs.
extern const CFBooleanRef kBool;
void testNonnullBool() {
  clang_analyzer_eval(kBool); // expected-warning{{TRUE}}
}

// And again, only for const one.
extern CFBooleanRef kBoolMutable;
void testNonnullNonconstBool() {
  clang_analyzer_eval(kBoolMutable); // expected-warning{{UNKNOWN}}
}
