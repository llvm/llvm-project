// RUN: %check_clang_tidy %s performance-substr-self-assignment %t
#include <string>

void OneArg() {
  std::string s = "hello world";

  // Basic case: s = s.substr(pos)
  s = s.substr(5);
  // CHECK-MESSAGES: [[@LINE-1]]:5: warning: inefficient self-assignment via 'substr'; use 'erase' to modify the string in-place [performance-substr-self-assignment]
  // CHECK-FIXES: s.erase(0, 5);

  // With a variable as the argument.
  int pos = 3;
  s = s.substr(pos);
  // CHECK-MESSAGES: [[@LINE-1]]:5: warning: inefficient self-assignment via 'substr'; use 'erase' to modify the string in-place
  // CHECK-FIXES: s.erase(0, pos);

  // With a more complex expression.
  s = s.substr(pos + 1);
  // CHECK-MESSAGES: [[@LINE-1]]:5: warning: inefficient self-assignment via 'substr'; use 'erase' to modify the string in-place
  // CHECK-FIXES: s.erase(0, pos + 1);
}

void WholeStringCopies() {
  std::string s = "hello world";

  // A full self-copy via substr() has no erase() rewrite; not diagnosed.
  s = s.substr();

  // substr(0) is still a pointless full copy; the fix-it degenerates to a
  // no-op erase.
  s = s.substr(0);
  // CHECK-MESSAGES: [[@LINE-1]]:5: warning: inefficient self-assignment via 'substr'; use 'erase' to modify the string in-place
  // CHECK-FIXES: s.erase(0, 0);
}

void TruncationNoFixIt() {
  std::string s = "hello world";

  // The truncation form s = s.substr(0, count) is diagnosed but gets no
  // fix-it: 'erase(count)' throws std::out_of_range when count > size(),
  // while 'substr(0, count)' clamps.
  s = s.substr(0, 5);
  // CHECK-MESSAGES: [[@LINE-1]]:5: warning: inefficient self-assignment via 'substr'; use 'erase' to modify the string in-place
  // CHECK-FIXES: s = s.substr(0, 5);

  size_t count = 3;
  s = s.substr(0, count);
  // CHECK-MESSAGES: [[@LINE-1]]:5: warning: inefficient self-assignment via 'substr'; use 'erase' to modify the string in-place
  // CHECK-FIXES: s = s.substr(0, count);
}

void TwoArgWithNpos() {
  std::string s = "hello world";

  // s = s.substr(pos, npos) is equivalent to s = s.substr(pos).
  s = s.substr(3, std::string::npos);
  // CHECK-MESSAGES: [[@LINE-1]]:5: warning: inefficient self-assignment via 'substr'; use 'erase' to modify the string in-place
  // CHECK-FIXES: s.erase(0, 3);

  int pos = 2;
  s = s.substr(pos, std::string::npos);
  // CHECK-MESSAGES: [[@LINE-1]]:5: warning: inefficient self-assignment via 'substr'; use 'erase' to modify the string in-place
  // CHECK-FIXES: s.erase(0, pos);

  // Member-syntax spelling of npos.
  s = s.substr(4, s.npos);
  // CHECK-MESSAGES: [[@LINE-1]]:5: warning: inefficient self-assignment via 'substr'; use 'erase' to modify the string in-place
  // CHECK-FIXES: s.erase(0, 4);
}

void LocalVariableNamedNpos() {
  std::string s = "hello world";

  // A local variable named 'npos' is not std::basic_string::npos; the count
  // is meaningful, so there is no single-'erase' rewrite and no warning.
  size_t npos = 3;
  s = s.substr(2, npos);
}

void ParamNamedNpos(std::string s, std::string::size_type npos) {
  s = s.substr(1, npos);
}

void TwoArgGeneral() {
  std::string s = "hello world";

  // General two-argument case: no diagnostic (no single erase() equivalent).
  s = s.substr(2, 3);

  int pos = 1;
  size_t len = 4;
  s = s.substr(pos, len);
}

void Negatives() {
  std::string s = "hello";
  std::string t = "world";

  // Different variables -- not a self-assignment.
  s = t.substr(1);
  s = t.substr(0, 3);

  // Not an assignment to the same variable.
  std::string r = s.substr(1);
}

// Unevaluated operands never materialize the temporary; no diagnostic.
std::string GlobalStr;
using UnevaluatedT = decltype(GlobalStr = GlobalStr.substr(1));

void UnevaluatedContexts() {
  std::string s = "hello";
  (void)sizeof(s = s.substr(1));
}

struct Holder {
  std::string Str;
  // Self-assignment through a class member is not matched; only plain
  // variables are.
  void trim() { Str = Str.substr(2); }
};

template <typename T>
void dependentType(T t) {
  // Type-dependent: no diagnostic, including in instantiations.
  t = t.substr(1);
}
void instantiate() { dependentType(std::string("hello")); }

void WideString() {
  std::wstring ws = L"hello world";

  ws = ws.substr(3);
  // CHECK-MESSAGES: [[@LINE-1]]:6: warning: inefficient self-assignment via 'substr'; use 'erase' to modify the string in-place
  // CHECK-FIXES: ws.erase(0, 3);

  // Truncation form: warning only, no fix-it.
  ws = ws.substr(0, 5);
  // CHECK-MESSAGES: [[@LINE-1]]:6: warning: inefficient self-assignment via 'substr'; use 'erase' to modify the string in-place
  // CHECK-FIXES: ws = ws.substr(0, 5);
}

void Parenthesized() {
  std::string s = "hello";

  // Parenthesized object on the RHS -- should still match.
  s = (s).substr(2);
  // CHECK-MESSAGES: [[@LINE-1]]:5: warning: inefficient self-assignment via 'substr'; use 'erase' to modify the string in-place
  // CHECK-FIXES: s.erase(0, 2);

  // Parenthesized substr call on the RHS.
  s = (s.substr(3));
  // CHECK-MESSAGES: [[@LINE-1]]:5: warning: inefficient self-assignment via 'substr'; use 'erase' to modify the string in-place
  // CHECK-FIXES: s.erase(0, 3);

  // Parenthesized left-hand side.
  (s) = s.substr(1);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: inefficient self-assignment via 'substr'; use 'erase' to modify the string in-place
  // CHECK-FIXES: s.erase(0, 1);
}

#define STRIP_PREFIX(str, n) str = str.substr(n)
void MacroExpansion() {
  std::string s = "hello";

  // Diagnosed, but no fix-it: rewriting a macro expansion is unsafe.
  STRIP_PREFIX(s, 2);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: inefficient self-assignment via 'substr'; use 'erase' to modify the string in-place
  // CHECK-FIXES: STRIP_PREFIX(s, 2);
}

#define OFFSET 2
void MacroArgument() {
  std::string s = "hello";

  // Only the argument comes from a macro; the fix-it preserves its spelling.
  s = s.substr(OFFSET);
  // CHECK-MESSAGES: [[@LINE-1]]:5: warning: inefficient self-assignment via 'substr'; use 'erase' to modify the string in-place
  // CHECK-FIXES: s.erase(0, OFFSET);
}
