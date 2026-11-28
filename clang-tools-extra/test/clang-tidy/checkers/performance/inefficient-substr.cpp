// RUN: %check_clang_tidy %s performance-inefficient-substr %t
#include <string>

void AppendForm() {
  std::string s = "hello";
  std::string t = "world wide";

  s += t.substr(5);
  // CHECK-MESSAGES: [[@LINE-1]]:5: warning: inefficient concatenation via 'substr' temporary; use 'append' to avoid the temporary string [performance-inefficient-substr]
  // CHECK-FIXES: s.append(t, 5);

  s += t.substr(5, 3);
  // CHECK-MESSAGES: [[@LINE-1]]:5: warning: inefficient concatenation via 'substr'
  // CHECK-FIXES: s.append(t, 5, 3);

  // Expression arguments pass through verbatim.
  int pos = 1;
  int len = 4;
  s += t.substr(pos + 1, len - 2);
  // CHECK-MESSAGES: [[@LINE-1]]:5: warning: inefficient concatenation via 'substr'
  // CHECK-FIXES: s.append(t, pos + 1, len - 2);

  // npos passes through verbatim: append clamps it identically.
  s += t.substr(2, std::string::npos);
  // CHECK-MESSAGES: [[@LINE-1]]:5: warning: inefficient concatenation via 'substr'
  // CHECK-FIXES: s.append(t, 2, std::string::npos);

  const std::string c = "const source";
  s += c.substr(3);
  // CHECK-MESSAGES: [[@LINE-1]]:5: warning: inefficient concatenation via 'substr'
  // CHECK-FIXES: s.append(c, 3);

  // Parenthesized receiver and parenthesized call.
  s += (t).substr(2);
  // CHECK-MESSAGES: [[@LINE-1]]:5: warning: inefficient concatenation via 'substr'
  // CHECK-FIXES: s.append(t, 2);

  s += (t.substr(2));
  // CHECK-MESSAGES: [[@LINE-1]]:5: warning: inefficient concatenation via 'substr'
  // CHECK-FIXES: s.append(t, 2);
}

void AssignForm() {
  std::string dst;
  std::string src = "hello world";

  dst = src.substr(2);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: inefficient assignment via 'substr' temporary; use 'assign' to avoid the temporary string [performance-inefficient-substr]
  // CHECK-FIXES: dst.assign(src, 2);

  dst = src.substr(2, 3);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: inefficient assignment via 'substr'
  // CHECK-FIXES: dst.assign(src, 2, 3);
}

void SelfAssign() {
  std::string s = "hello";

  // Same-variable assignment is excluded: an in-place 'erase' rewrite is
  // strictly better than a self-aliasing 'assign'; this check stays silent.
  s = s.substr(2);
}

void SelfAppend() {
  std::string s = "hello";

  // Diagnosed, but not rewritten: the replacement would introduce a
  // self-aliasing append(s, ...) call.
  s += s.substr(3);
  // CHECK-MESSAGES: [[@LINE-1]]:5: warning: inefficient concatenation via 'substr'
  // CHECK-FIXES: s += s.substr(3);
}

void ZeroArg() {
  std::string s = "hello";
  std::string t = "world";

  // s += t.substr() is just s += t in disguise; no diagnostic.
  s += t.substr();
}

struct MyString {
  MyString substr(unsigned pos) const;
  MyString &operator+=(const MyString &);
  MyString &operator=(const MyString &);
};

void NotStringLike(MyString a, MyString b) {
  // Not in StringLikeClasses; no diagnostic.
  a += b.substr(1);
  a = b.substr(1);
}

struct Derived : std::string {};

void DerivedReceiver(std::string s, Derived d) {
  // Receiver type differs from the destination type; the check is
  // conservative and stays silent.
  s += d.substr(1);
}

struct Holder {
  std::string S;
  void add(const std::string &t) {
    // Class members are not matched; only plain variables are.
    S += t.substr(1);
  }
};

void MemberSource(Holder h) {
  std::string s;
  s += h.S.substr(1);
}

template <typename T>
void dependentType(T a, T b) {
  // Type-dependent: no diagnostic, including in instantiations.
  a += b.substr(1);
}
void instantiate() {
  dependentType(std::string("hello"), std::string("world"));
}

void Initialization(std::string s) {
  // Initializations are copy-elided since C++17; nothing to save.
  std::string t = s.substr(1);
  std::string u(s.substr(1));
}

void WideString() {
  std::wstring wd;
  std::wstring ws = L"hello world";

  wd += ws.substr(1);
  // CHECK-MESSAGES: [[@LINE-1]]:6: warning: inefficient concatenation via 'substr'
  // CHECK-FIXES: wd.append(ws, 1);

  wd = ws.substr(1);
  // CHECK-MESSAGES: [[@LINE-1]]:6: warning: inefficient assignment via 'substr'
  // CHECK-FIXES: wd.assign(ws, 1);
}

#define APPEND_TAIL(a, b, n) a += b.substr(n)
void MacroExpansion() {
  std::string s = "hello";
  std::string t = "world";

  // Diagnosed, but no fix-it: rewriting a macro expansion is unsafe.
  APPEND_TAIL(s, t, 2);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: inefficient concatenation via 'substr'
  // CHECK-FIXES: APPEND_TAIL(s, t, 2);
}

#define OFFSET 2
void MacroArgument() {
  std::string s = "hello";
  std::string t = "world";

  // Only the argument comes from a macro; the fix-it preserves its spelling.
  s += t.substr(OFFSET);
  // CHECK-MESSAGES: [[@LINE-1]]:5: warning: inefficient concatenation via 'substr'
  // CHECK-FIXES: s.append(t, OFFSET);
}
