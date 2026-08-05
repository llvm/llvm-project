// RUN: rm -fR %t
// RUN: mkdir %t
// RUN: %clang_analyze_cc1 -analyzer-checker=webkit.NoUncountedMemberChecker \
// RUN:   -analyzer-output=html -o %t %s
// RUN: ls %t | grep report | count 2

// Two member variables with identical spelling in different classes
// must not collide in the HTML issue hash: the enclosing class
// differs.

class Info {
public:
  void ref() const;
  void deref() const;
};

class A {
public:
  A(Info& info) : m_info(info) { }

private:
  Info& m_info;
};

class B {
public:
  B(Info& info) : m_info(info) { }

private:
  Info& m_info;
};
