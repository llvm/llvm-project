// RUN: %check_clang_tidy %s readability-redundant-access-specifiers %t

class FooPublic {
public:
  int a;
public: // comment-0
  // CHECK-MESSAGES: :[[@LINE-1]]:1: warning: redundant access specifier has the same accessibility as the previous access specifier [readability-redundant-access-specifiers]
  // CHECK-MESSAGES: :[[@LINE-4]]:1: note: previously declared here
  // CHECK-FIXES: {{^}}// comment-0{{$}}
  int b;
private:
  int c;
};

struct StructPublic {
public:
  int a;
public: // comment-1
  // CHECK-MESSAGES: :[[@LINE-1]]:1: warning: redundant access specifier has the same accessibility as the previous access specifier [readability-redundant-access-specifiers]
  // CHECK-MESSAGES: :[[@LINE-4]]:1: note: previously declared here
  // CHECK-FIXES: {{^}}// comment-1{{$}}
  int b;
private:
  int c;
};

union UnionPublic {
public:
  int a;
public: // comment-2
  // CHECK-MESSAGES: :[[@LINE-1]]:1: warning: redundant access specifier has the same accessibility as the previous access specifier [readability-redundant-access-specifiers]
  // CHECK-MESSAGES: :[[@LINE-4]]:1: note: previously declared here
  // CHECK-FIXES: {{^}}// comment-2{{$}}
  int b;
private:
  int c;
};

class FooProtected {
protected:
  int a;
protected: // comment-3
  // CHECK-MESSAGES: :[[@LINE-1]]:1: warning: redundant access specifier has the same accessibility as the previous access specifier [readability-redundant-access-specifiers]
  // CHECK-MESSAGES: :[[@LINE-4]]:1: note: previously declared here
  // CHECK-FIXES: {{^}}// comment-3{{$}}
  int b;
private:
  int c;
};

class FooPrivate {
private:
  int a;
private: // comment-4
  // CHECK-MESSAGES: :[[@LINE-1]]:1: warning: redundant access specifier has the same accessibility as the previous access specifier [readability-redundant-access-specifiers]
  // CHECK-MESSAGES: :[[@LINE-4]]:1: note: previously declared here
  // CHECK-FIXES: {{^}}// comment-4{{$}}
  int b;
public:
  int c;
};

class FooMacro {
private:
  int a;
#if defined(ZZ)
  public:
  int b;
#endif
private: // comment-5
  // CHECK-MESSAGES: :[[@LINE-1]]:1: warning: redundant access specifier has the same accessibility as the previous access specifier [readability-redundant-access-specifiers]
  // CHECK-MESSAGES: :[[@LINE-8]]:1: note: previously declared here
  // CHECK-FIXES: {{^}}// comment-5{{$}}
  int c;
protected:
  int d;
public:
  int e;
};

class Valid {
private:
  int a;
public:
  int b;
private:
  int c;
protected:
  int d;
public:
  int e;
};

class ValidInnerClass {
public:
  int a;

  class Inner {
  public:
    int b;
  };
};

#define MIXIN private: int b;

class ValidMacro {
private:
  int a;
MIXIN
private:
  int c;
protected:
  int d;
public:
  int e;
};
