// RUN: %check_clang_tidy %s bugprone-copy-constructor-init %t

class NonCopyable {
public:
  NonCopyable() = default;
  NonCopyable(const NonCopyable &) = delete;

private:
  int a;
};

class NonCopyable2 {
public:
  NonCopyable2() = default;

private:
  NonCopyable2(const NonCopyable2 &);
  int a;
};

class Copyable {
public:
  Copyable() = default;
  Copyable(const Copyable &) = default;

private:
  int a;
};

class Copyable2 {
public:
  Copyable2() = default;
  Copyable2(const Copyable2 &) = default;

private:
  int a;
};

class Copyable3 : public Copyable {
public:
  Copyable3() = default;
  Copyable3(const Copyable3 &) = default;
};

template <class C>
class Copyable4 {
public:
  Copyable4() = default;
  Copyable4(const Copyable4 &) = default;

private:
  int a;
};

template <class T, class S>
class Copyable5 {
public:
  Copyable5() = default;
  Copyable5(const Copyable5 &) = default;

private:
  int a;
};

class EmptyCopyable {
public:
  EmptyCopyable() = default;
  EmptyCopyable(const EmptyCopyable &) = default;
};

template <typename T>
using CopyableAlias = Copyable5<T, int>;

typedef Copyable5<int, int> CopyableAlias2;

class X : public Copyable, public EmptyCopyable {
  X(const X &other) : Copyable(other) {}
};

class X2 : public Copyable2 {
  X2(const X2 &other) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: calling a base constructor other than the copy constructor [bugprone-copy-constructor-init]
  // CHECK-FIXES: X2(const X2 &other)  : Copyable2(other) {}
};

class X2_A : public Copyable2 {
  X2_A(const X2_A &) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: calling a base constructor
  // CHECK-FIXES: X2_A(const X2_A &) {}
};

class X3 : public Copyable, public Copyable2 {
  X3(const X3 &other) : Copyable(other) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: calling a base constructor
  // CHECK-FIXES: X3(const X3 &other) : Copyable(other) {}
};

class X4 : public Copyable {
  X4(const X4 &other) : Copyable() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: calling a base constructor
  // CHECK-FIXES: X4(const X4 &other) : Copyable(other) {}
};

class X5 : public Copyable3 {
  X5(const X5 &other) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: calling a base constructor
  // CHECK-FIXES: X5(const X5 &other)  : Copyable3(other) {}
};

class X6 : public Copyable2, public Copyable3 {
  X6(const X6 &other) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: calling a base constructor
  // CHECK-FIXES: X6(const X6 &other)  : Copyable2(other), Copyable3(other) {}
};

class X7 : public Copyable, public Copyable2 {
  X7(const X7 &other) : Copyable() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: calling a base constructor
  // CHECK-FIXES: X7(const X7 &other) : Copyable(other) {}
};

class X8 : public Copyable4<int> {
  X8(const X8 &other) : Copyable4(other) {}
};

class X9 : public Copyable4<int> {
  X9(const X9 &other) : Copyable4() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: calling a base constructor
  // CHECK-FIXES: X9(const X9 &other) : Copyable4(other) {}
};

class X10 : public Copyable4<int> {
  X10(const X10 &other) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: calling a base constructor
  // CHECK-FIXES: X10(const X10 &other)  : Copyable4(other) {}
};

class X11 : public Copyable5<int, float> {
  X11(const X11 &other) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: calling a base constructor
  // CHECK-FIXES: X11(const X11 &other)  : Copyable5(other) {}
};

class X12 : public CopyableAlias<float> {
  X12(const X12 &other) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: calling a base constructor
};

template <typename T>
class X13 : T {
  X13(const X13 &other) {}
};

template class X13<EmptyCopyable>;
template class X13<Copyable>;

#define FROMMACRO                \
  class X14 : public Copyable2 { \
    X14(const X14 &other) {}     \
  };

FROMMACRO
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: calling a base constructor

class X15 : public CopyableAlias2 {
  X15(const X15 &other) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: calling a base constructor
  // CHECK-FIXES: X15(const X15 &other)  : Copyable5(other) {}
};

class X16 : public NonCopyable {
  X16(const X16 &other) {}
};

class X17 : public NonCopyable2 {
  X17(const X17 &other) {}
};

class X18 : private Copyable {
  X18(const X18 &other) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: calling a base constructor
  // CHECK-FIXES: X18(const X18 &other)  : Copyable(other) {}
};

class X19 : private Copyable {
  X19(const X19 &other) : Copyable(other) {}
};
