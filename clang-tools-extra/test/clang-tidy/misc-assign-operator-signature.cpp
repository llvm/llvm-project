// RUN: %check_clang_tidy %s misc-assign-operator-signature %t

struct Good {
  Good& operator=(const Good&);
  Good& operator=(Good&&);

  // Assign from other types is fine too.
  Good& operator=(int);
};

struct AlsoGood {
  // By value is also fine.
  AlsoGood& operator=(AlsoGood);
};

struct BadReturn {
  void operator=(const BadReturn&);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: operator=() should return 'BadReturn&' [misc-assign-operator-signature]
  const BadReturn& operator=(BadReturn&&);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: operator=() should return 'Bad
  void operator=(int);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: operator=() should return 'Bad
};
struct BadReturn2 {
  BadReturn2&& operator=(const BadReturn2&);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: operator=() should return 'Bad
  int operator=(BadReturn2&&);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: operator=() should return 'Bad
};

struct BadArgument {
  BadArgument& operator=(BadArgument&);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: operator=() should take 'BadArgument const&', 'BadArgument&&' or 'BadArgument'
  BadArgument& operator=(const BadArgument&&);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: operator=() should take 'BadAr
};

struct BadModifier {
  BadModifier& operator=(const BadModifier&) const;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: operator=() should not be marked 'const'
};

struct Deleted {
  // We don't check the return value of deleted operators.
  void operator=(const Deleted&) = delete;
  void operator=(Deleted&&) = delete;
};

class Private {
  // We don't check the return value of private operators.
  // Pre-C++11 way of disabling assignment.
  void operator=(const Private &);
};

struct Virtual {
  virtual Virtual& operator=(const Virtual &);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: operator=() should not be marked 'virtual'
};
