// RUN: %clang_analyze_cc1 -analyzer-checker=core,optin.cplusplus.UninitializedObject \
// RUN:   -analyzer-config optin.cplusplus.UninitializedObject:Pedantic=true -DPEDANTIC \
// RUN:   -analyzer-config optin.cplusplus.UninitializedObject:IgnoreGuardedFields=true \
// RUN:   -std=c++11 -verify  %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,optin.cplusplus.UninitializedObject \
// RUN:   -analyzer-config optin.cplusplus.UninitializedObject:Pedantic=true -DPEDANTIC \
// RUN:   -analyzer-config optin.cplusplus.UninitializedObject:IgnoreGuardedFields=true \
// RUN:   -std=c++11 -verify  %s -DHEAP_ALLOCATION

#ifdef HEAP_ALLOCATION
#define INIT(CLS, ARGS) new CLS ARGS
#else
#define INIT(CLS, ARGS) (void) CLS ARGS
#endif

//===----------------------------------------------------------------------===//
// Helper functions for tests.
//===----------------------------------------------------------------------===//

[[noreturn]] void halt();

void assert(int b) {
  if (!b)
    halt();
}

int rand();

//===----------------------------------------------------------------------===//
// Tests for fields properly guarded by asserts.
//===----------------------------------------------------------------------===//

class NoUnguardedFieldsTest {
public:
  enum Kind {
    V,
    A
  };

private:
  int Volume, Area;
  Kind K;

public:
  NoUnguardedFieldsTest(Kind K) : K(K) {
    switch (K) {
    case V:
      Volume = 0;
      break;
    case A:
      Area = 0;
      break;
    }
  }

  void operator-() {
    assert(K == Kind::A);
    (void)Area;
  }

  void operator+() {
    assert(K == Kind::V);
    (void)Volume;
  }
};

void fNoUnguardedFieldsTest() {
  INIT(NoUnguardedFieldsTest, (NoUnguardedFieldsTest::Kind::A));
  INIT(NoUnguardedFieldsTest, (NoUnguardedFieldsTest::Kind::V));
}

class NoUngardedFieldsNoReturnFuncCalledTest {
public:
  enum Kind {
    V,
    A
  };

private:
  int Volume, Area;
  Kind K;

public:
  NoUngardedFieldsNoReturnFuncCalledTest(Kind K) : K(K) {
    switch (K) {
    case V:
      Volume = 0;
      break;
    case A:
      Area = 0;
      break;
    }
  }

  void operator-() {
    halt();
    (void)Area;
  }

  void operator+() {
    halt();
    (void)Volume;
  }
};

void fNoUngardedFieldsNoReturnFuncCalledTest() {
  INIT(NoUngardedFieldsNoReturnFuncCalledTest, (NoUngardedFieldsNoReturnFuncCalledTest::Kind::A));
  INIT(NoUngardedFieldsNoReturnFuncCalledTest, (NoUngardedFieldsNoReturnFuncCalledTest::Kind::V));
}

class NoUnguardedFieldsWithUndefMethodTest {
public:
  enum Kind {
    V,
    A
  };

private:
  int Volume, Area;
  Kind K;

public:
  NoUnguardedFieldsWithUndefMethodTest(Kind K) : K(K) {
    switch (K) {
    case V:
      Volume = 0;
      break;
    case A:
      Area = 0;
      break;
    }
  }

  void operator-() {
    assert(K == Kind::A);
    (void)Area;
  }

  void operator+() {
    assert(K == Kind::V);
    (void)Volume;
  }

  // We're checking method definitions for guards, so this is a no-crash test
  // whether we handle methods without definitions.
  void methodWithoutDefinition();
};

void fNoUnguardedFieldsWithUndefMethodTest() {
  INIT(NoUnguardedFieldsWithUndefMethodTest, (NoUnguardedFieldsWithUndefMethodTest::Kind::A));
  INIT(NoUnguardedFieldsWithUndefMethodTest, (NoUnguardedFieldsWithUndefMethodTest::Kind::V));
}

class UnguardedFieldThroughMethodTest {
public:
  enum Kind {
    V,
    A
  };

private:
  int Volume, Area; // expected-note {{uninitialized field 'this->Volume'}}
  Kind K;

public:
  UnguardedFieldThroughMethodTest(Kind K) : K(K) {
    switch (K) {
    case V:
      Volume = 0;
      break;
    case A:
      Area = 0;
      break; // expected-warning {{1 uninitialized field}}
    }
  }

  void operator-() {
    assert(K == Kind::A);
    (void)Area;
  }

  void operator+() {
    (void)Volume;
  }
};

void fUnguardedFieldThroughMethodTest() {
  INIT(UnguardedFieldThroughMethodTest, (UnguardedFieldThroughMethodTest::Kind::A));
}

class UnguardedPublicFieldsTest {
public:
  enum Kind {
    V,
    A
  };

public:
  // Note that fields are public.
  int Volume, Area; // expected-note {{uninitialized field 'this->Volume'}}
  Kind K;

public:
  UnguardedPublicFieldsTest(Kind K) : K(K) {
    switch (K) {
    case V:
      Volume = 0;
      break;
    case A:
      Area = 0;
      break; // expected-warning {{1 uninitialized field}}
    }
  }

  void operator-() {
    assert(K == Kind::A);
    (void)Area;
  }

  void operator+() {
    assert(K == Kind::V);
    (void)Volume;
  }
};

void fUnguardedPublicFieldsTest() {
  INIT(UnguardedPublicFieldsTest, (UnguardedPublicFieldsTest::Kind::A));
}

//===----------------------------------------------------------------------===//
// Highlights of some false negatives due to syntactic checking.
//===----------------------------------------------------------------------===//

class UnguardedFalseNegativeTest1 {
public:
  enum Kind {
    V,
    A
  };

private:
  int Volume, Area;
  Kind K;

public:
  UnguardedFalseNegativeTest1(Kind K) : K(K) {
    switch (K) {
    case V:
      Volume = 0;
      break;
    case A:
      Area = 0;
      break;
    }
  }

  void operator-() {
    if (rand())
      assert(K == Kind::A);
    (void)Area;
  }

  void operator+() {
    if (rand())
      assert(K == Kind::V);
    (void)Volume;
  }
};

void fUnguardedFalseNegativeTest1() {
  INIT(UnguardedFalseNegativeTest1, (UnguardedFalseNegativeTest1::Kind::A));
}

class UnguardedFalseNegativeTest2 {
public:
  enum Kind {
    V,
    A
  };

private:
  int Volume, Area;
  Kind K;

public:
  UnguardedFalseNegativeTest2(Kind K) : K(K) {
    switch (K) {
    case V:
      Volume = 0;
      break;
    case A:
      Area = 0;
      break;
    }
  }

  void operator-() {
    assert(rand());
    (void)Area;
  }

  void operator+() {
    assert(rand());
    (void)Volume;
  }
};

void fUnguardedFalseNegativeTest2() {
  INIT(UnguardedFalseNegativeTest2, (UnguardedFalseNegativeTest2::Kind::A));
}

//===----------------------------------------------------------------------===//
// Tests for other guards. These won't be as thorough, as other guards are
// matched the same way as asserts, so if they are recognized, they are expected
// to work as well as asserts do.
//
// None of these tests expect warnings, since the flag works correctly if these
// fields are regarded properly guarded.
//===----------------------------------------------------------------------===//

class IfGuardedFieldsTest {
public:
  enum Kind {
    V,
    A
  };

private:
  int Volume, Area;
  Kind K;

public:
  IfGuardedFieldsTest(Kind K) : K(K) {
    switch (K) {
    case V:
      Volume = 0;
      break;
    case A:
      Area = 0;
      break;
    }
  }

  void operator-() {
    if (K != Kind::A)
      return;
    (void)Area;
  }

  void operator+() {
    if (K != Kind::V)
      return;
    (void)Volume;
  }
};

void fIfGuardedFieldsTest() {
  INIT(IfGuardedFieldsTest, (IfGuardedFieldsTest::Kind::A));
  INIT(IfGuardedFieldsTest, (IfGuardedFieldsTest::Kind::V));
}

class SwitchGuardedFieldsTest {
public:
  enum Kind {
    V,
    A
  };

private:
  int Volume, Area;
  Kind K;

public:
  SwitchGuardedFieldsTest(Kind K) : K(K) {
    switch (K) {
    case V:
      Volume = 0;
      break;
    case A:
      Area = 0;
      break;
    }
  }

  int operator-() {
    switch (K) {
    case Kind::A:
      return Area;
    case Kind::V:
      return -1;
    }
  }

  int operator+() {
    switch (K) {
    case Kind::A:
      return Area;
    case Kind::V:
      return -1;
    }
  }
};

void fSwitchGuardedFieldsTest() {
  INIT(SwitchGuardedFieldsTest, (SwitchGuardedFieldsTest::Kind::A));
  INIT(SwitchGuardedFieldsTest, (SwitchGuardedFieldsTest::Kind::V));
}

class ConditionalOperatorGuardedFieldsTest {
public:
  enum Kind {
    V,
    A
  };

private:
  int Volume, Area;
  Kind K;

public:
  ConditionalOperatorGuardedFieldsTest(Kind K) : K(K) {
    switch (K) {
    case V:
      Volume = 0;
      break;
    case A:
      Area = 0;
      break;
    }
  }

  int operator-() {
    return K == Kind::A ? Area : -1;
  }

  int operator+() {
    return K == Kind::V ? Volume : -1;
  }
};

void fConditionalOperatorGuardedFieldsTest() {
  INIT(ConditionalOperatorGuardedFieldsTest, (ConditionalOperatorGuardedFieldsTest::Kind::A));
  INIT(ConditionalOperatorGuardedFieldsTest, (ConditionalOperatorGuardedFieldsTest::Kind::V));
}
