// RUN: %clang_cc1 -fsyntax-only -std=c++11 -Wloop-analysis -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++11 -Wrange-loop-analysis -verify %s

void test_POD_64_bytes() {
  struct Record {
    char a[64];
  };

  Record records[8];
  for (const auto r : records)
    (void)r;
}

void test_POD_65_bytes() {
  struct Record {
    char a[65];
  };

  // expected-warning@+3 {{loop variable 'r' creates a copy from type 'const Record'}}
  // expected-note@+2 {{use reference type 'const Record &' to prevent copying}}
  Record records[8];
  for (const auto r : records)
    (void)r;
}

void test_TriviallyCopyable_64_bytes() {
  struct Record {
    Record() {}
    char a[64];
  };

  Record records[8];
  for (const auto r : records)
    (void)r;
}

void test_TriviallyCopyable_65_bytes() {
  struct Record {
    Record() {}
    char a[65];
  };

  // expected-warning@+3 {{loop variable 'r' creates a copy from type 'const Record'}}
  // expected-note@+2 {{use reference type 'const Record &' to prevent copying}}
  Record records[8];
  for (const auto r : records)
    (void)r;
}

void test_NonTriviallyCopyable() {
  struct Record {
    Record() {}
    ~Record() {}
    volatile int a;
    int b;
  };

  // expected-warning@+3 {{loop variable 'r' creates a copy from type 'const Record'}}
  // expected-note@+2 {{use reference type 'const Record &' to prevent copying}}
  Record records[8];
  for (const auto r : records)
    (void)r;
}

void test_TrivialABI_64_bytes() {
  struct [[clang::trivial_abi]] Record {
    Record() {}
    ~Record() {}
    char a[64];
  };

  Record records[8];
  for (const auto r : records)
    (void)r;
}

void test_TrivialABI_65_bytes() {
  struct [[clang::trivial_abi]] Record {
    Record() {}
    ~Record() {}
    char a[65];
  };

  // expected-warning@+3 {{loop variable 'r' creates a copy from type 'const Record'}}
  // expected-note@+2 {{use reference type 'const Record &' to prevent copying}}
  Record records[8];
  for (const auto r : records)
    (void)r;
}
