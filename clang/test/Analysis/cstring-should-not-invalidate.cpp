// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.cstring,alpha.unix.cstring,debug.ExprInspection -verify %s

typedef decltype(sizeof(int)) size_t;
void clang_analyzer_eval(bool);

char *strncpy(char *dest, const char *src, size_t x);

constexpr int initB = 100;
struct Base {
  int b;
  Base(): b(initB) {}
};

// issue 143807
struct strncpyTestClass: public Base {
  int *m_ptr;
  char m_buff[1000];

  void KnownLen(char *src) {
    m_ptr = new int;
    strncpy(m_buff, src, sizeof(m_buff)); // known len but unknown src size
    delete m_ptr;                         // no warning
  }

  void KnownSrcLen(size_t n) {
    m_ptr = new int;
    strncpy(m_buff, "xyz", n); // known src size but unknown len
    delete m_ptr;              // no warning
  }
};

void strncpyTest(char *src, size_t n) {
  strncpyTestClass rep;
  rep.KnownLen(src);
  rep.KnownSrcLen(n);
  clang_analyzer_eval(rep.b == initB); // expected-warning{{TRUE}}
}

size_t strlcpy(char *dest, const char *src, size_t size);

struct strlcpyTestClass: public Base {
  int *m_ptr;
  char m_buff[1000];

  void KnownLen(char *src) {
    m_ptr = new int;
    strlcpy(m_buff, src, sizeof(m_buff)); // known len but unknown src size
    delete m_ptr;                         // no warning
  }

  void KnownSrcLen(size_t n) {
    m_ptr = new int;
    strlcpy(m_buff, "xyz", n); // known src size but unknown len
    delete m_ptr;              // no warning
  }
};

void strlcpyTest(char *src, size_t n) {
  strlcpyTestClass rep;
  rep.KnownLen(src);
  rep.KnownSrcLen(n);
  clang_analyzer_eval(rep.b == initB); // expected-warning{{TRUE}}
}

char *strncat(char *s1, const char *s2, size_t n);

struct strncatTestClass: public Base {
  int *m_ptr;
  char m_buff[1000];

  void KnownLen(char *src) {
    m_ptr = new int;
    strncat(m_buff, src, sizeof(m_buff) - 1); // known len but unknown src size
    delete m_ptr;                         // no warning
  }

  void KnownSrcLen(size_t n) {
    m_ptr = new int;
    strncat(m_buff, "xyz", n); // known src size but unknown len
    delete m_ptr;              // no warning
  }
};

void strncatTest(char *src, size_t n) {
  strncatTestClass rep;
  rep.KnownLen(src);
  rep.KnownSrcLen(n);
  clang_analyzer_eval(rep.b == initB); // expected-warning{{TRUE}}
}

struct strncatReportOutOfBoundTestClass {
  int *m_ptr;
  char m_buff[1000];

  void KnownLen(char *src) {
    m_ptr = new int;
    // expected-warning@+1{{String concatenation function overflows the destination buffer}}
    strncat(m_buff, src, sizeof(m_buff)); // known len but unknown src size
    delete m_ptr;                         // no warning
  }
};

void strncatReportOutOfBoundTest(char *src, size_t n) {
  strncatReportOutOfBoundTestClass rep;
  rep.KnownLen(src);
}

size_t strlcat(char *dst, const char *src, size_t size);

struct strlcatTestClass: public Base {
  int *m_ptr;
  char m_buff[1000];

  void KnownLen(char *src) {
    m_ptr = new int;
    strlcat(m_buff, src, sizeof(m_buff)); // known len but unknown src size
    delete m_ptr;                         // no warning
  }

  void KnownSrcLen(size_t n) {
    m_ptr = new int;
    strlcat(m_buff, "xyz", n); // known src size but unknown len
    delete m_ptr;              // no warning
  }
};

void strlcatTest(char *src, size_t n) {
  strlcatTestClass rep;
  rep.KnownLen(src);
  rep.KnownSrcLen(n);
  clang_analyzer_eval(rep.b == initB); // expected-warning{{TRUE}}
}
