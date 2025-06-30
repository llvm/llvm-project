// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection
// -analyzer-config c++-inlining=constructors -verify %s

// expected-no-diagnostics

typedef unsigned int size_t;

char *strncpy(char *dest, const char *src, size_t x);

// issue 143807
struct strncpyTestClass {
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
}

size_t strlcpy(char *dest, const char *src, size_t size);

struct strlcpyTestClass {
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
}

char *strncat(char *s1, const char *s2, size_t n);

struct strncatTestClass {
  int *m_ptr;
  char m_buff[1000];

  void KnownLen(char *src) {
    m_ptr = new int;
    strncat(m_buff, src, sizeof(m_buff)); // known len but unknown src size
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
}

size_t strlcat(char *dst, const char *src, size_t size);

struct strlcatTestClass {
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
}
