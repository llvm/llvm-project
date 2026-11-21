// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=core,cplusplus,debug.DebugIteratorModeling,debug.ExprInspection -analyzer-config aggressive-binary-operation-simplification=true -analyzer-config display-checker-name=false -verify %s

template <typename Container>
long clang_analyzer_container_begin(const Container &);
template <typename Container>
long clang_analyzer_container_end(const Container &);
template <typename Iterator>
long clang_analyzer_iterator_position(const Iterator &);
void clang_analyzer_denote(long, const char *);
void clang_analyzer_express(long);

// The member operators only make IteratorModeling recognize the type as an
// iterator. The test calls the free operators explicitly so that the
// CXXInstanceCall path is not taken.
struct MyIterator {
  MyIterator(const MyIterator &) {}
  MyIterator &operator=(const MyIterator &) { return *this; }
  ~MyIterator() {}
  MyIterator &operator++();
  MyIterator operator++(int);
  int operator*() const;
};

struct MyContainer {
  MyIterator begin() const;
  MyIterator end() const;
};

MyIterator &operator++(MyIterator &I) { return I; }
MyIterator operator++(MyIterator &I, int) { return I; }
MyIterator &operator--(MyIterator &I) { return I; }
MyIterator operator--(MyIterator &I, int) { return I; }

void prefix_increment(const MyContainer &c) {
  auto i = c.begin();
  clang_analyzer_denote(clang_analyzer_container_begin(c), "$c.begin()");

  operator++(i);

  clang_analyzer_express(clang_analyzer_iterator_position(i)); // expected-warning-re {{$c.begin() + 1{{$}}}}
}

void postfix_increment(const MyContainer &c) {
  auto i = c.begin();
  clang_analyzer_denote(clang_analyzer_container_begin(c), "$c.begin()");

  operator++(i, 0);

  clang_analyzer_express(clang_analyzer_iterator_position(i)); // expected-warning-re {{$c.begin() + 1{{$}}}}
}

void prefix_decrement(const MyContainer &c) {
  auto i = c.end();
  clang_analyzer_denote(clang_analyzer_container_end(c), "$c.end()");

  operator--(i);

  clang_analyzer_express(clang_analyzer_iterator_position(i)); // expected-warning-re {{$c.end() - 1{{$}}}}
}

void postfix_decrement(const MyContainer &c) {
  auto i = c.end();
  clang_analyzer_denote(clang_analyzer_container_end(c), "$c.end()");

  operator--(i, 0);

  clang_analyzer_express(clang_analyzer_iterator_position(i)); // expected-warning-re {{$c.end() - 1{{$}}}}
}
