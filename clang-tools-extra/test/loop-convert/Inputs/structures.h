#ifndef _LLVM_TOOLS_CLANG_TOOLS_TESTS_TOOLING_STRUCTURES_H_
#define _LLVM_TOOLS_CLANG_TOOLS_TESTS_TOOLING_STRUCTURES_H_

extern "C" {
extern int printf(const char *restrict, ...);
}

struct Val {int x; void g(); };

struct MutableVal {
  void constFun(int) const;
  void nonConstFun(int, int);
  void constFun(MutableVal &) const;
  void constParamFun(const MutableVal &) const;
  void nonConstParamFun(const MutableVal &);
  int x;
};

struct S {
  typedef MutableVal *iterator;
  typedef const MutableVal *const_iterator;
  const_iterator begin() const;
  const_iterator end() const;
  iterator begin();
  iterator end();
};

struct T {
  struct iterator {
    int& operator*();
    const int& operator*()const;
    iterator& operator ++();
    bool operator!=(const iterator &other);
    void insert(int);
    int x;
  };
  iterator begin();
  iterator end();
};

struct U {
  struct iterator {
    Val& operator*();
    const Val& operator*()const;
    iterator& operator ++();
    bool operator!=(const iterator &other);
    Val *operator->();
  };
  iterator begin();
  iterator end();
  int x;
};

struct X {
  S s;
  T t;
  U u;
  S getS();
};

template<typename ElemType>
class dependent{
 public:
  struct iterator_base {
    const ElemType& operator*()const;
    iterator_base& operator ++();
    bool operator!=(const iterator_base &other) const;
    const ElemType *operator->() const;
  };

  struct iterator : iterator_base {
    ElemType& operator*();
    iterator& operator ++();
    ElemType *operator->();
  };

  typedef iterator_base const_iterator;
  const_iterator begin() const;
  const_iterator end() const;
  iterator begin();
  iterator end();
  unsigned size() const;
  ElemType & operator[](unsigned);
  const ElemType & operator[](unsigned) const;
  ElemType & at(unsigned);
  const ElemType & at(unsigned) const;

  // Intentionally evil.
  dependent<ElemType> operator*();

  void foo();
  void constFoo() const;
};

template<typename First, typename Second>
class doublyDependent{
 public:
  struct Value {
    First first;
    Second second;
  };

  struct iterator_base {
    const Value& operator*()const;
    iterator_base& operator ++();
    bool operator!=(const iterator_base &other) const;
    const Value *operator->() const;
  };

  struct iterator : iterator_base {
    Value& operator*();
    Value& operator ++();
    Value *operator->();
  };

  typedef iterator_base const_iterator;
  const_iterator begin() const;
  const_iterator end() const;
  iterator begin();
  iterator end();
};

template<typename Contained>
class transparent {
 public:
  Contained *at();
  Contained *operator->();
  Contained operator*();
};

template<typename IteratorType>
struct Nested {
  typedef IteratorType* iterator;
  IteratorType *operator->();
  IteratorType operator*();
  iterator begin();
  iterator end();
};

#endif  // _LLVM_TOOLS_CLANG_TOOLS_TESTS_TOOLING_STRUCTURES_H_
