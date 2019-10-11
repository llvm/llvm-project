#ifndef STRUCTURES_H
#define STRUCTURES_H

extern "C" {
extern int printf(const char *restrict, ...);
}

struct Val {int X; void g(); };

struct MutableVal {
  void constFun(int) const;
  void nonConstFun(int, int);
  void constFun(MutableVal &) const;
  void constParamFun(const MutableVal &) const;
  void nonConstParamFun(const MutableVal &);
  int X;
};

struct NonTriviallyCopyable {
  NonTriviallyCopyable() = default;
  // Define this constructor to make this class non-trivially copyable.
  NonTriviallyCopyable(const NonTriviallyCopyable& Ntc);
  int X;
};

struct TriviallyCopyableButBig {
  int X;
  char Array[16];
};

struct S {
  typedef MutableVal *iterator;
  typedef const MutableVal *const_iterator;
  const_iterator begin() const;
  const_iterator end() const;
  const_iterator cbegin() const;
  const_iterator cend() const;
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
    int X;
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
  int X;
};

struct X {
  S Ss;
  T Tt;
  U Uu;
  S getS();
};

template<typename ElemType>
class dependent {
 public:
  dependent<ElemType>();
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
  ElemType & at(unsigned, unsigned);
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
  typedef const IteratorType* const_iterator;
  IteratorType *operator->();
  IteratorType operator*();
  iterator begin();
  iterator end();
  const_iterator begin() const;
  const_iterator end() const;
};

// Like llvm::SmallPtrSet, the iterator has a dereference operator that returns
// by value instead of by reference.
template <typename T>
struct PtrSet {
  struct iterator {
    bool operator!=(const iterator &other) const;
    const T operator*();
    iterator &operator++();
  };
  iterator begin() const;
  iterator end() const;
};

template <typename T>
struct TypedefDerefContainer {
  struct iterator {
    typedef T &deref_type;
    bool operator!=(const iterator &other) const;
    deref_type operator*();
    iterator &operator++();
  };
  iterator begin() const;
  iterator end() const;
};

template <typename T>
struct RValueDerefContainer {
  struct iterator {
    typedef T &&deref_type;
    bool operator!=(const iterator &other) const;
    deref_type operator*();
    iterator &operator++();
  };
  iterator begin() const;
  iterator end() const;
};

namespace Macros {

struct MacroStruct {
  int Arr[10];
};
static MacroStruct *MacroSt;
#define CONT MacroSt->

} // namespace Macros

#endif  // STRUCTURES_H
