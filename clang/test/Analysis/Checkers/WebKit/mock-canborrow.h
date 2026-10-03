#ifndef mock_canborrow_h
#define mock_canborrow_h

#define LIFETIME_BOUND [[clang::lifetimebound]]

class CanBorrow {
public:
  ~CanBorrow() { crashIfBorrowed(); }

  void crashIfBorrowed() const {}

  bool setIsBorrowed(bool isBorrowed) const {
    bool previous = m_isBorrowed;
    m_isBorrowed = isBorrowed;
    return previous;
  }

private:
  mutable bool m_isBorrowed { false };
};

template <typename T> class Borrow {
public:
  Borrow(T &ref LIFETIME_BOUND)
      : m_ref(ref), m_previous(ref.setIsBorrowed(true)) {}
  ~Borrow() { m_ref.setIsBorrowed(m_previous); }

  Borrow(const Borrow &) = delete;
  Borrow &operator=(const Borrow &) = delete;

  operator T &() const LIFETIME_BOUND { return m_ref; }
  T &get() const LIFETIME_BOUND { return m_ref; }
  T *operator->() const LIFETIME_BOUND { return &m_ref; }

private:
  T &m_ref;
  bool m_previous;
};

template <typename T> Borrow(T &) -> Borrow<T>;

template <typename T> Borrow<T> borrow(T &ref LIFETIME_BOUND) {
  return Borrow<T>(ref);
}

template <typename T> class VectorBufferBase {
public:
  void crashIfBorrowed() const {}

  bool setIsBorrowed(bool isBorrowed) const {
    bool previous = m_isBorrowed;
    m_isBorrowed = isBorrowed;
    return previous;
  }

protected:
  T *m_buffer { nullptr };
  mutable bool m_isBorrowed { false };
};

template <typename T> class VectorBuffer : private VectorBufferBase<T> {
  typedef VectorBufferBase<T> Base;

public:
  using Base::crashIfBorrowed;
  using Base::setIsBorrowed;

protected:
  using Base::m_buffer;
};

template <typename T> class Vector : private VectorBuffer<T> {
  typedef VectorBuffer<T> Buffer;

public:
  using Buffer::setIsBorrowed;

  T &operator[](unsigned i) LIFETIME_BOUND { return Buffer::m_buffer[i]; }
  const T &operator[](unsigned i) const LIFETIME_BOUND {
    return Buffer::m_buffer[i];
  }
  T *data() LIFETIME_BOUND { return Buffer::m_buffer; }
  const T *data() const LIFETIME_BOUND { return Buffer::m_buffer; }
  T *begin() LIFETIME_BOUND { return Buffer::m_buffer; }
  T *end() LIFETIME_BOUND { return Buffer::m_buffer + m_size; }
  const T *begin() const LIFETIME_BOUND { return Buffer::m_buffer; }
  const T *end() const LIFETIME_BOUND { return Buffer::m_buffer + m_size; }
  unsigned size() const { return m_size; }

  void append(const T &);

private:
  unsigned m_size { 0 };
};

template <typename T> class SimpleContainer : public CanBorrow {
public:
  T &operator[](unsigned i) LIFETIME_BOUND { return m_buffer[i]; }
  void append(const T &);

private:
  T *m_buffer { nullptr };
};

class StringView {
public:
  StringView() = default;
  StringView(const char *data LIFETIME_BOUND) : m_data(data) {}

private:
  const char *m_data { nullptr };
};

class [[gsl::Pointer]] CharSpan {
public:
  CharSpan() = default;
  CharSpan(char *data) : m_data(data) {}

private:
  char *m_data { nullptr };
};

class NotBorrowable {
public:
  char &at(unsigned i) LIFETIME_BOUND { return m_buffer[i]; }
  void mutate();

private:
  char *m_buffer { nullptr };
};

template <typename T> class Owner {
public:
  T *get() const LIFETIME_BOUND { return m_ptr; }
  T &operator*() const LIFETIME_BOUND { return *m_ptr; }
  T *operator->() const LIFETIME_BOUND { return m_ptr; }

private:
  T *m_ptr { nullptr };
};

CharSpan makeSpan(Vector<char> &vec LIFETIME_BOUND);
StringView makeView(const char *data LIFETIME_BOUND);

CharSpan makeSpanUnannotated(Vector<char> &vec);

Vector<char> &forwardRef(Vector<char> &vec LIFETIME_BOUND);
Vector<char> *forwardPtr(Vector<char> &vec LIFETIME_BOUND);

class Node : public CanBorrow {
public:
  Node &firstChild() LIFETIME_BOUND { return m_children[0]; }
  void appendChild();

private:
  Vector<Node> m_children;
};

const char *pick(const char *a LIFETIME_BOUND, const char *b LIFETIME_BOUND);

class TwoStringViews {
public:
  TwoStringViews(const char *a LIFETIME_BOUND, const char *b LIFETIME_BOUND);
};

template <typename T> class Registry;

template <typename T> class Cursor {
public:
  Cursor(Registry<T> &registry LIFETIME_BOUND, unsigned index)
      : m_registry(&registry), m_index(index) {}

  T &value() const LIFETIME_BOUND { return m_registry->at(m_index); }
  void remove() const { m_registry->removeAt(m_index); }

private:
  Registry<T> *m_registry;
  unsigned m_index;
};

template <typename T> class RegistryIterator {
public:
  RegistryIterator(Registry<T> &registry LIFETIME_BOUND, unsigned index)
      : m_registry(&registry), m_index(index) {}

  Cursor<T> operator*() const LIFETIME_BOUND {
    return Cursor<T>(*m_registry, m_index);
  }
  RegistryIterator &operator++() {
    ++m_index;
    return *this;
  }
  bool operator!=(const RegistryIterator &other) const {
    return m_index != other.m_index;
  }

private:
  Registry<T> *m_registry;
  unsigned m_index;
};

template <typename T> class Registry : public CanBorrow {
public:
  T &at(unsigned i) LIFETIME_BOUND { return m_buffer[i]; }

  void removeAt(unsigned i) { crashIfBorrowed(); }

  RegistryIterator<T> begin() LIFETIME_BOUND {
    return RegistryIterator<T>(*this, 0);
  }
  RegistryIterator<T> end() LIFETIME_BOUND {
    return RegistryIterator<T>(*this, m_size);
  }

private:
  T *m_buffer { nullptr };
  unsigned m_size { 0 };
};

class Element {
public:
  void mutate();
  void inspect() const;
};

namespace detail {
class CallableBase {
public:
  virtual ~CallableBase() {}
  virtual void call() = 0;
};

template <typename F> class Callable : public CallableBase {
public:
  Callable(F f) : m_f(f) {}
  void call() override { m_f(); }

private:
  F m_f;
};
} // namespace detail

class Function {
public:
  template <typename F>
  Function(F f) : m_impl(new detail::Callable<F>(f)) {}
  ~Function() { delete m_impl; }

  void operator()() const { m_impl->call(); }

private:
  detail::CallableBase *m_impl { nullptr };
};

void callEscaping(const Function &);
void callNoEscape([[clang::noescape]] const Function &);

namespace std {
inline namespace __1 {
using size_t = decltype(sizeof(0));

namespace ranges {
template <typename Derived> class view_interface {};
} // namespace ranges

template <typename Iterator> class reverse_iterator {
public:
  reverse_iterator(Iterator);
  auto &operator*() const { return *m_it; }
  reverse_iterator &operator++();
  bool operator!=(const reverse_iterator &) const;

private:
  Iterator m_it;
};

template <typename A, typename B> struct pair {
  A first;
  B second;
};

template <size_t I, typename A, typename B> A &get(pair<A, B> &);

template <typename T> T *data(Vector<T> &);
} // namespace __1
} // namespace std

#endif
