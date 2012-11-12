// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

template<typename T>
class unique_ptr {
  T *ptr;

  unique_ptr(const unique_ptr&) = delete; // expected-note 3{{function has been explicitly marked deleted here}}
  unique_ptr &operator=(const unique_ptr&) = delete; // expected-note{{candidate function has been explicitly deleted}}
public:
  unique_ptr() : ptr(0) { }
  unique_ptr(unique_ptr &&other) : ptr(other.ptr) { other.ptr = 0; }
  explicit unique_ptr(T *ptr) : ptr(ptr) { }

  ~unique_ptr() { delete ptr; }

  unique_ptr &operator=(unique_ptr &&other) { // expected-note{{candidate function not viable: no known conversion from 'unique_ptr<int>' to 'unique_ptr<int> &&' for 1st argument}}
    if (this == &other)
      return *this;

    delete ptr;
    ptr = other.ptr;
    other.ptr = 0;
    return *this;
  }
};

template<typename T>
struct remove_reference {
  typedef T type;
};

template<typename T>
struct remove_reference<T&> {
  typedef T type;
};

template<typename T>
struct remove_reference<T&&> {
  typedef T type;
};


template <class T> typename remove_reference<T>::type&& move(T&& t) {
  return static_cast<typename remove_reference<T>::type&&>(t);
}

template <class T> T&& forward(typename remove_reference<T>::type& t) {
  return static_cast<T&&>(t);
}

template <class T> T&& forward(typename remove_reference<T>::type&& t) {
  return static_cast<T&&>(t);
}

template<typename T, typename ...Args>
unique_ptr<T> make_unique_ptr(Args &&...args) {
  return unique_ptr<T>(new T(forward<Args>(args)...));
}

template<typename T> void accept_unique_ptr(unique_ptr<T>); // expected-note{{passing argument to parameter here}}

unique_ptr<int> test_unique_ptr() {
  // Simple construction
  unique_ptr<int> p;
  unique_ptr<int> p1(new int);

  // Move construction
  unique_ptr<int> p2(make_unique_ptr<int>(17));
  unique_ptr<int> p3 = make_unique_ptr<int>(17);

  // Copy construction (failures)
  unique_ptr<int> p4(p); // expected-error{{call to deleted constructor of 'unique_ptr<int>'}}
  unique_ptr<int> p5 = p; // expected-error{{call to deleted constructor of 'unique_ptr<int>'}}

  // Move assignment
  p2 = move(p);
  p2 = make_unique_ptr<int>(0);

  // Copy assignment (failures);
  p2 = p3; // expected-error{{overload resolution selected deleted operator '='}}

  // Implicit copies
  accept_unique_ptr(make_unique_ptr<double>(0.0));
  accept_unique_ptr(move(p2));

  // Implicit copies (failures);
  accept_unique_ptr(p); // expected-error{{call to deleted constructor of 'unique_ptr<int>'}}

  return p;
}

namespace perfect_forwarding {
  struct A { };

  struct F0 {
    void operator()(A&, const A&, A&&, const A&&, A&&, const A&&); // expected-note{{candidate function not viable: 5th argument ('const perfect_forwarding::A') would lose const qualifier}}
  };

  template<typename F, typename ...Args>
  void forward(F f, Args &&...args) {
    f(static_cast<Args&&>(args)...); // expected-error{{no matching function for call to object of type 'perfect_forwarding::F0'}}
  }

  template<typename T> T get();

  void test_forward() {
    forward(F0(), get<A&>(), get<A const&>(), get<A>(), get<const A>(),
            get<A&&>(), get<const A&&>());
    forward(F0(), get<A&>(), get<A const&>(), get<A>(), get<const A>(), // expected-note{{in instantiation of function template specialization 'perfect_forwarding::forward<perfect_forwarding::F0, perfect_forwarding::A &, const perfect_forwarding::A &, perfect_forwarding::A, const perfect_forwarding::A, const perfect_forwarding::A, const perfect_forwarding::A>' requested here}}
            get<const A&&>(), get<const A&&>());
  }
};
