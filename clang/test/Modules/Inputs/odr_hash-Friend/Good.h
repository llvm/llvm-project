template <class T>
struct iterator {
  void Compare(const iterator &x) { }
  friend void Check(iterator) {}
};

template <class T = int> struct Box {
  iterator<T> I;

  void test() {
    Check(I);
    I.Compare(I);
  }
};

// Force instantiation of Box<int>
Box<> B;
