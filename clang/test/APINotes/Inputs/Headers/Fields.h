struct IntWrapper {
  int value;

  bool : 1;

  enum {
    one,
    two,
    three
  };
  union {
    int a;
    char b;
  };
};

struct Outer {
  struct Inner {
    int value;
  };
};
