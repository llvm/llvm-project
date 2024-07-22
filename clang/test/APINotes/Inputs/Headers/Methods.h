struct IntWrapper {
  int value;

  IntWrapper getIncremented() const { return {value + 1}; }
};

// TODO: support nested tags
struct Outer {
  struct Inner {
    int value;

    Inner getDecremented() const { return {value - 1}; }
  };
};
