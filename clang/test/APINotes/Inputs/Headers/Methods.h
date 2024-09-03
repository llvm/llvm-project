struct IntWrapper {
  int value;

  IntWrapper getIncremented() const { return {value + 1}; }

  IntWrapper operator+(const IntWrapper& RHS) const { return {value + RHS.value}; }
};

struct Outer {
  struct Inner {
    int value;

    Inner getDecremented() const { return {value - 1}; }

    bool operator==(const Inner& RHS) const {
      return value == RHS.value;
    }
  };
};
