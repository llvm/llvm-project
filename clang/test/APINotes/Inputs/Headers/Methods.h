struct IntWrapper {
  int value;

  IntWrapper getIncremented() const { return {value + 1}; }

  IntWrapper operator+(const IntWrapper& RHS) const { return {value + RHS.value}; }

  const int& operator*() const { return value; }
};

extern "C++" {
struct IntWrapper2 {
  int value;

  IntWrapper2 getIncremented() const { return {value + 1}; }
};
}

extern "C++" {
extern "C" {
  struct IntWrapper3 {
    static IntWrapper3 getIncremented(IntWrapper3 val) { return val; }
  };
}
}

struct Outer {
  struct Inner {
    int value;

    Inner getDecremented() const { return {value - 1}; }

    bool operator==(const Inner& RHS) const {
      return value == RHS.value;
    }
  };
};
