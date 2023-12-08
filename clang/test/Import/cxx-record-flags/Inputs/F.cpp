class FTrivial {
  int i;
};

struct FNonTrivial {
  virtual ~FNonTrivial() = default;
  int i;
};

