struct Incomplete;

struct X : Incomplete {
  X();
};

struct Y : X {
  using X::X;
};
