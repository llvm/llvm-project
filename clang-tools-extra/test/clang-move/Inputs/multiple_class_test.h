namespace a {
class Move1 {
public:
  int f();
};
} // namespace a

namespace b {
class Move2 {
public:
  int f();
};
} // namespace b

namespace c {
class Move3 {
public:
  int f();
};

class Move4 {
public:
  int f();
};

class NoMove {
public:
  int f();
};
} // namespace c
