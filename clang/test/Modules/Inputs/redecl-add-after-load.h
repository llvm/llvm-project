struct A {};
extern const int variable = 0;
extern constexpr int function() { return 0; }

namespace N {
  struct A {};
  extern const int variable = 0;
  extern constexpr int function() { return 0; }
}

@import redecl_add_after_load_top;
struct C::A {};
const int C::variable = 0;
constexpr int C::function() { return 0; }

struct D {
  struct A;
  static const int variable;
  static constexpr int function();
};
struct D::A {};
const int D::variable = 0;
constexpr int D::function() { return 0; }
