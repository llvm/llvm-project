typedef struct A B;
extern const int variable;
extern constexpr int function();
constexpr int test(bool b) { return b ? variable : function(); }

namespace N {
  typedef struct A B;
  extern const int variable;
  extern constexpr int function();
}
typedef N::B NB;
constexpr int N_test(bool b) { return b ? N::variable : N::function(); }

@import redecl_add_after_load_top;
typedef C::A CB;
constexpr int C_test(bool b) { return b ? C::variable : C::function(); }

struct D {
  struct A;
  static const int variable;
  static constexpr int function();
};
typedef D::A DB;
constexpr int D_test(bool b) { return b ? D::variable : D::function(); }
