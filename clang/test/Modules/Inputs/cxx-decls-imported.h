class HasFriends {
  friend void friend_1(HasFriends);
  friend void friend_2(HasFriends);
  void private_thing();
};

struct HasNontrivialDefaultConstructor {
  HasNontrivialDefaultConstructor() = default;
  HasNontrivialDefaultConstructor(int n = 0);

  // Ensure this class is not POD but is still trivially-copyable.
  // This is necessary to exercise the second static_assert below,
  // because GCC's spec for __has_trivial_constructor is absurd.
  int m;
private:
  int n;
};

static_assert(!__is_trivial(HasNontrivialDefaultConstructor), "");
static_assert(!__has_trivial_constructor(HasNontrivialDefaultConstructor), "");

void *operator new[](__SIZE_TYPE__);

extern int mergeUsedFlag;
inline int getMergeUsedFlag() { return mergeUsedFlag; }

typedef struct {
  int n;
  int m;
} NameForLinkage;

struct HasVirtualFunctions {
  virtual void f();
};
struct OverridesVirtualFunctions : HasVirtualFunctions {
  void f();
};
extern "C" void ExternCFunction();

typedef struct {
  struct Inner {
    int n;
  };
} NameForLinkage2;
auto name_for_linkage2_inner_a = NameForLinkage2::Inner();
typedef decltype(name_for_linkage2_inner_a) NameForLinkage2Inner;

namespace Aliased { extern int a; }
namespace Alias = Aliased;

struct InhCtorA { InhCtorA(int); };
struct InhCtorB : InhCtorA { using InhCtorA::InhCtorA; };

struct ClassWithVBases : HasFriends, virtual HasNontrivialDefaultConstructor {
  int n;
};
struct ClassWithVBases;
