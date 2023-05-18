extern int mergeUsedFlag;

typedef struct {
  int n;
  int m;
} NameForLinkage;
extern NameForLinkage name_for_linkage;

struct HasVirtualFunctions {
  virtual void f();
};
struct OverridesVirtualFunctions : HasVirtualFunctions {
  void f();
};
extern OverridesVirtualFunctions overrides_virtual_functions;
extern "C" void ExternCFunction();

typedef struct {
  struct Inner {
    int n;
  };
} NameForLinkage2;
auto name_for_linkage2_inner_b = NameForLinkage2::Inner();
typedef decltype(name_for_linkage2_inner_b) NameForLinkage2Inner;

namespace Aliased { extern int b; }
namespace Alias = Aliased;

struct InhCtorA { InhCtorA(int); };
struct InhCtorB : InhCtorA { using InhCtorA::InhCtorA; };
