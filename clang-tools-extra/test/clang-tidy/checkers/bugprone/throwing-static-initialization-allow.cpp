// RUN: %check_clang_tidy %s bugprone-throwing-static-initialization %t -- \
// RUN:   -config="{CheckOptions: \
// RUN:             {bugprone-throwing-static-initialization.AllowedTypes: \"Allow;^ns::S1$;^ns::Template<1>$\"}}" \
// RUN:   -- -fexceptions

struct S1 {
  S1() noexcept(false);
};

struct S1_Allow {
  S1_Allow() noexcept(false);
};

namespace ns {
struct S1 {
  S1();
};
struct S1_Allow {
  S1_Allow();
};
template<int>
struct Template {
  Template() noexcept(false);
};
}

template<class>
struct TemplateAllowed {
  TemplateAllowed() noexcept(false);
};

S1_Allow getS1() noexcept(false);

S1 VarThrow;
// CHECK-MESSAGES: :[[@LINE-1]]:4: warning: initialization of 'VarThrow' with static storage duration may throw an exception that cannot be caught
ns::Template<2> VarTempl;
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: initialization of 'VarTempl' with static storage duration may throw an exception that cannot be caught

S1_Allow VarAllowedConstr;
S1_Allow VarAllowedInitF = getS1();
ns::S1 VarAllowed2;
ns::S1_Allow VarAllowed3;
TemplateAllowed<int> VarAllowed4;
ns::Template<1> VarAllowed5;
