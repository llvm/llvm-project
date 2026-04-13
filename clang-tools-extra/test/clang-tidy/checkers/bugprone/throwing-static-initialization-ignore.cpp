// RUN: %check_clang_tidy %s bugprone-throwing-static-initialization %t -- \
// RUN:   -config="{CheckOptions: \
// RUN:             {bugprone-throwing-static-initialization.IgnoredTypes: \"Ignore;^ns::S1$;^ns::Template<1>$\"}}" \
// RUN:   -- -fexceptions

struct S1 {
  S1() noexcept(false);
};

struct S1_Ignore {
  S1_Ignore() noexcept(false);
};

namespace ns {
struct S1 {
  S1();
};
struct S1_Ignore {
  S1_Ignore();
};
template<int>
struct Template {
  Template() noexcept(false);
};
}

template<class>
struct TemplateIgnored {
  TemplateIgnored() noexcept(false);
};

S1_Ignore getS1() noexcept(false);

S1 VarThrow;
// CHECK-MESSAGES: :[[@LINE-1]]:4: warning: initialization of 'VarThrow' with static storage duration may throw an exception that cannot be caught
ns::Template<2> VarTempl;
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: initialization of 'VarTempl' with static storage duration may throw an exception that cannot be caught

S1_Ignore VarIgnoreConstr;
S1_Ignore VarIgnoreInitF = getS1();
ns::S1 VarIgnore2;
ns::S1_Ignore VarIgnore3;
TemplateIgnored<int> VarIgnore4;
ns::Template<1> VarIgnore5;
