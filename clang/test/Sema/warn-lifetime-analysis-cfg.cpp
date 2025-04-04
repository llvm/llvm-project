// RUN: %clang_cc1 -Wno-dangling -Wdangling-cfg -verify -std=c++11 %s
#include "Inputs/lifetime-analysis.h"

std::string_view unknown(std::string_view s);
std::string_view unknown();
std::string temporary();

int return_value(int t) { return t; }

std::string_view simple_with_ctor() {
  std::string s1; // expected-note {{reference to this stack variable is returned}}
  std::string_view cs1 = s1;
  return cs1; // expected-warning {{returning reference to a stack variable}}
}

std::string_view simple_with_assignment() {
  std::string s1; // expected-note {{reference to this stack variable is returned}}
  std::string_view cs1;
  cs1 = s1;
  return cs1; // expected-warning {{returning reference to a stack variable}}
}

std::string_view use_unknown() {
  std::string s1;
  std::string_view cs1 = s1;
  cs1 = unknown();
  return cs1; // ok.
}

std::string_view overwrite_unknown() {
  std::string s1; // expected-note {{reference to this stack variable is returned}}
  std::string_view cs1 = s1;
  cs1 = unknown();
  cs1 = s1;
  return cs1; // expected-warning {{returning reference to a stack variable}}
}

std::string_view overwrite_with_unknown() {
  std::string s1;
  std::string_view cs1 = s1;
  cs1 = s1;
  cs1 = unknown();
  return cs1; // Ok.
}

std::string_view return_unknown() {
  std::string s1;
  std::string_view cs1 = s1;
  return unknown(cs1);
}

std::string_view multiple_assignments() {
  std::string s1;
  std::string s2 = s1; // expected-note {{reference to this stack variable is returned}}
  std::string_view cs1 = s1;
  std::string_view cs2 = s1;
  s2 = s1; // Ignore owner transfers.
  cs1 = s1;
  cs2 = s2;
  cs1 = cs2;
  cs2 = s1;
  return cs1; // expected-warning {{returning reference to a stack variable}}
}

std::string_view global_view;
std::string_view ignore_global_view() {
  std::string s; // expected-warning {{reference to local object dangles}}
  global_view = s; // TODO: We can still track writes to static storage!
  return global_view; // Ok.
}
std::string global_owner;
std::string_view ignore_global_owner() {
  std::string_view sv;
  sv = global_owner;
  return sv; // Ok.
}

std::string_view return_temporary() {
  return 
    temporary(); // expected-warning {{returning reference to a temporary object}}
}

std::string_view store_temporary() {
  std::string_view sv =
    temporary(); // expected-warning {{returning reference to a temporary object}}
  return sv;
}

std::string_view return_local() {
  std::string local; // expected-note {{reference to this stack variable is returned}}
  return // expected-warning {{returning reference to a stack variable}}
    local; 
}

int* return_null_ptr() {
  auto T = nullptr;
  return T;
}

// Parameters
std::string_view params_owner_and_views(
  std::string_view sv,
  std::string s) { // expected-note {{reference to this stack variable is returned}}
  sv = s;
  return sv; // expected-warning {{returning reference to a stack variable}}
}

std::string_view param_owner(std::string s) { // expected-note {{reference to this stack variable is returned}}
  return s; // expected-warning {{returning reference to a stack variable}}
}
std::string_view param_owner_ref(const std::string& s) {
  return s;
}

std::string& get_str_ref();
std::string_view ref_to_global_str() {
  const std::string& s =  get_str_ref();
  std::string_view sv = s;
  return sv;
}
std::string_view view_to_global_str() {
  std::string_view s =  get_str_ref();
  std::string_view sv = s;
  sv = s;
  return sv;
}
std::string_view copy_of_global_str() {
  std::string s =  get_str_ref(); // expected-note {{reference to this stack variable is returned}}
  std::string_view sv = s;
  return sv; // expected-warning {{returning reference to a stack variable}}
}

struct Struct { std::string s; };
std::string_view field() {
  Struct s;
  std::string_view sv;
  sv = s.s;
  return sv; // FIXME.
}

std::string_view loopReturnFor(int n) {
  for (int i = 0; i < n; ++i) {
    std::string local = "inside-loop"; // expected-note {{reference to this stack variable is returned}}
    return local; // expected-warning {{returning reference to a stack variable}}
  }
  return "safe"; // Safe return.
}

std::string_view loopReturnWhile(bool condition) {
  while (condition) {
    std::string local = "inside-loop"; // expected-note {{reference to this stack variable is returned}}
    return local; // expected-warning {{returning reference to a stack variable}}
  }
  return "safe"; // Safe return.
}

std::string_view loopReturnDoWhile() {
  do {
    std::string local = "inside-loop"; // expected-note {{reference to this stack variable is returned}}
    return local; // expected-warning {{returning reference to a stack variable}}
  } while (false);
}

std::string_view nestedConditionals(bool a, bool b) {
  std::string local = "nested"; // expected-note {{reference to this stack variable is returned}}
  if (a) {
    if (b) {
      return local; // expected-warning {{returning reference to a stack variable}}
    }
  }
  return "safe";
}

std::string_view nestedLoops(int n) {
  std::string local = "loop"; // expected-note {{reference to this stack variable is returned}}
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      return local; // expected-warning {{returning reference to a stack variable}}
    }
  }
  return "safe";
}

namespace lifetimebound {

std::string_view func_lb_sv(std::string_view sv [[clang::lifetimebound]]);
std::string_view use_func_lb_sv() {
  std::string s; // expected-note {{reference to this stack variable is returned}}
  std::string_view sv = func_lb_sv(s);
  sv = func_lb_sv(s);
  std::string_view sv2;
  sv2 = sv;
  return sv2; // expected-warning {{returning reference to a stack variable}}
}

std::string_view use_string_func_lb_sv() {
  std::string s; // expected-note {{reference to this stack variable is returned}}
  std::string_view sv = s;
  std::string_view sv_lb = func_lb_sv(sv);
  return sv_lb; // expected-warning {{returning reference to a stack variable}}
}

std::string_view direct_return() {
  std::string s; // expected-note {{reference to this stack variable is returned}}
  std::string_view sv = s;
  return func_lb_sv(sv); // expected-warning {{returning reference to a stack variable}}
}

std::string_view if_branches(std::string param, bool cond) { // expected-note 4 {{reference to this stack variable is returned}}
  std::string_view view;
  view = func_lb_sv(param);
  view = func_lb_sv(view);
  std::string_view stack = view;
  if (cond) view = func_lb_sv(view);
  if (cond) view = func_lb_sv(view);
  if (cond)
    return view; // expected-warning {{returning reference to a stack variable}}
  if (cond)
    view = unknown();
  // Now it is potentially safe!
  if (cond)
    return view; // Ok. We can't know for sure.
  if (cond) {
    view = stack;
    return view;  // expected-warning {{returning reference to a stack variable}}
  }
  // Unconditionally restore to stack.
  view = stack;
  if (cond)
    return view; // expected-warning {{returning reference to a stack variable}}
  return view; // expected-warning {{returning reference to a stack variable}}
}

std::string_view nested_early_return(std::string param, bool a, bool b) { // expected-note 2 {{reference to this stack variable is returned}}
  std::string_view view;
  view = func_lb_sv(param);
  view = func_lb_sv(view);
  std::string_view stack = view;

  if (a) {
    if (b) {
      return view; // expected-warning {{returning reference to a stack variable}}
    }
    view = unknown();
  } else {
    if (b) {
      view = stack;
      return view;  // expected-warning {{returning reference to a stack variable}}
    }
  }
  return view; // Ok, because one branch set it to unknown.
}

std::string_view func_with_branches(std::string param, bool cond1, bool cond2) { // expected-note 2 {{reference to this stack variable is returned}}
  std::string_view view;
  view = func_lb_sv(param);
  view = func_lb_sv(view);
  std::string_view stack = view;

  if (cond1)
    view = func_lb_sv(view);

  if (cond2)
    view = unknown();
  else if (cond1)
    return view;  // expected-warning {{returning reference to a stack variable}}

  if (cond1 && cond2)
    return view; // Ok. We can't be sure.

  view = stack;
  return view; // expected-warning {{returning reference to a stack variable}}
}


namespace MemberFunctions {
struct S {
  std::string return_string() const;
  const std::string& return_string_ref() const [[clang::lifetimebound]];
};

std::string_view use_return_string() {
  S s;
  return s.return_string(); // expected-warning {{returning reference to a temporary object}}
}
std::string_view use_return_string_ref() {
  S s; // expected-note {{reference to this stack variable is returned}}
  return s.return_string_ref(); // expected-warning {{returning reference to a stack variable}}
}
std::string_view use_return_string_ref(const S* s) {
  return s->return_string_ref();
}
std::string_view use_return_string_ref_temporary() {
  return S{}.return_string_ref(); // expected-warning {{returning reference to a temporary object}}
}
} // namespace MemberFunctions

namespace Optional {
std::optional<std::string_view> getOptional(std::string_view);
std::string_view usOptional(std::string_view s) {
  return getOptional(s).value();
}
} // namespace Optional
struct ThisIsLB {
std::string_view get() [[clang::lifetimebound]];
};

std::string_view use_lifetimebound_member_fn() {
  ThisIsLB obj; // expected-note {{reference to this stack variable is returned}}
  return obj.get(); // expected-warning {{returning reference to a stack variable}}
}

std::string_view return_temporary_get() {
  return ThisIsLB{}.get(); // expected-warning {{returning reference to a temporary object}}
}

std::string_view store_temporary_get() {
  // FIXME: Move this diagnostic to the return loc!!
  std::string_view sv1 = ThisIsLB{}.get(); // expected-warning {{returning reference to a temporary object}}
  std::string_view sv2 = sv1;
  std::string_view sv3 = func_lb_sv(sv2);
  return sv3;
}

std::string_view multiple_lifetimebound_calls() {
  std::string s; // expected-note {{reference to this stack variable is returned}}
  std::string_view sv = func_lb_sv(func_lb_sv(func_lb_sv(s)));
  sv = func_lb_sv(func_lb_sv(func_lb_sv(sv)));
  return sv; // expected-warning {{returning reference to a stack variable}}
}

} // namespace lifetimebound

std::string_view return_char_star() {
  const char* key;
  key = "foo";
  return key;
}

void lambda() {
  std::string s;
  auto l1 = [&s]() {
    std::string_view sv = s;
    return sv;
  };
  auto l2 = []() {
    std::string s; // expected-note {{reference to this stack variable is returned}}
    std::string_view sv = s;
    return sv; // expected-warning {{returning reference to a stack variable}}
  };
}

std::string_view default_arg(std::string_view sv = std::string()) {
  return sv; // Ok!
}
std::string_view default_arg_overwritten(std::string_view sv = std::string()) {
  std::string s; // expected-note {{reference to this stack variable is returned}}
  sv = s;
  return sv; // expected-warning {{returning reference to a stack variable}}
}

std::string_view containerOfString(bool cond) {
  if (cond) {
    return std::vector<std::string>{"one"}.at(0); // expected-warning {{returning reference to a temporary object}}
  }
  std::vector<std::string> local; // expected-note 2 {{reference to this stack variable is returned}}
  if (cond) {
    return local.at(0); // expected-warning {{returning reference to a stack variable}}
  }
  std::string_view sv = local.at(0);
  if (cond) {
    return sv; // expected-warning {{returning reference to a stack variable}}
  }
  return unknown();
}

std::optional<std::string_view> optional_return_dangling(bool flag) {
  if (flag) {
    std::string local = "return"; // expected-note {{reference to this stack variable is returned}}
    return local; // expected-warning {{returning reference to a stack variable}}
  }
  return {};
}

std::optional<std::string_view> containerOfPointers(bool cond) {
  std::string s; // expected-note 5 {{reference to this stack variable is returned}}
  std::string_view sv;
  std::optional<std::string_view> osv1;
  std::optional<std::string_view> osv2;
  sv = s;
  osv1 = sv;
  if (cond)
    return s; // expected-warning {{returning reference to a stack variable}}
  if (cond)
    return sv; // expected-warning {{returning reference to a stack variable}}
  if (cond)
    return osv1; // expected-warning {{returning reference to a stack variable}}
  if (cond)
    return osv2;
  osv2 = osv1;
  if (cond)
    return osv2; // expected-warning {{returning reference to a stack variable}}
  return s; // expected-warning {{returning reference to a stack variable}}
}

std::string_view conditionalReturn(bool flag) {
  std::string local1 = "if-branch"; // expected-note {{reference to this stack variable is returned}}
  std::string local2 = "else-branch"; // expected-note {{reference to this stack variable is returned}}
  if (flag)
    return local1; // expected-warning {{returning reference to a stack variable}}
  else
    return local2; // expected-warning {{returning reference to a stack variable}}
}

std::string_view conditionalReturnMixed(bool flag, const std::string& external) {
  std::string local = "stack-var"; // expected-note {{reference to this stack variable is returned}}
  if (flag)
    return local; // expected-warning {{returning reference to a stack variable}}
  else
    return external; // OK, returning reference to external.
}

void move_to_arena(std::unique_ptr<int>);
void move_to_arena(int*);
int* move_unique_ptr(bool cond) {
  std::unique_ptr<int> local; // expected-note 3 {{reference to this stack variable is returned}}
  int* pointer = local.get();
  if (cond)
    return local.get(); // expected-warning {{returning reference to a stack variable}}
  if (cond)
    return pointer; // expected-warning {{returning reference to a stack variable}}
  if (cond) {
    // 'local' definitely moved to unknown.
    move_to_arena(std::move(local));
    return pointer;
  }
  if(cond)
    return pointer; // expected-warning {{returning reference to a stack variable}}
  // Maybe 'local' is moved to unknown.
  if (cond)
    move_to_arena(local.release());
  return pointer; // Ok.
}

namespace ViewConstructedFromNonPointerNonOwner {
  const std::string &getstring();

struct KeyView {
  KeyView(const std::string &c [[clang::lifetimebound]]);
  KeyView();
};
struct [[gsl::Pointer]] SetView {
  SetView(const KeyView &c);
};
SetView getstringLB(const std::string &s [[clang::lifetimebound]]);

SetView foo() { return KeyView(); }
} // namespace ViewConstructedFromNonPointerNonOwner

namespace SuggestLifetimebound {
std::string_view return_param(std::string_view s) { // expected-warning {{param should be marked lifetimebound}}
  return s;
}
std::string_view getLB(std::string_view s [[clang::lifetimebound]]);
std::string_view return_lifetimebound_call(std::string_view s) { // expected-warning {{param should be marked lifetimebound}}
  return getLB(s);
}
std::string_view controlflow(std::string_view s) { // expected-warning {{param should be marked lifetimebound}}
  std::string_view result = s;
  result = getLB(result);
  return result;
}
} // namespace SuggestLifetimebound

namespace DanglingReferences {
// FIXME: Detect dangling-references to temporary.
void use(std::string_view);
void while_without_use(bool cond) {
  std::string_view live_pointer;
  while (cond) {
    std::string scope_local; // expected-warning {{reference to local object dangles}}
    live_pointer = scope_local;
  }
  while (cond) {
    std::string_view dead_pointer;
    std::string scope_local;
    dead_pointer = scope_local;
  }
}

void conditional_scope(bool flag) {
  std::string_view dangling;
  if (flag) {
    std::string local; // expected-warning {{reference to local object dangles}}
    dangling = local;
  }
  use(dangling);
}

void conditional_scope_without_use(bool flag) {
  std::string_view dangling;
  if (flag) {
    std::string local;
    dangling = local;
  }
}

void nested_blocks(bool flag) {
  std::string_view dangling;
  if (flag) {
    if (flag) {
      std::string local; // expected-warning {{reference to local object dangles}}
      dangling = local;
    }
  }
  use(dangling);
}

void switch_case(int cond) {
  std::string_view dangling;
  switch (cond) {
    case 1: {
      std::string local; // expected-warning {{reference to local object dangles}}
      dangling = local;
      break;
    }
    case 2: {
      std::string another_local; // expected-warning {{reference to local object dangles}}
      dangling = another_local;
      break;
    }
  }
  use(dangling);
}

// FIXME: use capture by to detect this!
void dangling_in_vector(bool cond) {
  std::vector<std::string_view> dangling_views;
  {
    std::string local;
    dangling_views.push_back(local);
  }
  if (cond) {
    use(dangling_views.at(0));
  }
}

void optional_dangling(bool cond) {
  std::optional<std::string_view> opt_view;
  {
    std::string local = "temporary"; // expected-warning {{reference to local object dangles}}
    opt_view = local;
  }
  if (cond) {
    use(*opt_view); // Dereferencing a dangling view.
  }
}
void optional_nested_if_else(bool cond1, bool cond2) {
  std::optional<std::string_view> opt_view;
  if (cond1) {
    if (cond2) {
      std::string local = "nested"; // expected-warning {{reference to local object dangles}}
      opt_view = local;
    }
  } else {
    std::string other_local = "else branch"; // expected-warning {{reference to local object dangles}}
    opt_view = other_local;
  }
  if (cond1) {
    use(*opt_view); // Dangling use.
  }
}

void optional_in_loop(int n) {
  std::optional<std::string_view> opt_view;
  for (int i = 0; i < n; ++i) {
    std::string local = "loop"; // expected-warning {{reference to local object dangles}}
    opt_view = local;
  }
  if (n > 0) {
    use(*opt_view); // Dangling use after loop exits.
  }
}

void optional_early_return(bool cond) {
  std::optional<std::string_view> opt_view;
  {
    std::string local = "early return"; // expected-warning {{reference to local object dangles}}
    opt_view = local;
  }
  if (cond) {
    return; // Dangling reference still exists but not used.
  }
  use(*opt_view);
}

void optional_dangling_reassign(bool cond) {
  std::optional<std::string_view> opt_view;
  {
    std::string local = "first assignment"; // expected-warning {{reference to local object dangles}}
    opt_view = local;
  }
  if (cond) {
    std::string other_local = "second assignment"; // expected-warning {{reference to local object dangles}}
    opt_view = other_local;
  }
  use(*opt_view);
}
} // namespace DanglingReferences