namespace UsingDecl {
  namespace B { typedef int inner; }
  using B::inner;
}

#include "using-decl-a.h"

namespace UsingDecl {
  using ::using_decl_type;
  using ::using_decl_var;
  using ::merged;
}

namespace X {
  int conflicting_hidden_using_decl;
  int conflicting_hidden_using_decl_fn();
  int conflicting_hidden_using_decl_var;
  struct conflicting_hidden_using_decl_struct;

  int conflicting_hidden_using_decl_mixed_1;
  int conflicting_hidden_using_decl_mixed_2();
  struct conflicting_hidden_using_decl_mixed_3 {};
}

using X::conflicting_hidden_using_decl;
using X::conflicting_hidden_using_decl_fn;
using X::conflicting_hidden_using_decl_var;
using X::conflicting_hidden_using_decl_struct;
int conflicting_hidden_using_decl_fn_2();
int conflicting_hidden_using_decl_var_2;
struct conflicting_hidden_using_decl_struct_2 {};

using X::conflicting_hidden_using_decl_mixed_1;
using X::conflicting_hidden_using_decl_mixed_2;
using X::conflicting_hidden_using_decl_mixed_3;
int conflicting_hidden_using_decl_mixed_4;
int conflicting_hidden_using_decl_mixed_5();
struct conflicting_hidden_using_decl_mixed_6 {};
