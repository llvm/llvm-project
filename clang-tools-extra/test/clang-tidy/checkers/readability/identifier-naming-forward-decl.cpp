// RUN: %check_clang_tidy -std=c++98-or-later %s readability-identifier-naming %t -- \
// RUN:   -config='{CheckOptions: { \
// RUN:     readability-identifier-naming.DefaultCase: lower_case, \
// RUN:     readability-identifier-naming.ClassCase: CamelCase, \
// RUN:     readability-identifier-naming.StructCase: CamelCase, \
// RUN:     readability-identifier-naming.UnionCase: CamelCase, \
// RUN:     readability-identifier-naming.TemplateParameterCase: CamelCase \
// RUN:   }}'

// Forward declarations should use their semantic declaration kind
// instead of falling back to DefaultCase.

// Namespace-scope forward declarations.

class GoodClass;

class bad_class;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for class 'bad_class'
// CHECK-FIXES: class BadClass;

struct GoodStruct;

struct bad_struct;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for struct 'bad_struct'
// CHECK-FIXES: struct BadStruct;

union GoodUnion;

union bad_union;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for union 'bad_union'
// CHECK-FIXES: union BadUnion;

// Nested forward declarations.

class Outer {
  class GoodInnerClass;

  class bad_inner_class;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for class 'bad_inner_class'
  // CHECK-FIXES: class BadInnerClass;

  struct GoodInnerStruct;

  struct bad_inner_struct;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: invalid case style for struct 'bad_inner_struct'
  // CHECK-FIXES: struct BadInnerStruct;

  union GoodInnerUnion;

  union bad_inner_union;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for union 'bad_inner_union'
  // CHECK-FIXES: union BadInnerUnion;

};

// Template forward declarations.

template <typename T>
class GoodTemplateClass;

template <typename T>
class bad_template_class;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for class 'bad_template_class'
// CHECK-FIXES: class BadTemplateClass;

template <typename T>
struct GoodTemplateStruct;

template <typename T>
struct bad_template_struct;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for struct 'bad_template_struct'
// CHECK-FIXES: struct BadTemplateStruct;

template <typename T>
union GoodTemplateUnion;

template <typename T>
union bad_template_union;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for union 'bad_template_union'
// CHECK-FIXES: union BadTemplateUnion;
