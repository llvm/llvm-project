template<typename ...Args>
int f(Args ...args) {
  return sizeof...(args) + sizeof...(Args);
}

void test() {
  int a;
  decltype(a) b;

  typedef int Integer;
  typedef float Float;
  typedef bool Bool;
  bool b2 = __is_trivially_constructible(Integer, Float, Bool);
}

typedef int Int;

class B {
 virtual void foo(Int);
};

class S : public B {
  virtual void foo(Int) override;
};

// Need std::initializer_list
namespace std {
  typedef decltype(sizeof(int)) size_t;

  // libc++'s implementation
  template <class _E>
  class initializer_list
  {
    const _E* __begin_;
    size_t    __size_;

    initializer_list(const _E* __b, size_t __s)
      : __begin_(__b),
        __size_(__s)
    {}

  public:
    typedef _E        value_type;
    typedef const _E& reference;
    typedef const _E& const_reference;
    typedef size_t    size_type;

    typedef const _E* iterator;
    typedef const _E* const_iterator;

    initializer_list() : __begin_(nullptr), __size_(0) {}

    size_t    size()  const {return __size_;}
    const _E* begin() const {return __begin_;}
    const _E* end()   const {return __begin_ + __size_;}
  };
}

struct Foo {
  Foo(std::initializer_list<int> il);
};

void test2() {
  Foo{10};
}


// RUN: c-index-test -test-annotate-tokens=%s:1:1:5:1 -fno-delayed-template-parsing -std=c++11 %s | FileCheck %s
// CHECK: Identifier: "args" [3:20 - 3:24] SizeOfPackExpr=args:2:15
// CHECK: Identifier: "Args" [3:38 - 3:42] TypeRef=Args:1:22

// RUN: c-index-test -test-annotate-tokens=%s:8:1:9:1 -std=c++11 %s | FileCheck -check-prefix=CHECK-DECLTYPE %s
// CHECK-DECLTYPE: Identifier: "a" [8:12 - 8:13] DeclRefExpr=a:7:7

// RUN: c-index-test -test-annotate-tokens=%s:13:1:14:1 -std=c++11 %s | FileCheck -check-prefix=CHECK-TRAIT %s
// CHECK-TRAIT: Identifier: "Integer" [13:42 - 13:49] TypeRef=Integer:10:15
// CHECK-TRAIT: Identifier: "Float" [13:51 - 13:56] TypeRef=Float:11:17
// CHECK-TRAIT: Identifier: "Bool" [13:58 - 13:62] TypeRef=Bool:12:16

// RUN: c-index-test -test-annotate-tokens=%s:16:1:24:1 -std=c++11 %s | FileCheck -check-prefix=CHECK-WITH-OVERRIDE %s
// CHECK-WITH-OVERRIDE: Keyword: "virtual" [19:2 - 19:9] CXXMethod=foo:19:15 (virtual)
// CHECK-WITH-OVERRIDE: Keyword: "void" [19:10 - 19:14] CXXMethod=foo:19:15 (virtual)
// CHECK-WITH-OVERRIDE: Identifier: "foo" [19:15 - 19:18] CXXMethod=foo:19:15 (virtual)
// CHECK-WITH-OVERRIDE: Punctuation: "(" [19:18 - 19:19] CXXMethod=foo:19:15 (virtual)
// CHECK-WITH-OVERRIDE: Identifier: "Int" [19:19 - 19:22] TypeRef=Int:16:13
// CHECK-WITH-OVERRIDE: Punctuation: ")" [19:22 - 19:23] CXXMethod=foo:19:15 (virtual)
// CHECK-WITH-OVERRIDE: Punctuation: ";" [19:23 - 19:24] ClassDecl=B:18:7 (Definition)
// CHECK-WITH-OVERRIDE: Keyword: "virtual" [23:3 - 23:10] CXXMethod=foo:23:16 (virtual) [Overrides @19:15]
// CHECK-WITH-OVERRIDE: Keyword: "void" [23:11 - 23:15] CXXMethod=foo:23:16 (virtual) [Overrides @19:15]
// CHECK-WITH-OVERRIDE: Identifier: "foo" [23:16 - 23:19] CXXMethod=foo:23:16 (virtual) [Overrides @19:15]
// CHECK-WITH-OVERRIDE: Punctuation: "(" [23:19 - 23:20] CXXMethod=foo:23:16 (virtual) [Overrides @19:15]
// CHECK-WITH-OVERRIDE: Identifier: "Int" [23:20 - 23:23] TypeRef=Int:16:13
// CHECK-WITH-OVERRIDE: Punctuation: ")" [23:23 - 23:24] CXXMethod=foo:23:16 (virtual) [Overrides @19:15]
// CHECK-WITH-OVERRIDE: Keyword: "override" [23:25 - 23:33] attribute(override)=
// CHECK-WITH-OVERRIDE: Punctuation: ";" [23:33 - 23:34] ClassDecl=S:22:7 (Definition)

// RUN: c-index-test -test-annotate-tokens=%s:64:1:65:1 -std=c++11 %s | FileCheck -check-prefix=CHECK-INITLIST %s
// CHECK-INITLIST: Identifier: "Foo" [64:3 - 64:6] TypeRef=struct Foo:59:8
