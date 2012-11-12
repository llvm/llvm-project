namespace outer {
  namespace inner {
    template<typename T>
    struct vector {
      typedef T* iterator;
    };
  }
}

namespace outer_alias = outer;

struct X { };

using outer_alias::inner::vector;

struct X_vector : outer_alias::inner::vector<X> {
  using outer_alias::inner::vector<X>::iterator;
};

namespace outer {
  namespace inner {
    template<typename T, unsigned N>
    struct array {
      void foo();
      static int max_size;
    };
  }
}

template<typename T, unsigned N>
void outer::inner::array<T, N>::foo() {
}

template<typename T, unsigned N>
int outer::inner::array<T, N>::max_size = 17;

template<typename T>
struct X2 : outer::inner::vector<T> {
  typedef T type;
  using typename outer::inner::vector<type>::iterator;
  using outer::inner::vector<type>::push_back;
};

namespace outer {
  namespace inner {
    namespace secret {
    }
  }
}

using namespace outer_alias::inner::secret;
namespace super_secret = outer_alias::inner::secret;

template<typename T>
struct X3 {
  void f(T *t) {
    t->::outer_alias::inner::template vector<T>::~vector<T>();
  }
};

namespace outer {
  namespace inner {
    void f(int);
    void f(double);
  }
}

template<typename T>
struct X4 {
  typedef T type;
  void g(int);
  void g(float);

  void h(T t) {
    ::outer_alias::inner::f(t);
    ::X4<type>::g(t);
    this->::X4<type>::g(t);
  }
};

typedef int Integer;
template<>
struct X4<Integer> {
  typedef Integer type;

  void g(int);
  void g(float);

  void h(type t) {
    ::outer_alias::inner::f(t);
    ::X4<type>::g(t);
    this->::X4<type>::g(t);
  }
};


template<typename T>
struct X5 {
  typedef T type;
  typedef typename outer_alias::inner::vector<type>::iterator iter_type;
  typedef typename outer_alias::inner::vector<int>::iterator int_ptr_type;
};

template<typename T>
struct X6 {
  typedef T* type;
  typedef typename outer_alias::inner::vector<type>::template rebind<type> type1;
  typedef typename outer_alias::inner::vector<type>::template rebind<type>::other type2;
  typedef class outer_alias::inner::vector<type>::template rebind<type> type3;
  typedef class outer_alias::inner::vector<type>::template rebind<type>::other type4;
};

namespace outer {
  namespace inner {
    template<typename T, template<class> class TT>
    struct apply_meta {
      typedef typename TT<T>::type type;
    };
  }
}

template<typename T, typename U>
struct X7 {
  typedef T T_type;
  typedef U U_type;
  typedef outer_alias::inner::apply_meta<T_type, U_type::template apply> type;
};

struct X8 {
  void f();
};

struct X9 : X8 {
  typedef X8 inherited;
  void f() { 
    inherited::f();
  }
};

// RUN: c-index-test -test-annotate-tokens=%s:13:1:137:1 -fno-delayed-template-parsing %s | FileCheck %s

// CHECK: Keyword: "using" [14:1 - 14:6] UsingDeclaration=vector[4:12]
// CHECK: Identifier: "outer_alias" [14:7 - 14:18] NamespaceRef=outer_alias:10:11
// CHECK: Punctuation: "::" [14:18 - 14:20] UsingDeclaration=vector[4:12]
// CHECK: Identifier: "inner" [14:20 - 14:25] NamespaceRef=inner:2:13
// CHECK: Punctuation: "::" [14:25 - 14:27] UsingDeclaration=vector[4:12]
// CHECK: Identifier: "vector" [14:27 - 14:33] OverloadedDeclRef=vector[4:12]
// CHECK: Punctuation: ";" [14:33 - 14:34]

// Base specifiers
// CHECK: Identifier: "outer_alias" [16:19 - 16:30] NamespaceRef=outer_alias:10:11
// CHECK: Punctuation: "::" [16:30 - 16:32] C++ base class specifier=outer_alias::inner::vector<X>:4:12 [access=public isVirtual=false]
// CHECK: Identifier: "inner" [16:32 - 16:37] NamespaceRef=inner:2:13
// CHECK: Punctuation: "::" [16:37 - 16:39] C++ base class specifier=outer_alias::inner::vector<X>:4:12 [access=public isVirtual=false]
// CHECK: Identifier: "vector" [16:39 - 16:45] TemplateRef=vector:4:12
// CHECK: Punctuation: "<" [16:45 - 16:46] C++ base class specifier=outer_alias::inner::vector<X>:4:12 [access=public isVirtual=false]
// CHECK: Identifier: "X" [16:46 - 16:47] TypeRef=struct X:12:8
// CHECK: Punctuation: ">" [16:47 - 16:48] C++ base class specifier=outer_alias::inner::vector<X>:4:12 [access=public isVirtual=false]


// CHECK: Keyword: "using" [17:3 - 17:8] UsingDeclaration=iterator[5:18]
// CHECK: Identifier: "outer_alias" [17:9 - 17:20] NamespaceRef=outer_alias:10:11
// CHECK: Punctuation: "::" [17:20 - 17:22] UsingDeclaration=iterator[5:18]
// CHECK: Identifier: "inner" [17:22 - 17:27] NamespaceRef=inner:2:13
// CHECK: Punctuation: "::" [17:27 - 17:29] UsingDeclaration=iterator[5:18]
// CHECK: Identifier: "vector" [17:29 - 17:35] TemplateRef=vector:4:12
// CHECK: Punctuation: "<" [17:35 - 17:36] UsingDeclaration=iterator[5:18]
// CHECK: Identifier: "X" [17:36 - 17:37] TypeRef=struct X:12:8
// CHECK: Punctuation: ">" [17:37 - 17:38] UsingDeclaration=iterator[5:18]
// CHECK: Punctuation: "::" [17:38 - 17:40] UsingDeclaration=iterator[5:18]
// CHECK: Identifier: "iterator" [17:40 - 17:48] OverloadedDeclRef=iterator[5:18]

// CHECK: Keyword: "void" [31:1 - 31:5] CXXMethod=foo:31:33 (Definition)
// CHECK: Identifier: "outer" [31:6 - 31:11] NamespaceRef=outer:20:11
// CHECK: Punctuation: "::" [31:11 - 31:13] CXXMethod=foo:31:33 (Definition)
// CHECK: Identifier: "inner" [31:13 - 31:18] NamespaceRef=inner:21:13
// CHECK: Punctuation: "::" [31:18 - 31:20] CXXMethod=foo:31:33 (Definition)
// CHECK: Identifier: "array" [31:20 - 31:25] TemplateRef=array:23:12
// CHECK: Punctuation: "<" [31:25 - 31:26] CXXMethod=foo:31:33 (Definition)
// CHECK: Identifier: "T" [31:26 - 31:27] TypeRef=T:30:19
// CHECK: Punctuation: "," [31:27 - 31:28] CXXMethod=foo:31:33 (Definition)
// CHECK: Identifier: "N" [31:29 - 31:30] DeclRefExpr=N:30:31
// CHECK: Punctuation: ">" [31:30 - 31:31] CXXMethod=foo:31:33 (Definition)
// CHECK: Punctuation: "::" [31:31 - 31:33] CXXMethod=foo:31:33 (Definition)
// CHECK: Identifier: "foo" [31:33 - 31:36] CXXMethod=foo:31:33 (Definition)
// CHECK: Punctuation: "(" [31:36 - 31:37] CXXMethod=foo:31:33 (Definition)
// CHECK: Punctuation: ")" [31:37 - 31:38] CXXMethod=foo:31:33 (Definition)

// CHECK: Keyword: "int" [35:1 - 35:4] VarDecl=max_size:35:32 (Definition)
// CHECK: Identifier: "outer" [35:5 - 35:10] NamespaceRef=outer:20:11
// CHECK: Punctuation: "::" [35:10 - 35:12] VarDecl=max_size:35:32 (Definition)
// CHECK: Identifier: "inner" [35:12 - 35:17] NamespaceRef=inner:21:13
// CHECK: Punctuation: "::" [35:17 - 35:19] VarDecl=max_size:35:32 (Definition)
// CHECK: Identifier: "array" [35:19 - 35:24] TemplateRef=array:23:12
// CHECK: Punctuation: "<" [35:24 - 35:25] VarDecl=max_size:35:32 (Definition)
// CHECK: Identifier: "T" [35:25 - 35:26] TypeRef=T:34:19
// CHECK: Punctuation: "," [35:26 - 35:27] VarDecl=max_size:35:32 (Definition)
// CHECK: Identifier: "N" [35:28 - 35:29] DeclRefExpr=N:34:31
// CHECK: Punctuation: ">" [35:29 - 35:30] VarDecl=max_size:35:32 (Definition)
// CHECK: Punctuation: "::" [35:30 - 35:32] VarDecl=max_size:35:32 (Definition)
// CHECK: Identifier: "max_size" [35:32 - 35:40] VarDecl=max_size:35:32 (Definition)
// CHECK: Punctuation: "=" [35:41 - 35:42] VarDecl=max_size:35:32 (Definition)
// CHECK: Literal: "17" [35:43 - 35:45] IntegerLiteral=
// CHECK: Punctuation: ";" [35:45 - 35:46]

// CHECK: Keyword: "using" [40:3 - 40:8] UsingDeclaration=iterator:40:46
// CHECK: Keyword: "typename" [40:9 - 40:17] UsingDeclaration=iterator:40:46
// CHECK: Identifier: "outer" [40:18 - 40:23] NamespaceRef=outer:20:11
// CHECK: Punctuation: "::" [40:23 - 40:25] UsingDeclaration=iterator:40:46
// CHECK: Identifier: "inner" [40:25 - 40:30] NamespaceRef=inner:21:13
// CHECK: Punctuation: "::" [40:30 - 40:32] UsingDeclaration=iterator:40:46
// CHECK: Identifier: "vector" [40:32 - 40:38] TemplateRef=vector:4:12
// CHECK: Punctuation: "<" [40:38 - 40:39] UsingDeclaration=iterator:40:46
// CHECK: Identifier: "type" [40:39 - 40:43] TypeRef=type:39:13
// CHECK: Punctuation: ">" [40:43 - 40:44] UsingDeclaration=iterator:40:46
// CHECK: Punctuation: "::" [40:44 - 40:46] UsingDeclaration=iterator:40:46
// CHECK: Identifier: "iterator" [40:46 - 40:54] UsingDeclaration=iterator:40:46
// CHECK: Punctuation: ";" [40:54 - 40:55] ClassTemplate=X2:38:8 (Definition)
// CHECK: Keyword: "using" [41:3 - 41:8] UsingDeclaration=push_back:41:37
// CHECK: Identifier: "outer" [41:9 - 41:14] NamespaceRef=outer:20:11
// CHECK: Punctuation: "::" [41:14 - 41:16] UsingDeclaration=push_back:41:37
// CHECK: Identifier: "inner" [41:16 - 41:21] NamespaceRef=inner:21:13
// CHECK: Punctuation: "::" [41:21 - 41:23] UsingDeclaration=push_back:41:37
// CHECK: Identifier: "vector" [41:23 - 41:29] TemplateRef=vector:4:12
// CHECK: Punctuation: "<" [41:29 - 41:30] UsingDeclaration=push_back:41:37
// CHECK: Identifier: "type" [41:30 - 41:34] TypeRef=type:39:13
// CHECK: Punctuation: ">" [41:34 - 41:35] UsingDeclaration=push_back:41:37
// CHECK: Punctuation: "::" [41:35 - 41:37] UsingDeclaration=push_back:41:37
// CHECK: Identifier: "push_back" [41:37 - 41:46] UsingDeclaration=push_back:41:37

// Using directive
// CHECK: Keyword: "using" [51:1 - 51:6] UsingDirective=:51:37
// CHECK: Keyword: "namespace" [51:7 - 51:16] UsingDirective=:51:37
// CHECK: Identifier: "outer_alias" [51:17 - 51:28] NamespaceRef=outer_alias:10:11
// CHECK: Punctuation: "::" [51:28 - 51:30] UsingDirective=:51:37
// CHECK: Identifier: "inner" [51:30 - 51:35] NamespaceRef=inner:45:13
// CHECK: Punctuation: "::" [51:35 - 51:37] UsingDirective=:51:37
// CHECK: Identifier: "secret" [51:37 - 51:43] NamespaceRef=secret:46:15

// Namespace alias
// CHECK: Keyword: "namespace" [52:1 - 52:10] NamespaceAlias=super_secret:52:11
// CHECK: Identifier: "super_secret" [52:11 - 52:23] NamespaceAlias=super_secret:52:11
// CHECK: Punctuation: "=" [52:24 - 52:25] NamespaceAlias=super_secret:52:11
// CHECK: Identifier: "outer_alias" [52:26 - 52:37] NamespaceRef=outer_alias:10:11
// CHECK: Punctuation: "::" [52:37 - 52:39] NamespaceAlias=super_secret:52:11
// CHECK: Identifier: "inner" [52:39 - 52:44] NamespaceRef=inner:45:13
// CHECK: Punctuation: "::" [52:44 - 52:46] NamespaceAlias=super_secret:52:11
// CHECK: Identifier: "secret" [52:46 - 52:52] NamespaceRef=secret:46:15
// CHECK: Punctuation: ";" [52:52 - 52:53]

// Pseudo-destructor
// CHECK: Identifier: "t" [57:5 - 57:6] DeclRefExpr=t:56:13
// CHECK: Punctuation: "->" [57:6 - 57:8] MemberRefExpr=
// CHECK: Punctuation: "::" [57:8 - 57:10] MemberRefExpr=
// CHECK: Identifier: "outer_alias" [57:10 - 57:21] NamespaceRef=outer_alias:10:11
// CHECK: Punctuation: "::" [57:21 - 57:23] MemberRefExpr=
// CHECK: Identifier: "inner" [57:23 - 57:28] NamespaceRef=inner:45:13
// CHECK: Punctuation: "::" [57:28 - 57:30] MemberRefExpr=
// CHECK: Keyword: "template" [57:30 - 57:38] MemberRefExpr=
// CHECK: Identifier: "vector" [57:39 - 57:45] TemplateRef=vector:4:12
// CHECK: Punctuation: "<" [57:45 - 57:46] MemberRefExpr=
// CHECK: Identifier: "T" [57:46 - 57:47] TypeRef=T:54:19
// CHECK: Punctuation: ">" [57:47 - 57:48] MemberRefExpr=
// CHECK: Punctuation: "::" [57:48 - 57:50] MemberRefExpr=
// CHECK: Punctuation: "~" [57:50 - 57:51] MemberRefExpr=
// CHECK: Identifier: "vector" [57:51 - 57:57] TemplateRef=vector:4:12
// CHECK: Punctuation: "<" [57:57 - 57:58] MemberRefExpr=
// CHECK: Identifier: "T" [57:58 - 57:59] TypeRef=T:54:19
// CHECK: Punctuation: ">" [57:59 - 57:60] CallExpr=
// CHECK: Punctuation: "(" [57:60 - 57:61] CallExpr=
// CHECK: Punctuation: ")" [57:61 - 57:62] CallExpr=

// Unresolved member and non-member references
// CHECK: Punctuation: "::" [75:5 - 75:7] DeclRefExpr=[63:10, 64:10]
// CHECK: Identifier: "outer_alias" [75:7 - 75:18] NamespaceRef=outer_alias:10:11
// CHECK: Punctuation: "::" [75:18 - 75:20] DeclRefExpr=[63:10, 64:10]
// CHECK: Identifier: "inner" [75:20 - 75:25] NamespaceRef=inner:62:13
// CHECK: Punctuation: "::" [75:25 - 75:27] DeclRefExpr=[63:10, 64:10]
// CHECK: Identifier: "f" [75:27 - 75:28] OverloadedDeclRef=f[63:10, 64:10]
// CHECK: Punctuation: "(" [75:28 - 75:29] CallExpr=
// CHECK: Identifier: "t" [75:29 - 75:30] DeclRefExpr=t:74:12
// CHECK: Punctuation: ")" [75:30 - 75:31] CallExpr=
// CHECK: Punctuation: "::" [76:5 - 76:7] MemberRefExpr=[71:8, 72:8]
// CHECK: Identifier: "X4" [76:7 - 76:9] TemplateRef=X4:69:8
// CHECK: Punctuation: "<" [76:9 - 76:10] MemberRefExpr=[71:8, 72:8]
// CHECK: Identifier: "type" [76:10 - 76:14] TypeRef=type:70:13
// CHECK: Punctuation: ">" [76:14 - 76:15] MemberRefExpr=[71:8, 72:8]
// CHECK: Punctuation: "::" [76:15 - 76:17] MemberRefExpr=[71:8, 72:8]
// CHECK: Identifier: "g" [76:17 - 76:18] OverloadedDeclRef=g[71:8, 72:8]
// CHECK: Punctuation: "(" [76:18 - 76:19] CallExpr=
// CHECK: Identifier: "t" [76:19 - 76:20] DeclRefExpr=t:74:12
// CHECK: Punctuation: ")" [76:20 - 76:21] CallExpr=
// CHECK: Punctuation: ";" [76:21 - 76:22] CompoundStmt=
// CHECK: Keyword: "this" [77:5 - 77:9] CXXThisExpr=
// CHECK: Punctuation: "->" [77:9 - 77:11] MemberRefExpr=
// CHECK: Punctuation: "::" [77:11 - 77:13] MemberRefExpr=
// CHECK: Identifier: "X4" [77:13 - 77:15] TemplateRef=X4:69:8
// CHECK: Punctuation: "<" [77:15 - 77:16] MemberRefExpr=
// CHECK: Identifier: "type" [77:16 - 77:20] TypeRef=type:70:13
// CHECK: Punctuation: ">" [77:20 - 77:21] MemberRefExpr=
// CHECK: Punctuation: "::" [77:21 - 77:23] MemberRefExpr=
// CHECK: Identifier: "g" [77:23 - 77:24] MemberRefExpr=
// CHECK: Punctuation: "(" [77:24 - 77:25] CallExpr=
// CHECK: Identifier: "t" [77:25 - 77:26] DeclRefExpr=t:74:12
// CHECK: Punctuation: ")" [77:26 - 77:27] CallExpr=

// Resolved member and non-member references
// CHECK: Punctuation: "::" [90:5 - 90:7] DeclRefExpr=f:63:10
// CHECK: Identifier: "outer_alias" [90:7 - 90:18] NamespaceRef=outer_alias:10:11
// CHECK: Punctuation: "::" [90:18 - 90:20] DeclRefExpr=f:63:10
// CHECK: Identifier: "inner" [90:20 - 90:25] NamespaceRef=inner:62:13
// CHECK: Punctuation: "::" [90:25 - 90:27] DeclRefExpr=f:63:10
// CHECK: Identifier: "f" [90:27 - 90:28] DeclRefExpr=f:63:10
// CHECK: Punctuation: "(" [90:28 - 90:29] CallExpr=f:63:10
// CHECK: Identifier: "t" [90:29 - 90:30] DeclRefExpr=t:89:15
// CHECK: Punctuation: ")" [90:30 - 90:31] CallExpr=f:63:10
// CHECK: Punctuation: ";" [90:31 - 90:32] CompoundStmt=
// CHECK: Punctuation: "::" [91:5 - 91:7] MemberRefExpr=g:86:8
// CHECK: Identifier: "X4" [91:7 - 91:9] TemplateRef=X4:69:8
// CHECK: Punctuation: "<" [91:9 - 91:10] MemberRefExpr=g:86:8
// CHECK: Identifier: "type" [91:10 - 91:14] TypeRef=type:84:19
// CHECK: Punctuation: ">" [91:14 - 91:15] MemberRefExpr=g:86:8
// CHECK: Punctuation: "::" [91:15 - 91:17] MemberRefExpr=g:86:8
// CHECK: Identifier: "g" [91:17 - 91:18] MemberRefExpr=g:86:8
// CHECK: Punctuation: "(" [91:18 - 91:19] CallExpr=g:86:8
// CHECK: Identifier: "t" [91:19 - 91:20] DeclRefExpr=t:89:15
// CHECK: Punctuation: ")" [91:20 - 91:21] CallExpr=g:86:8
// CHECK: Punctuation: ";" [91:21 - 91:22] CompoundStmt=
// CHECK: Keyword: "this" [92:5 - 92:9] CXXThisExpr=
// CHECK: Punctuation: "->" [92:9 - 92:11] MemberRefExpr=g:86:8
// CHECK: Punctuation: "::" [92:11 - 92:13] MemberRefExpr=g:86:8
// CHECK: Identifier: "X4" [92:13 - 92:15] TemplateRef=X4:69:8
// CHECK: Punctuation: "<" [92:15 - 92:16] MemberRefExpr=g:86:8
// CHECK: Identifier: "type" [92:16 - 92:20] TypeRef=type:84:19
// CHECK: Punctuation: ">" [92:20 - 92:21] MemberRefExpr=g:86:8
// CHECK: Punctuation: "::" [92:21 - 92:23] MemberRefExpr=g:86:8
// CHECK: Identifier: "g" [92:23 - 92:24] MemberRefExpr=g:86:8
// CHECK: Punctuation: "(" [92:24 - 92:25] CallExpr=g:86:8
// CHECK: Identifier: "t" [92:25 - 92:26] DeclRefExpr=t:89:15
// CHECK: Punctuation: ")" [92:26 - 92:27] CallExpr=g:86:8

// Dependent name type
// CHECK: Keyword: "typedef" [100:3 - 100:10] ClassTemplate=X5:98:8 (Definition)
// CHECK: Keyword: "typename" [100:11 - 100:19] TypedefDecl=iter_type:100:63 (Definition)
// CHECK: Identifier: "outer_alias" [100:20 - 100:31] NamespaceRef=outer_alias:10:11
// CHECK: Punctuation: "::" [100:31 - 100:33] TypedefDecl=iter_type:100:63 (Definition)
// CHECK: Identifier: "inner" [100:33 - 100:38] NamespaceRef=inner:62:13
// CHECK: Punctuation: "::" [100:38 - 100:40] TypedefDecl=iter_type:100:63 (Definition)
// CHECK: Identifier: "vector" [100:40 - 100:46] TemplateRef=vector:4:12
// CHECK: Punctuation: "<" [100:46 - 100:47] TypedefDecl=iter_type:100:63 (Definition)
// CHECK: Identifier: "type" [100:47 - 100:51] TypeRef=type:99:13
// CHECK: Punctuation: ">" [100:51 - 100:52] TypedefDecl=iter_type:100:63 (Definition)
// CHECK: Punctuation: "::" [100:52 - 100:54] TypedefDecl=iter_type:100:63 (Definition)
// CHECK: Identifier: "iterator" [100:54 - 100:62] TypedefDecl=iter_type:100:63 (Definition)
// CHECK: Identifier: "iter_type" [100:63 - 100:72] TypedefDecl=iter_type:100:63 (Definition)

// CHECK: Keyword: "typedef" [101:3 - 101:10] ClassTemplate=X5:98:8 (Definition)
// CHECK: Keyword: "typename" [101:11 - 101:19] TypedefDecl=int_ptr_type:101:62 (Definition)
// CHECK: Identifier: "outer_alias" [101:20 - 101:31] NamespaceRef=outer_alias:10:11
// CHECK: Punctuation: "::" [101:31 - 101:33] TypedefDecl=int_ptr_type:101:62 (Definition)
// CHECK: Identifier: "inner" [101:33 - 101:38] NamespaceRef=inner:62:13
// CHECK: Punctuation: "::" [101:38 - 101:40] TypedefDecl=int_ptr_type:101:62 (Definition)
// CHECK: Identifier: "vector" [101:40 - 101:46] TemplateRef=vector:4:12
// CHECK: Punctuation: "<" [101:46 - 101:47] TypedefDecl=int_ptr_type:101:62 (Definition)
// CHECK: Keyword: "int" [101:47 - 101:50] TypedefDecl=int_ptr_type:101:62 (Definition)
// CHECK: Punctuation: ">" [101:50 - 101:51] TypedefDecl=int_ptr_type:101:62 (Definition)
// CHECK: Punctuation: "::" [101:51 - 101:53] TypedefDecl=int_ptr_type:101:62 (Definition)
// CHECK: Identifier: "iterator" [101:53 - 101:61] TypeRef=iterator:5:18
// CHECK: Identifier: "int_ptr_type" [101:62 - 101:74] TypedefDecl=int_ptr_type:101:62 (Definition)

// Dependent template specialization types
// CHECK: Keyword: "typename" [107:11 - 107:19] TypedefDecl=type1:107:76 (Definition)
// CHECK: Identifier: "outer_alias" [107:20 - 107:31] NamespaceRef=outer_alias:10:11
// CHECK: Punctuation: "::" [107:31 - 107:33] TypedefDecl=type1:107:76 (Definition)
// CHECK: Identifier: "inner" [107:33 - 107:38] NamespaceRef=inner:62:13
// CHECK: Punctuation: "::" [107:38 - 107:40] TypedefDecl=type1:107:76 (Definition)
// CHECK: Identifier: "vector" [107:40 - 107:46] TemplateRef=vector:4:12
// CHECK: Punctuation: "<" [107:46 - 107:47] TypedefDecl=type1:107:76 (Definition)
// CHECK: Identifier: "type" [107:47 - 107:51] TypeRef=type:106:14
// CHECK: Punctuation: ">" [107:51 - 107:52] TypedefDecl=type1:107:76 (Definition)
// CHECK: Punctuation: "::" [107:52 - 107:54] TypedefDecl=type1:107:76 (Definition)
// CHECK: Keyword: "template" [107:54 - 107:62] TypedefDecl=type1:107:76 (Definition)
// CHECK: Identifier: "rebind" [107:63 - 107:69] TypedefDecl=type1:107:76 (Definition)
// CHECK: Punctuation: "<" [107:69 - 107:70] TypedefDecl=type1:107:76 (Definition)
// CHECK: Identifier: "type" [107:70 - 107:74] TypeRef=type:106:14
// CHECK: Punctuation: ">" [107:74 - 107:75] TypedefDecl=type1:107:76 (Definition)
// CHECK: Identifier: "type1" [107:76 - 107:81] TypedefDecl=type1:107:76 (Definition)

// CHECK: Keyword: "typedef" [108:3 - 108:10] ClassTemplate=X6:105:8 (Definition)
// CHECK: Keyword: "typename" [108:11 - 108:19] TypedefDecl=type2:108:83 (Definition)
// CHECK: Identifier: "outer_alias" [108:20 - 108:31] NamespaceRef=outer_alias:10:11
// CHECK: Punctuation: "::" [108:31 - 108:33] TypedefDecl=type2:108:83 (Definition)
// CHECK: Identifier: "inner" [108:33 - 108:38] NamespaceRef=inner:62:13
// CHECK: Punctuation: "::" [108:38 - 108:40] TypedefDecl=type2:108:83 (Definition)
// CHECK: Identifier: "vector" [108:40 - 108:46] TemplateRef=vector:4:12
// CHECK: Punctuation: "<" [108:46 - 108:47] TypedefDecl=type2:108:83 (Definition)
// CHECK: Identifier: "type" [108:47 - 108:51] TypeRef=type:106:14
// CHECK: Punctuation: ">" [108:51 - 108:52] TypedefDecl=type2:108:83 (Definition)
// CHECK: Punctuation: "::" [108:52 - 108:54] TypedefDecl=type2:108:83 (Definition)
// CHECK: Keyword: "template" [108:54 - 108:62] TypedefDecl=type2:108:83 (Definition)
// CHECK: Identifier: "rebind" [108:63 - 108:69] TypedefDecl=type2:108:83 (Definition)
// CHECK: Punctuation: "<" [108:69 - 108:70] TypedefDecl=type2:108:83 (Definition)
// CHECK: Identifier: "type" [108:70 - 108:74] TypeRef=type:106:14
// CHECK: Punctuation: ">" [108:74 - 108:75] TypedefDecl=type2:108:83 (Definition)
// CHECK: Punctuation: "::" [108:75 - 108:77] TypedefDecl=type2:108:83 (Definition)
// CHECK: Identifier: "other" [108:77 - 108:82] TypedefDecl=type2:108:83 (Definition)
// CHECK: Identifier: "type2" [108:83 - 108:88] TypedefDecl=type2:108:83 (Definition)

// CHECK: Keyword: "typedef" [109:3 - 109:10] ClassTemplate=X6:105:8 (Definition)
// CHECK: Keyword: "class" [109:11 - 109:16] TypedefDecl=type3:109:73 (Definition)
// CHECK: Identifier: "outer_alias" [109:17 - 109:28] NamespaceRef=outer_alias:10:11
// CHECK: Punctuation: "::" [109:28 - 109:30] TypedefDecl=type3:109:73 (Definition)
// CHECK: Identifier: "inner" [109:30 - 109:35] NamespaceRef=inner:62:13
// CHECK: Punctuation: "::" [109:35 - 109:37] TypedefDecl=type3:109:73 (Definition)
// CHECK: Identifier: "vector" [109:37 - 109:43] TemplateRef=vector:4:12
// CHECK: Punctuation: "<" [109:43 - 109:44] TypedefDecl=type3:109:73 (Definition)
// CHECK: Identifier: "type" [109:44 - 109:48] TypeRef=type:106:14
// CHECK: Punctuation: ">" [109:48 - 109:49] TypedefDecl=type3:109:73 (Definition)
// CHECK: Punctuation: "::" [109:49 - 109:51] TypedefDecl=type3:109:73 (Definition)
// CHECK: Keyword: "template" [109:51 - 109:59] TypedefDecl=type3:109:73 (Definition)
// CHECK: Identifier: "rebind" [109:60 - 109:66] TypedefDecl=type3:109:73 (Definition)
// CHECK: Punctuation: "<" [109:66 - 109:67] TypedefDecl=type3:109:73 (Definition)
// CHECK: Identifier: "type" [109:67 - 109:71] TypeRef=type:106:14
// CHECK: Punctuation: ">" [109:71 - 109:72] TypedefDecl=type3:109:73 (Definition)
// CHECK: Identifier: "type3" [109:73 - 109:78] TypedefDecl=type3:109:73 (Definition)

// CHECK: Keyword: "class" [110:11 - 110:16] TypedefDecl=type4:110:80 (Definition)
// CHECK: Identifier: "outer_alias" [110:17 - 110:28] NamespaceRef=outer_alias:10:11
// CHECK: Punctuation: "::" [110:28 - 110:30] TypedefDecl=type4:110:80 (Definition)
// CHECK: Identifier: "inner" [110:30 - 110:35] NamespaceRef=inner:62:13
// CHECK: Punctuation: "::" [110:35 - 110:37] TypedefDecl=type4:110:80 (Definition)
// CHECK: Identifier: "vector" [110:37 - 110:43] TemplateRef=vector:4:12
// CHECK: Punctuation: "<" [110:43 - 110:44] TypedefDecl=type4:110:80 (Definition)
// CHECK: Identifier: "type" [110:44 - 110:48] TypeRef=type:106:14
// CHECK: Punctuation: ">" [110:48 - 110:49] TypedefDecl=type4:110:80 (Definition)
// CHECK: Punctuation: "::" [110:49 - 110:51] TypedefDecl=type4:110:80 (Definition)
// CHECK: Keyword: "template" [110:51 - 110:59] TypedefDecl=type4:110:80 (Definition)
// CHECK: Identifier: "rebind" [110:60 - 110:66] TypedefDecl=type4:110:80 (Definition)
// CHECK: Punctuation: "<" [110:66 - 110:67] TypedefDecl=type4:110:80 (Definition)
// CHECK: Identifier: "type" [110:67 - 110:71] TypeRef=type:106:14
// CHECK: Punctuation: ">" [110:71 - 110:72] TypedefDecl=type4:110:80 (Definition)
// CHECK: Punctuation: "::" [110:72 - 110:74] TypedefDecl=type4:110:80 (Definition)
// CHECK: Identifier: "other" [110:74 - 110:79] TypedefDecl=type4:110:80 (Definition)
// CHECK: Identifier: "type4" [110:80 - 110:85] TypedefDecl=type4:110:80 (Definition)

// Template template arguments
// CHECK: Keyword: "typedef" [126:3 - 126:10] ClassTemplate=X7:123:8 (Definition)
// CHECK: Identifier: "outer_alias" [126:11 - 126:22] NamespaceRef=outer_alias:10:11
// CHECK: Punctuation: "::" [126:22 - 126:24] TypedefDecl=type:126:74 (Definition)
// CHECK: Identifier: "inner" [126:24 - 126:29] NamespaceRef=inner:114:13
// CHECK: Punctuation: "::" [126:29 - 126:31] TypedefDecl=type:126:74 (Definition)
// CHECK: Identifier: "apply_meta" [126:31 - 126:41] TemplateRef=apply_meta:116:12
// CHECK: Punctuation: "<" [126:41 - 126:42] TypedefDecl=type:126:74 (Definition)
// CHECK: Identifier: "T_type" [126:42 - 126:48] TypeRef=T_type:124:13
// CHECK: Punctuation: "," [126:48 - 126:49] TypedefDecl=type:126:74 (Definition)
// CHECK: Identifier: "U_type" [126:50 - 126:56] TypeRef=U_type:125:13
// CHECK: Punctuation: "::" [126:56 - 126:58] TypedefDecl=type:126:74 (Definition)
// CHECK: Keyword: "template" [126:58 - 126:66] TypedefDecl=type:126:74 (Definition)
// CHECK: Identifier: "apply" [126:67 - 126:72] TypedefDecl=type:126:74 (Definition)
// CHECK: Punctuation: ">" [126:72 - 126:73] TypedefDecl=type:126:74 (Definition)
// CHECK: Identifier: "type" [126:74 - 126:78] TypedefDecl=type:126:74 (Definition)

// Member access expressions
// CHECK: Identifier: "inherited" [136:5 - 136:14] TypeRef=inherited:134:14
// CHECK: Punctuation: "::" [136:14 - 136:16] MemberRefExpr=f:130:8
// CHECK: Identifier: "f" [136:16 - 136:17] MemberRefExpr=f:130:8
