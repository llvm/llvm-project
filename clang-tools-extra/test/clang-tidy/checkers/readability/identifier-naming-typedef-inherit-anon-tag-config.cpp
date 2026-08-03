// RUN: %check_clang_tidy -check-suffixes=TAGSTYLE,SHARED -std=c++17 %s \
// RUN:   readability-identifier-naming %t -- \
// RUN:   -config='{CheckOptions: { \
// RUN:     readability-identifier-naming.TypedefInheritAnonTagConfig: true, \
// RUN:     readability-identifier-naming.AbstractClassCase: CamelCase, \
// RUN:     readability-identifier-naming.ClassCase: CamelCase, \
// RUN:     readability-identifier-naming.EnumCase: CamelCase, \
// RUN:     readability-identifier-naming.EnumIgnoredRegexp: "ignored_.*", \
// RUN:     readability-identifier-naming.StructCase: lower_case, \
// RUN:     readability-identifier-naming.UnionCase: UPPER_CASE, \
// RUN:     readability-identifier-naming.TypeAliasCase: camelBack, \
// RUN:     readability-identifier-naming.TypedefCase: camelBack, \
// RUN:   }}'

// Re-running the check on the fixed file must not produce any further
// warning. Directly after the run it validates, because the runs below
// overwrite %t.cpp.
// RUN: clang-tidy %t.cpp -checks='-*,readability-identifier-naming' \
// RUN:   -warnings-as-errors='-*,readability-identifier-naming' \
// RUN:   -config='{CheckOptions: { \
// RUN:     readability-identifier-naming.TypedefInheritAnonTagConfig: true, \
// RUN:     readability-identifier-naming.AbstractClassCase: CamelCase, \
// RUN:     readability-identifier-naming.ClassCase: CamelCase, \
// RUN:     readability-identifier-naming.EnumCase: CamelCase, \
// RUN:     readability-identifier-naming.EnumIgnoredRegexp: "ignored_.*", \
// RUN:     readability-identifier-naming.StructCase: lower_case, \
// RUN:     readability-identifier-naming.UnionCase: UPPER_CASE, \
// RUN:     readability-identifier-naming.TypeAliasCase: camelBack, \
// RUN:     readability-identifier-naming.TypedefCase: camelBack, \
// RUN:   }}' -- -std=c++17

// RUN: %check_clang_tidy -check-suffixes=TYPEDEFSTYLE,SHARED -std=c++17 %s \
// RUN:   readability-identifier-naming %t -- \
// RUN:   -config='{CheckOptions: { \
// RUN:     readability-identifier-naming.AbstractClassCase: CamelCase, \
// RUN:     readability-identifier-naming.ClassCase: CamelCase, \
// RUN:     readability-identifier-naming.EnumCase: CamelCase, \
// RUN:     readability-identifier-naming.EnumIgnoredRegexp: "ignored_.*", \
// RUN:     readability-identifier-naming.StructCase: lower_case, \
// RUN:     readability-identifier-naming.UnionCase: UPPER_CASE, \
// RUN:     readability-identifier-naming.TypeAliasCase: camelBack, \
// RUN:     readability-identifier-naming.TypedefCase: camelBack, \
// RUN:   }}'

// RUN: %check_clang_tidy -check-suffixes=TYPEDEFSTYLE,SHARED -std=c++17 %s \
// RUN:   readability-identifier-naming %t -- \
// RUN:   -config='{CheckOptions: { \
// RUN:     readability-identifier-naming.TypedefInheritAnonTagConfig: true, \
// RUN:     readability-identifier-naming.TypeAliasCase: camelBack, \
// RUN:     readability-identifier-naming.TypedefCase: camelBack, \
// RUN:   }}'

// The typedef is the only name of the tag it defines, so it can inherit the
// style configured for that tag kind.

typedef enum { EV_ANON } my_enum;
// CHECK-MESSAGES-TAGSTYLE: :[[@LINE-1]]:26: warning: invalid case style for enum 'my_enum' [readability-identifier-naming]
// CHECK-MESSAGES-TYPEDEFSTYLE: :[[@LINE-2]]:26: warning: invalid case style for typedef 'my_enum' [readability-identifier-naming]
// CHECK-FIXES-TAGSTYLE: typedef enum { EV_ANON } MyEnum;
// CHECK-FIXES-TYPEDEFSTYLE: typedef enum { EV_ANON } myEnum;

typedef struct { int Field; } My_Struct;
// CHECK-MESSAGES-TAGSTYLE: :[[@LINE-1]]:31: warning: invalid case style for struct 'My_Struct' [readability-identifier-naming]
// CHECK-MESSAGES-TYPEDEFSTYLE: :[[@LINE-2]]:31: warning: invalid case style for typedef 'My_Struct' [readability-identifier-naming]
// CHECK-FIXES-TAGSTYLE: typedef struct { int Field; } my_struct;
// CHECK-FIXES-TYPEDEFSTYLE: typedef struct { int Field; } myStruct;

typedef union { int I; float F; } my_union;
// CHECK-MESSAGES-TAGSTYLE: :[[@LINE-1]]:35: warning: invalid case style for union 'my_union' [readability-identifier-naming]
// CHECK-MESSAGES-TYPEDEFSTYLE: :[[@LINE-2]]:35: warning: invalid case style for typedef 'my_union' [readability-identifier-naming]
// CHECK-FIXES-TAGSTYLE: typedef union { int I; float F; } MY_UNION;
// CHECK-FIXES-TYPEDEFSTYLE: typedef union { int I; float F; } myUnion;

typedef class { int Field; } my_class;
// CHECK-MESSAGES-TAGSTYLE: :[[@LINE-1]]:30: warning: invalid case style for class 'my_class' [readability-identifier-naming]
// CHECK-MESSAGES-TYPEDEFSTYLE: :[[@LINE-2]]:30: warning: invalid case style for typedef 'my_class' [readability-identifier-naming]
// CHECK-FIXES-TAGSTYLE: typedef class { int Field; } MyClass;
// CHECK-FIXES-TYPEDEFSTYLE: typedef class { int Field; } myClass;

typedef class { public: virtual void f() = 0; } my_abstract;
// CHECK-MESSAGES-TAGSTYLE: :[[@LINE-1]]:49: warning: invalid case style for abstract class 'my_abstract' [readability-identifier-naming]
// CHECK-MESSAGES-TYPEDEFSTYLE: :[[@LINE-2]]:49: warning: invalid case style for typedef 'my_abstract' [readability-identifier-naming]
// CHECK-FIXES-TAGSTYLE: typedef class { public: virtual void f() = 0; } MyAbstract;
// CHECK-FIXES-TYPEDEFSTYLE: typedef class { public: virtual void f() = 0; } myAbstract;

using my_alias_enum = enum { EV_ALIAS };
// CHECK-MESSAGES-TAGSTYLE: :[[@LINE-1]]:7: warning: invalid case style for enum 'my_alias_enum' [readability-identifier-naming]
// CHECK-MESSAGES-TYPEDEFSTYLE: :[[@LINE-2]]:7: warning: invalid case style for type alias 'my_alias_enum' [readability-identifier-naming]
// CHECK-FIXES-TAGSTYLE: using MyAliasEnum = enum { EV_ALIAS };
// CHECK-FIXES-TYPEDEFSTYLE: using myAliasEnum = enum { EV_ALIAS };

// The whole style of the tag kind is inherited, not just its case, so the
// ignored regexp of the enum applies here.

typedef enum { EV_IGNORED } ignored_enum_t;
// CHECK-MESSAGES-TYPEDEFSTYLE: :[[@LINE-1]]:29: warning: invalid case style for typedef 'ignored_enum_t' [readability-identifier-naming]
// CHECK-FIXES-TAGSTYLE: typedef enum { EV_IGNORED } ignored_enum_t;
// CHECK-FIXES-TYPEDEFSTYLE: typedef enum { EV_IGNORED } ignoredEnumT;

// The tag has a name of its own, so the typedef is just an alias for it and
// keeps the typedef style.

typedef enum Kind { EV_NAMED } my_kind;
// CHECK-MESSAGES-SHARED: :[[@LINE-1]]:32: warning: invalid case style for typedef 'my_kind' [readability-identifier-naming]
// CHECK-FIXES-SHARED: typedef enum Kind { EV_NAMED } myKind;

using my_kind_alias = Kind;
// CHECK-MESSAGES-SHARED: :[[@LINE-1]]:7: warning: invalid case style for type alias 'my_kind_alias' [readability-identifier-naming]
// CHECK-FIXES-SHARED: using myKindAlias = Kind;

typedef struct data { int Field; } my_data;
// CHECK-MESSAGES-SHARED: :[[@LINE-1]]:36: warning: invalid case style for typedef 'my_data' [readability-identifier-naming]
// CHECK-FIXES-SHARED: typedef struct data { int Field; } myData;

// Of several declarators, the first one that denotes the tag type itself names
// the tag. The others are ordinary typedefs.

typedef enum { EV_MULTI } FirstEnum, second_enum;
// CHECK-MESSAGES-TYPEDEFSTYLE: :[[@LINE-1]]:27: warning: invalid case style for typedef 'FirstEnum' [readability-identifier-naming]
// CHECK-MESSAGES-SHARED: :[[@LINE-2]]:38: warning: invalid case style for typedef 'second_enum' [readability-identifier-naming]
// CHECK-FIXES-TAGSTYLE: typedef enum { EV_MULTI } FirstEnum, secondEnum;
// CHECK-FIXES-TYPEDEFSTYLE: typedef enum { EV_MULTI } firstEnum, secondEnum;

typedef struct { int Field; } *first_ptr, second_struct;
// CHECK-MESSAGES-SHARED: :[[@LINE-1]]:32: warning: invalid case style for typedef 'first_ptr' [readability-identifier-naming]
// CHECK-MESSAGES-TYPEDEFSTYLE: :[[@LINE-2]]:43: warning: invalid case style for typedef 'second_struct' [readability-identifier-naming]
// CHECK-FIXES-TAGSTYLE: typedef struct { int Field; } *firstPtr, second_struct;
// CHECK-FIXES-TYPEDEFSTYLE: typedef struct { int Field; } *firstPtr, secondStruct;

// The typedef does not name a tag type at all.

typedef struct { int Field; } *my_struct_ptr;
// CHECK-MESSAGES-SHARED: :[[@LINE-1]]:32: warning: invalid case style for typedef 'my_struct_ptr' [readability-identifier-naming]
// CHECK-FIXES-SHARED: typedef struct { int Field; } *myStructPtr;

typedef int my_int;
// CHECK-MESSAGES-SHARED: :[[@LINE-1]]:13: warning: invalid case style for typedef 'my_int' [readability-identifier-naming]
// CHECK-FIXES-SHARED: typedef int myInt;

// A typedef of a typedef does not name the tag either.

typedef My_Struct my_struct_alias;
// CHECK-MESSAGES-SHARED: :[[@LINE-1]]:19: warning: invalid case style for typedef 'my_struct_alias' [readability-identifier-naming]
// CHECK-FIXES-TAGSTYLE: typedef my_struct myStructAlias;
// CHECK-FIXES-TYPEDEFSTYLE: typedef myStruct myStructAlias;

// The typedef names the tag in the template pattern as well as in its
// instantiations, so it is reported only once.

template <typename T>
struct holder {
  typedef enum { EV_TPL } inner_enum;
  // CHECK-MESSAGES-TAGSTYLE: :[[@LINE-1]]:27: warning: invalid case style for enum 'inner_enum' [readability-identifier-naming]
  // CHECK-MESSAGES-TYPEDEFSTYLE: :[[@LINE-2]]:27: warning: invalid case style for typedef 'inner_enum' [readability-identifier-naming]
  // CHECK-FIXES-TAGSTYLE: typedef enum { EV_TPL } InnerEnum;
  // CHECK-FIXES-TYPEDEFSTYLE: typedef enum { EV_TPL } innerEnum;
};

template struct holder<int>;
