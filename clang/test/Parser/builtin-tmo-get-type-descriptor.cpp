// RUN: %clang_cc1 -fsyntax-only -verify -Wno-unused %s
// RUN: %clang_cc1 -Rtmo-remarks -fsyntax-only -verify=expected,tmoremarks -Wno-unused %s

_Static_assert(__has_builtin(__builtin_tmo_get_type_descriptor), "No type descriptor builtin");

typedef unsigned __INT64_TYPE__ uint64_t;

struct Test {
  int x;
  void *p;
};

void bad_syntax(int var) {
  // expected-error@+1{{expected '('}}
  __builtin_tmo_get_type_descriptor;

  // expected-error@+1{{expected '('}}
  __builtin_tmo_get_type_descriptor{};

  // expected-error@+1{{expected a type}}
  __builtin_tmo_get_type_descriptor();

  // expected-error@+1{{expected '('}}
  __builtin_tmo_get_type_descriptor{var};

  // expected-error@+1{{expected '('}}
  __builtin_tmo_get_type_descriptor var;
}

void non_struct_type(int var) {
  __builtin_tmo_get_type_descriptor(int);
  // tmoremarks-remark@-1 {{__builtin_tmo_get_type_descriptor reported 'int' as 72057595422605840. Type semantics: { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 1384677904 }}}
  __builtin_tmo_get_type_descriptor(typeof(var));
  // tmoremarks-remark@-1 {{__builtin_tmo_get_type_descriptor reported 'typeof (var)' (aka 'int') as 72057595422605840. Type semantics: { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 1384677904 }}}
}

void void_type(const void *p) {
  // expected-error@+1{{invalid application of '__builtin_tmo_get_type_descriptor' to an incomplete type 'void'}}
  __builtin_tmo_get_type_descriptor(void);
  __builtin_tmo_get_type_descriptor(typeof(p));
  // tmoremarks-remark@-1 {{__builtin_tmo_get_type_descriptor reported 'typeof (p)' (aka 'const void *') as 3377702818697837. Type semantics: { "Summary": { "LayoutSemantics": [ "ImmutablePointer", "AnonymousPointer" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 3098169965 }}}
  __builtin_tmo_get_type_descriptor(typeof(&p));
  // tmoremarks-remark@-1 {{__builtin_tmo_get_type_descriptor reported 'typeof (&p)' (aka 'const void **') as 2251802906997560. Type semantics: { "Summary": { "LayoutSemantics": [ "AnonymousPointer" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 3093312312 }}}
  __builtin_tmo_get_type_descriptor(void **);
  // tmoremarks-remark@-1 {{__builtin_tmo_get_type_descriptor reported 'void **' as 2251802906997560. Type semantics: { "Summary": { "LayoutSemantics": [ "AnonymousPointer" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 3093312312 }}}
}

void forward_decl() {
  // expected-note@+1{{forward declaration of 'Forward'}}
  struct Forward;
  // expected-error@+1{{invalid application of '__builtin_tmo_get_type_descriptor' to an incomplete type 'Forward'}}
  __builtin_tmo_get_type_descriptor(Forward);
}

template <typename T> struct X {
  uint64_t s = __builtin_tmo_get_type_descriptor(T);
};

template <typename T> struct X_static {
  static uint64_t s;
};

template <typename T>
uint64_t X_static<T>::s = __builtin_tmo_get_type_descriptor(T);

template <int N> struct ValueTempl { int array[N]; };
uint64_t sv = __builtin_tmo_get_type_descriptor(ValueTempl<2>);
// tmoremarks-remark@-1 {{__builtin_tmo_get_type_descriptor reported 'ValueTempl<2>' as 72057594041155351. Type semantics: { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 3227415 }}}

ValueTempl<2> InstValueTempl;
uint64_t svi = __builtin_tmo_get_type_descriptor(typeof(InstValueTempl));
// tmoremarks-remark@-1 {{__builtin_tmo_get_type_descriptor reported 'typeof (InstValueTempl)' (aka 'ValueTempl<2>') as 72057594041155351. Type semantics: { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 3227415 }}}

template <typename T> struct TA { T t; };
struct TB {
  TA<int> f;
};
static uint64_t sTB = __builtin_tmo_get_type_descriptor(TB);
// tmoremarks-remark@-1 {{__builtin_tmo_get_type_descriptor reported 'TB' as 72057595422605840. Type semantics: { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 1384677904 }}}
void usage(struct Test s) {
  struct Test *p = &s;

  struct {
    char a;
    char b;
  } x;

  __builtin_tmo_get_type_descriptor(struct Test);
  // tmoremarks-remark@-1 {{__builtin_tmo_get_type_descriptor reported 'struct Test' as 74309395798930562. Type semantics: { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 1947317378 }}}
  __builtin_tmo_get_type_descriptor(typeof(s));
  // tmoremarks-remark@-1 {{__builtin_tmo_get_type_descriptor reported 'typeof (s)' (aka 'struct Test') as 74309395798930562. Type semantics: { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 1947317378 }}}
  __builtin_tmo_get_type_descriptor(typeof(*p));
  // tmoremarks-remark@-1 {{__builtin_tmo_get_type_descriptor reported 'typeof (*p)' (aka 'struct Test') as 74309395798930562. Type semantics: { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 1947317378 }}}
  __builtin_tmo_get_type_descriptor(typeof(x));
  // tmoremarks-remark@-1 {{as 72057597225271395. Type semantics: { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 3187343459 }}}
  __builtin_tmo_get_type_descriptor(X<int>);
  // tmoremarks-remark@-1 {{__builtin_tmo_get_type_descriptor reported 'X<int>' as 72057594041155351. Type semantics: { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 3227415 }}}
  __builtin_tmo_get_type_descriptor(X<void>);
  // tmoremarks-remark@-1 {{__builtin_tmo_get_type_descriptor reported 'X<void>' as 72057594041155351. Type semantics: { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 3227415 }}}
  __builtin_tmo_get_type_descriptor(X<ValueTempl<1>>);
  // tmoremarks-remark@-1 {{__builtin_tmo_get_type_descriptor reported 'X<ValueTempl<1>>' as 72057594041155351. Type semantics: { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 3227415 }}}
  __builtin_tmo_get_type_descriptor(void(*)(int));
  // tmoremarks-remark@-1 {{__builtin_tmo_get_type_descriptor reported 'void (*)(int)' as 2251802906997560. Type semantics: { "Summary": { "LayoutSemantics": [ "AnonymousPointer" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 3093312312 }}}
  __builtin_tmo_get_type_descriptor(typeof(&usage));
  // tmoremarks-remark@-1 {{__builtin_tmo_get_type_descriptor reported 'typeof (&usage)' (aka 'void (*)(struct Test)') as 2251802906997560. Type semantics: { "Summary": { "LayoutSemantics": [ "AnonymousPointer" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 3093312312 }}}
}
