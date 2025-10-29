// RUN: clang-import-test -dump-ast -import %S/Inputs/S.cpp -expression %s -Xcc -DSEQ | FileCheck --check-prefix=CHECK-SEQ %s
// RUN: clang-import-test -dump-ast -import %S/Inputs/S.cpp -expression %s -Xcc -DPACK | FileCheck --check-prefix=CHECK-PACK %s
// RUN: clang-import-test -dump-ast -import %S/Inputs/S.cpp -expression %s -Xcc -DDEDUP | FileCheck --check-prefix=CHECK-DEDUP %s
// RUN: clang-import-test -dump-ast -import %S/Inputs/S.cpp -expression %s -Xcc -DPACK -Xcc -DSEQ -Xcc -DDEDUP | FileCheck --check-prefixes=CHECK-SEQ,CHECK-PACK,CHECK-DEDUP %s

// CHECK-SEQ:  BuiltinTemplateDecl {{.+}} <<invalid sloc>> <invalid sloc> implicit __make_integer_seq{{$}}
// CHECK-PACK: BuiltinTemplateDecl {{.+}} <<invalid sloc>> <invalid sloc> implicit __type_pack_element{{$}}
// CHECK-DEDUP: BuiltinTemplateDecl {{.+}} <<invalid sloc>> <invalid sloc> implicit __builtin_dedup_pack{{$}}

void expr() {
#ifdef SEQ
  typedef MakeSeq<int, 3> M1;
  M1 m1;
  typedef MakeSeq<long, 4> M2;
  M2 m2;
  static_assert(M1::PackSize == 3, "");
  static_assert(M2::PackSize == 4, "");
#endif

#ifdef PACK
  static_assert(__is_same(TypePackElement<0, X<0>>, X<0>), "");
  static_assert(__is_same(TypePackElement<0, X<0>, X<1>>, X<0>), "");
  static_assert(__is_same(TypePackElement<1, X<0>, X<1>>, X<1>), "");
#endif

#ifdef DEDUP
  static_assert(__is_same(TypePackDedup<TypeList>, TypeList<>), "");
  static_assert(__is_same(TypePackDedup<TypeList, int, double, int>, TypeList<int, double>), "");
  static_assert(!__is_same(TypePackDedup<TypeList, int, double, int>, TypeList<double, int>), "");
  static_assert(__is_same(TypePackDedup<TypeList, X<0>, X<1>, X<1>, X<2>, X<0>>, TypeList<X<0>, X<1>, X<2>>), "");
  static_assert(__is_same(TypePackDedup<TypeList, X0, SameAsX<1>, X<1>, X<0>>, TypeList<X<0>,X<1>>), "");
#endif
}
