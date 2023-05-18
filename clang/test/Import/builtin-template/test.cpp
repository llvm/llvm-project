// RUN: clang-import-test -dump-ast -import %S/Inputs/S.cpp -expression %s -Xcc -DSEQ | FileCheck --check-prefix=CHECK-SEQ %s
// RUN: clang-import-test -dump-ast -import %S/Inputs/S.cpp -expression %s -Xcc -DPACK | FileCheck --check-prefix=CHECK-PACK %s
// RUN: clang-import-test -dump-ast -import %S/Inputs/S.cpp -expression %s -Xcc -DPACK -Xcc -DSEQ | FileCheck --check-prefixes=CHECK-SEQ,CHECK-PACK %s

// CHECK-SEQ: BuiltinTemplateDecl
// CHECK-SEQ-SAME: <invalid sloc>
// CHECK-SEQ-SAME: implicit
// CHECK-SEQ-SAME: __make_integer_seq

// CHECK-PACK: BuiltinTemplateDecl
// CHECK-PACK-SAME: <invalid sloc>
// CHECK-PACK-SAME: implicit
// CHECK-PACK-SAME: __type_pack_element

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
}
