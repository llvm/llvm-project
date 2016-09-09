// RUN: %clang_cc1 -triple x86_64-apple-darwin -debug-info-kind=standalone -o - -emit-llvm %s | FileCheck %s

// We had a bug in -fstandalone-debug where UnicodeString would not be completed
// when it was required to be complete. This orginally manifested as an
// assertion in CodeView emission on Windows with some dllexport stuff, but it's
// more general than that.

struct UnicodeString;
struct GetFwdDecl {
  static UnicodeString format;
};
GetFwdDecl force_fwd_decl;
struct UnicodeString {
private:
  virtual ~UnicodeString();
};
struct UseCompleteType {
  UseCompleteType();
  ~UseCompleteType();
  UnicodeString currencySpcAfterSym[1];
};
UseCompleteType require_complete;
// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "UnicodeString"
// CHECK-NOT:              DIFlagFwdDecl
// CHECK-SAME:             ){{$}}

namespace rdar14101097_1 { // see also PR16214
// Check that we emit debug info for the definition of a struct if the
// definition is available, even if it's used via a pointer wrapped in a
// typedef.
// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "foo"
// CHECK-NOT:              DIFlagFwdDecl
// CHECK-SAME:             ){{$}}
struct foo {
};

typedef foo *foop;

void bar() {
  foop f;
}
}

namespace rdar14101097_2 {
// As above, except trickier because we first encounter only a declaration of
// the type and no debug-info related use after we see the definition of the
// type.
// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "foo"
// CHECK-NOT:              DIFlagFwdDecl
// CHECK-SAME:             ){{$}}
struct foo;
void bar() {
  foo *f;
}
struct foo {
};
}
