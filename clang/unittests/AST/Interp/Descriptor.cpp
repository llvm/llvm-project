#include "../../../lib/AST/Interp/Descriptor.h"
#include "../../../lib/AST/Interp/Context.h"
#include "../../../lib/AST/Interp/Program.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace clang::interp;
using namespace clang::ast_matchers;

/// Inspect generated Descriptors as well as the pointers we create.
///
TEST(Descriptor, Primitives) {
  constexpr char Code[] =
      "struct A { bool a; bool b; };\n"
      "struct S {\n"
      "  float f;\n"
      "  char s[4];\n"
      "  A a[3];\n"
      "  short l[3][3];\n"
      "};\n"
      "constexpr S d = {0.0, \"foo\", {{true, false}, {false, true}, {false, false}},\n"
      "  {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}};\n";

  auto AST = tooling::buildASTFromCodeWithArgs(
      Code, {"-fexperimental-new-constant-interpreter"});

  const VarDecl *D = selectFirst<VarDecl>(
      "d", match(varDecl().bind("d"), AST->getASTContext()));
  ASSERT_NE(D, nullptr);

  const auto &Ctx = AST->getASTContext().getInterpContext();
  Program &Prog = Ctx.getProgram();
  // Global is registered.
  ASSERT_TRUE(Prog.getGlobal(D));

  // Get a Pointer to the global.
  const Pointer &GlobalPtr = Prog.getPtrGlobal(*Prog.getGlobal(D));

  // Test Descriptor of the struct S.
  const Descriptor *GlobalDesc = GlobalPtr.getFieldDesc();
  ASSERT_TRUE(GlobalDesc == GlobalPtr.getDeclDesc());

  ASSERT_TRUE(GlobalDesc->asDecl() == D);
  ASSERT_FALSE(GlobalDesc->asExpr());
  ASSERT_TRUE(GlobalDesc->asValueDecl() == D);
  ASSERT_FALSE(GlobalDesc->asFieldDecl());
  ASSERT_FALSE(GlobalDesc->asRecordDecl());

  // Still true because this is a global variable.
  ASSERT_TRUE(GlobalDesc->getMetadataSize() == sizeof(InlineDescriptor));
  ASSERT_FALSE(GlobalDesc->isPrimitiveArray());
  ASSERT_FALSE(GlobalDesc->isCompositeArray());
  ASSERT_FALSE(GlobalDesc->isZeroSizeArray());
  ASSERT_FALSE(GlobalDesc->isUnknownSizeArray());
  ASSERT_FALSE(GlobalDesc->isPrimitive());
  ASSERT_FALSE(GlobalDesc->isArray());
  ASSERT_TRUE(GlobalDesc->isRecord());

  // Test the Record for the struct S.
  const Record *SRecord = GlobalDesc->ElemRecord;
  ASSERT_TRUE(SRecord);
  ASSERT_TRUE(SRecord->getNumFields() == 4);
  ASSERT_TRUE(SRecord->getNumBases() == 0);
  ASSERT_FALSE(SRecord->getDestructor());

  // First field.
  const Record::Field *F1 = SRecord->getField(0u);
  ASSERT_TRUE(F1);
  ASSERT_FALSE(F1->isBitField());
  ASSERT_TRUE(F1->Desc->isPrimitive());

  // Second field.
  const Record::Field *F2 = SRecord->getField(1u);
  ASSERT_TRUE(F2);
  ASSERT_FALSE(F2->isBitField());
  ASSERT_TRUE(F2->Desc->isArray());
  ASSERT_FALSE(F2->Desc->isCompositeArray());
  ASSERT_TRUE(F2->Desc->isPrimitiveArray());
  ASSERT_FALSE(F2->Desc->isPrimitive());
  ASSERT_FALSE(F2->Desc->ElemDesc);
  ASSERT_EQ(F2->Desc->getNumElems(), 4u);
  ASSERT_TRUE(F2->Desc->getElemSize() > 0);

  // Third field.
  const Record::Field *F3 = SRecord->getField(2u);
  ASSERT_TRUE(F3);
  ASSERT_FALSE(F3->isBitField());
  ASSERT_TRUE(F3->Desc->isArray());
  ASSERT_TRUE(F3->Desc->isCompositeArray());
  ASSERT_FALSE(F3->Desc->isPrimitiveArray());
  ASSERT_FALSE(F3->Desc->isPrimitive());
  ASSERT_TRUE(F3->Desc->ElemDesc);
  ASSERT_EQ(F3->Desc->getNumElems(), 3u);
  ASSERT_TRUE(F3->Desc->getElemSize() > 0);

  // Fourth field.
  // Multidimensional arrays are treated as composite arrays, even
  // if the value type is primitive.
  const Record::Field *F4 = SRecord->getField(3u);
  ASSERT_TRUE(F4);
  ASSERT_FALSE(F4->isBitField());
  ASSERT_TRUE(F4->Desc->isArray());
  ASSERT_TRUE(F4->Desc->isCompositeArray());
  ASSERT_FALSE(F4->Desc->isPrimitiveArray());
  ASSERT_FALSE(F4->Desc->isPrimitive());
  ASSERT_TRUE(F4->Desc->ElemDesc);
  ASSERT_EQ(F4->Desc->getNumElems(), 3u);
  ASSERT_TRUE(F4->Desc->getElemSize() > 0);
  ASSERT_TRUE(F4->Desc->ElemDesc->isPrimitiveArray());

  // Check pointer stuff.
  // Global variables have an inline descriptor.
  ASSERT_FALSE(GlobalPtr.isRoot());
  ASSERT_TRUE(GlobalPtr.isLive());
  ASSERT_FALSE(GlobalPtr.isZero());
  ASSERT_FALSE(GlobalPtr.isField());
  ASSERT_TRUE(GlobalPtr.getFieldDesc() == GlobalPtr.getDeclDesc());
  ASSERT_TRUE(GlobalPtr.getOffset() == 0);
  ASSERT_FALSE(GlobalPtr.inArray());
  ASSERT_FALSE(GlobalPtr.isArrayElement());
  ASSERT_FALSE(GlobalPtr.isArrayRoot());
  ASSERT_FALSE(GlobalPtr.inPrimitiveArray());
  ASSERT_TRUE(GlobalPtr.isStatic());
  ASSERT_TRUE(GlobalPtr.isInitialized());
  ASSERT_FALSE(GlobalPtr.isOnePastEnd());
  ASSERT_FALSE(GlobalPtr.isElementPastEnd());

  // Pointer to the first field (a primitive).
  const Pointer &PF1 = GlobalPtr.atField(F1->Offset);
  ASSERT_TRUE(PF1.isLive());
  ASSERT_TRUE(PF1.isInitialized());
  ASSERT_TRUE(PF1.isField());
  ASSERT_FALSE(PF1.inArray());
  ASSERT_FALSE(PF1.isArrayElement());
  ASSERT_FALSE(PF1.isArrayRoot());
  ASSERT_FALSE(PF1.isOnePastEnd());
  ASSERT_FALSE(PF1.isRoot());
  ASSERT_TRUE(PF1.getFieldDesc()->isPrimitive());
  ASSERT_TRUE(Pointer::hasSameBase(PF1, GlobalPtr));
  ASSERT_TRUE(PF1.getBase() == GlobalPtr);

  // Pointer to the second field (a primitive array).
  const Pointer &PF2 = GlobalPtr.atField(F2->Offset);
  ASSERT_TRUE(PF2.isLive());
  ASSERT_TRUE(PF2.isInitialized());
  ASSERT_TRUE(PF2.isField());
  ASSERT_TRUE(PF2.inArray());
  ASSERT_FALSE(PF2.isArrayElement());
  ASSERT_TRUE(PF2.isArrayRoot());
  ASSERT_TRUE(PF2.getNumElems() == 4);
  ASSERT_FALSE(PF2.isOnePastEnd());
  ASSERT_FALSE(PF2.isRoot());
  ASSERT_FALSE(PF2.getFieldDesc()->isPrimitive());
  ASSERT_TRUE(PF2.getFieldDesc()->isArray());
  ASSERT_TRUE(Pointer::hasSameBase(PF2, GlobalPtr));
  ASSERT_TRUE(PF2.getBase() == GlobalPtr);

  // Check contents of field 2 (a primitive array).
  {
    const Pointer &E1 = PF2.atIndex(0);
    ASSERT_TRUE(E1.isLive());
    ASSERT_FALSE(E1.isArrayRoot());
    ASSERT_TRUE(E1.isArrayElement());
    ASSERT_TRUE(E1.inPrimitiveArray());
    ASSERT_TRUE(E1.deref<char>() == 'f');
    ASSERT_EQ(E1.getIndex(), 0u);
    ASSERT_TRUE(E1 == E1.atIndex(0));
    ASSERT_TRUE(Pointer::hasSameBase(E1, GlobalPtr));

    const Pointer &E2 = PF2.atIndex(1);
    ASSERT_TRUE(E2.isLive());
    ASSERT_FALSE(E2.isArrayRoot());
    ASSERT_TRUE(E2.isArrayElement());
    ASSERT_EQ(E2.getIndex(), 1u);
    // Narrow() doesn't do anything on primitive array elements, as there is
    // nothing to narrow into.
    ASSERT_EQ(E2.narrow(), E2);
    // ... so this should also hold.
    ASSERT_EQ(E2.expand(), E2);
    ASSERT_EQ(E2.narrow().expand(), E2);

    // .atIndex(1).atIndex(1) should be index 1.
    ASSERT_EQ(PF2.atIndex(1).atIndex(1), PF2.atIndex(1));
    ASSERT_EQ(PF2.atIndex(1).narrow().atIndex(1), PF2.atIndex(1));

    // getArray() should give us the array field again.
    ASSERT_EQ(E2.getArray(), PF2);

    // One-after-the-end pointer.
    const Pointer &O = PF2.atIndex(PF2.getNumElems());
    ASSERT_TRUE(O.isLive());
    ASSERT_TRUE(O.isOnePastEnd());
    ASSERT_TRUE(O.isInitialized());
    ASSERT_TRUE(O.getIndex() == PF2.getNumElems());
  }

  // Pointer to the third field (a composite array).
  const Pointer &PF3 = GlobalPtr.atField(F3->Offset);
  ASSERT_TRUE(PF3.isLive());
  ASSERT_TRUE(PF3.isInitialized());
  ASSERT_TRUE(PF3.isField());
  ASSERT_TRUE(PF3.inArray());
  ASSERT_TRUE(PF3.isArrayRoot());
  ASSERT_FALSE(PF3.isArrayElement());
  ASSERT_TRUE(PF3.getNumElems() == 3);
  ASSERT_FALSE(PF3.isOnePastEnd());
  ASSERT_FALSE(PF3.isRoot());
  ASSERT_FALSE(PF3.getFieldDesc()->isPrimitive());
  ASSERT_TRUE(PF3.getFieldDesc()->isArray());
  ASSERT_TRUE(Pointer::hasSameBase(PF3, GlobalPtr));
  ASSERT_TRUE(PF3.getBase() == GlobalPtr);
  ASSERT_EQ(PF3.getRecord(), nullptr);
  ASSERT_TRUE(PF3.getElemRecord());

  // Check contents of field 3 (a composite array).
  {
    const Pointer &E1 = PF3.atIndex(0);
    // Note that we didn't call narrow() above, so this points
    // to an array element and not just a field.
    ASSERT_TRUE(E1.isLive());
    ASSERT_EQ(E1.getIndex(), 0);
    ASSERT_TRUE(E1.isInitialized());
    ASSERT_TRUE(E1.isArrayElement());
    ASSERT_TRUE(E1.inArray());
    ASSERT_FALSE(E1.isArrayRoot());
    ASSERT_FALSE(E1.isRoot());
    ASSERT_EQ(E1.getArray(), PF3);
    ASSERT_TRUE(E1.isField());
    ASSERT_TRUE(E1.getElemRecord());
    ASSERT_FALSE(E1.getRecord());

    // Now the same with narrow().
    const Pointer &NE1 = PF3.atIndex(0).narrow();
    ASSERT_NE(E1, NE1);
    ASSERT_TRUE(NE1.isLive());
    ASSERT_EQ(NE1.getIndex(), 0);
    ASSERT_TRUE(NE1.isInitialized());
    ASSERT_FALSE(NE1.isArrayElement());
    ASSERT_TRUE(NE1.isField());
    ASSERT_FALSE(NE1.inArray());
    ASSERT_FALSE(NE1.isArrayRoot());
    ASSERT_FALSE(NE1.isRoot());
    // Not possible, since this is narrow()ed:
    // ASSERT_EQ(NE1.getArray(), PF3);
    ASSERT_EQ(NE1.expand(), E1);
    ASSERT_FALSE(NE1.getElemRecord());
    ASSERT_TRUE(NE1.getRecord());

    // Second element, NOT narrowed.
    const Pointer &E2 = PF3.atIndex(1);
    ASSERT_TRUE(E2.isLive());
    ASSERT_EQ(E2.getIndex(), 1);
    ASSERT_TRUE(E2.isInitialized());
    ASSERT_TRUE(E2.isArrayElement());
    ASSERT_TRUE(E2.isField());
    ASSERT_TRUE(E2.inArray());
    ASSERT_FALSE(E2.isArrayRoot());
    ASSERT_FALSE(E2.isRoot());
    ASSERT_EQ(E2.getArray(), PF3);

    // Second element, narrowed.
    const Pointer &NE2 = PF3.atIndex(1).narrow();
    ASSERT_TRUE(NE2.isLive());
    ASSERT_EQ(NE2.getIndex(), 0);
    ASSERT_TRUE(NE2.isInitialized());
    ASSERT_FALSE(NE2.isArrayElement());
    ASSERT_TRUE(NE2.isField());
    ASSERT_FALSE(NE2.inArray());
    ASSERT_FALSE(NE2.isArrayRoot());
    ASSERT_FALSE(NE2.isRoot());
    // Not possible, since this is narrow()ed:
    // ASSERT_EQ(NE2.getArray(), PF3);
    ASSERT_FALSE(NE2.getElemRecord());
    ASSERT_TRUE(NE2.getRecord());

    // Chained atIndex() without narrowing in between.
    ASSERT_EQ(PF3.atIndex(1).atIndex(1), PF3.atIndex(1));

    // First field of the second element.
    const Pointer &FP1 = NE2.atField(NE2.getRecord()->getField(0u)->Offset);
    ASSERT_TRUE(FP1.isLive());
    ASSERT_TRUE(FP1.isInitialized());
    ASSERT_EQ(FP1.getBase(), NE2);
    ASSERT_FALSE(FP1.isArrayElement());
    ASSERT_FALSE(FP1.inArray());
    ASSERT_FALSE(FP1.inPrimitiveArray());
    ASSERT_TRUE(FP1.isField());

    // One-past-the-end of a composite array.
    const Pointer &O = PF3.atIndex(PF3.getNumElems()).narrow();
    ASSERT_TRUE(O.isOnePastEnd());
    ASSERT_TRUE(O.isElementPastEnd());
  }

  // Pointer to the fourth field (a multidimensional primitive array).
  const Pointer &PF4 = GlobalPtr.atField(F4->Offset);
  ASSERT_TRUE(PF4.isLive());
  ASSERT_TRUE(PF4.isInitialized());
  ASSERT_TRUE(PF4.isField());
  ASSERT_TRUE(PF4.inArray());
  ASSERT_TRUE(PF4.isArrayRoot());
  ASSERT_FALSE(PF4.isArrayElement());
  ASSERT_TRUE(PF4.getNumElems() == 3);
  ASSERT_FALSE(PF4.isOnePastEnd());
  ASSERT_FALSE(PF4.isRoot());
  ASSERT_FALSE(PF4.getFieldDesc()->isPrimitive());
  ASSERT_TRUE(PF4.getFieldDesc()->isArray());
  ASSERT_TRUE(Pointer::hasSameBase(PF4, GlobalPtr));
  ASSERT_TRUE(PF4.getBase() == GlobalPtr);
  ASSERT_EQ(PF4.getRecord(), nullptr);
  ASSERT_EQ(PF4.getElemRecord(), nullptr);
  ASSERT_NE(PF4.getField(), nullptr);
  ASSERT_TRUE(PF4.getFieldDesc()->ElemDesc->isPrimitiveArray());
  // Check contents of field 4 (a primitive array).
  {
    // Pointer to the first element, is of type short[3].
    const Pointer &E1 = PF4.atIndex(0);
    ASSERT_NE(E1, PF4);
    ASSERT_TRUE(E1.isLive());
    ASSERT_TRUE(E1.isArrayElement());
    ASSERT_TRUE(E1.inArray());
    ASSERT_EQ(E1.getNumElems(), 3u);
    ASSERT_EQ(E1.getIndex(), 0u);
    ASSERT_EQ(E1.getArray(), PF4);

    // Now narrow()'ed.
    const Pointer &NE1 = PF4.atIndex(0).narrow();
    ASSERT_NE(NE1, PF4);
    ASSERT_NE(NE1, E1);
    ASSERT_TRUE(NE1.isLive());
    ASSERT_FALSE(NE1.isArrayElement());
    ASSERT_TRUE(NE1.isArrayRoot());
    ASSERT_FALSE(NE1.getFieldDesc()->isCompositeArray());
    ASSERT_TRUE(NE1.getFieldDesc()->isPrimitiveArray());
    ASSERT_EQ(NE1.getFieldDesc()->getNumElems(), 3u);
    ASSERT_TRUE(NE1.inArray());
    ASSERT_EQ(NE1.getNumElems(), 3u);
    ASSERT_EQ(NE1.getIndex(), 0u);

    // Last element of the first dimension.
    const Pointer &PE1 = PF4.atIndex(0).narrow().atIndex(2);
    ASSERT_TRUE(PE1.isLive());
    ASSERT_EQ(PE1.deref<short>(), 3);
    ASSERT_EQ(PE1.getArray(), NE1);
    ASSERT_EQ(PE1.getIndex(), 2u);

    // third dimension
    const Pointer &E3 = PF4.atIndex(2);
    ASSERT_NE(E3, PF4);
    ASSERT_TRUE(E3.isLive());
    ASSERT_TRUE(E3.isArrayElement());
    ASSERT_FALSE(E3.isArrayRoot());
    ASSERT_TRUE(E3.inArray());
    ASSERT_EQ(E3.getNumElems(), 3u);
    ASSERT_EQ(E3.getIndex(), 2u);

    // Same, but narrow()'ed.
    const Pointer &NE3 = PF4.atIndex(2).narrow();
    ASSERT_NE(NE3, PF4);
    ASSERT_NE(NE3, E1);
    ASSERT_TRUE(NE3.isLive());
    ASSERT_FALSE(NE3.isArrayElement());
    ASSERT_TRUE(NE3.isArrayRoot());
    ASSERT_FALSE(NE3.getFieldDesc()->isCompositeArray());
    ASSERT_TRUE(NE3.getFieldDesc()->isPrimitiveArray());
    ASSERT_EQ(NE3.getFieldDesc()->getNumElems(), 3u);
    ASSERT_TRUE(NE3.inArray());
    ASSERT_EQ(NE3.getNumElems(), 3u);
    // This is narrow()'ed, so not an "array elemnet"
    ASSERT_EQ(PF4.atIndex(2).getIndex(), 2u);
    ASSERT_EQ(NE3.getIndex(), 0u);

    // Last element of the last dimension
    const Pointer &PE3 = PF4.atIndex(2).narrow().atIndex(2);
    ASSERT_TRUE(PE3.isLive());
    ASSERT_EQ(PE3.deref<short>(), 9);
    ASSERT_EQ(PE3.getArray(), NE3);
    ASSERT_EQ(PE3.getIndex(), 2u);
  }
}
