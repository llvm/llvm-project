#include "../../../lib/AST/ByteCode/Context.h"
#include "../../../lib/AST/ByteCode/Descriptor.h"
#include "../../../lib/AST/ByteCode/Integral.h"
#include "../../../lib/AST/ByteCode/Program.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace clang::interp;
using namespace clang::ast_matchers;

TEST(Pointer, TypesRecord) {
  constexpr char Code[] = "struct A { bool a; bool b; };\n"
                          "constexpr A arr[3][2] = {\n"
                          "  {{ false, false }, {false, true} },\n"
                          "  {{ false, false }, {false, true} },\n"
                          "  {{ false, false }, {false, true} },\n"
                          "};\n";

  auto AST = tooling::buildASTFromCodeWithArgs(
      Code, {"-fexperimental-new-constant-interpreter"});
  ASTContext &ASTCtx = AST->getASTContext();
  const VarDecl *D =
      selectFirst<VarDecl>("arr", match(varDecl().bind("arr"), ASTCtx));
  ASSERT_NE(D, nullptr);

  const auto &Ctx = AST->getASTContext().getInterpContext();
  Program &Prog = Ctx.getProgram();
  // Global is registered.
  ASSERT_TRUE(Prog.getGlobal(D));

  // Get a Pointer to the global.
  const Pointer &GlobalPtr = Prog.getPtrGlobal(*Prog.getGlobal(D));

  // Type of the global ptr should be A[][]
  {
    QualType T = GlobalPtr.getType();
    ASSERT_TRUE(T->isArrayType());
    const auto *ArrTy = cast<ConstantArrayType>(T->getAsArrayTypeUnsafe());
    ASSERT_NE(ArrTy, nullptr);
    ASSERT_EQ(ArrTy->getZExtSize(), (uint64_t)3);

    QualType ElemTy = ArrTy->getElementType();
    const auto *ElemArrTy =
        cast<ConstantArrayType>(ElemTy->getAsArrayTypeUnsafe());
    ASSERT_NE(ElemArrTy, nullptr);
    ASSERT_EQ(ElemArrTy->getZExtSize(), (uint64_t)2);
  }

  // This is still A[][] because we didn't narrow().
  {
    Pointer Elem = GlobalPtr.atIndex(0);
    QualType T = Elem.getType();
    const auto *ArrTy = cast<ConstantArrayType>(T->getAsArrayTypeUnsafe());
    ASSERT_NE(ArrTy, nullptr);
    ASSERT_EQ(ArrTy->getZExtSize(), (uint64_t)3);

    QualType ElemTy = ArrTy->getElementType();
    const auto *ElemArrTy =
        cast<ConstantArrayType>(ElemTy->getAsArrayTypeUnsafe());
    ASSERT_NE(ElemArrTy, nullptr);
    ASSERT_EQ(ElemArrTy->getZExtSize(), (uint64_t)2);
  }

  // Now with narrow(). This is just A[2], the type of the first element.
  {
    Pointer Elem = GlobalPtr.atIndex(0).narrow();
    QualType T = Elem.getType();
    const auto *ArrTy = cast<ConstantArrayType>(T->getAsArrayTypeUnsafe());
    ASSERT_NE(ArrTy, nullptr);
    ASSERT_EQ(ArrTy->getZExtSize(), (uint64_t)2);
  }

  // Same with element 1.
  {
    Pointer Elem = GlobalPtr.atIndex(1).narrow();
    QualType T = Elem.getType();
    const auto *ArrTy = cast<ConstantArrayType>(T->getAsArrayTypeUnsafe());
    ASSERT_NE(ArrTy, nullptr);
    ASSERT_EQ(ArrTy->getZExtSize(), (uint64_t)2);
  }
  // And 2.
  {
    Pointer Elem = GlobalPtr.atIndex(2).narrow();
    QualType T = Elem.getType();
    const auto *ArrTy = cast<ConstantArrayType>(T->getAsArrayTypeUnsafe());
    ASSERT_NE(ArrTy, nullptr);
    ASSERT_EQ(ArrTy->getZExtSize(), (uint64_t)2);
  }

  // This is arr[I][0], but we didn't narrow() at the end so the type is that of
  // arr[I].
  {
    for (unsigned I = 0; I != 3; ++I) {
      Pointer Elem = GlobalPtr.atIndex(I).narrow().atIndex(0);
      QualType T = Elem.getType();
      const auto *ArrTy = cast<ConstantArrayType>(T->getAsArrayTypeUnsafe());
      ASSERT_NE(ArrTy, nullptr);
      ASSERT_EQ(ArrTy->getZExtSize(), (uint64_t)2);
    }
  }
  // Now WITH a narrow, the type should be just A.
  {
    for (unsigned I = 0; I != 3; ++I) {
      for (unsigned J = 0; J != 2; ++J) {
        Pointer Elem = GlobalPtr.atIndex(I).narrow().atIndex(J).narrow();
        QualType T = Elem.getType();
        ASSERT_TRUE(T->isRecordType());
      }
    }
  }

  // Same as the above, but we expand() again to undo the last narrow.
  // The type should therefore be A[].
  {
    for (unsigned I = 0; I != 3; ++I) {
      for (unsigned J = 0; J != 2; ++J) {
        Pointer Elem =
            GlobalPtr.atIndex(I).narrow().atIndex(J).narrow().expand();
        QualType T = Elem.getType();
        const auto *ArrTy = cast<ConstantArrayType>(T->getAsArrayTypeUnsafe());
        ASSERT_NE(ArrTy, nullptr);
        ASSERT_EQ(ArrTy->getZExtSize(), (uint64_t)2);
      }
    }
  }

  // Check narrow/expand invariants.
  {
    for (unsigned I = 0; I != 3; ++I) {
      for (unsigned J = 0; J != 2; ++J) {
        ASSERT_EQ(GlobalPtr.atIndex(I).narrow().atIndex(J),
                  GlobalPtr.atIndex(I).narrow().atIndex(J).narrow().expand());
        ASSERT_EQ(GlobalPtr.atIndex(I).narrow().atIndex(J), GlobalPtr.atIndex(I)
                                                                .narrow()
                                                                .atIndex(J)
                                                                .narrow()
                                                                .expand()
                                                                .expand());
        ASSERT_EQ(GlobalPtr.atIndex(I), GlobalPtr.atIndex(I).narrow().expand());
        ASSERT_EQ(GlobalPtr.atIndex(I),
                  GlobalPtr.atIndex(I).narrow().expand().expand());
        ASSERT_EQ(GlobalPtr.atIndex(I).atIndex(1), GlobalPtr.atIndex(1));
      }
    }
  }

  // getIndex()
  {
    // First dimension.
    ASSERT_EQ(GlobalPtr.atIndex(0).getIndex(), 0);
    ASSERT_EQ(GlobalPtr.atIndex(1).getIndex(), 1);
    ASSERT_EQ(GlobalPtr.atIndex(2).getIndex(), 2);

    // First dimension, with narrow().
    ASSERT_EQ(GlobalPtr.atIndex(0).narrow().getIndex(), 0);
    ASSERT_EQ(GlobalPtr.atIndex(1).narrow().getIndex(), 0);
    ASSERT_EQ(GlobalPtr.atIndex(2).narrow().getIndex(), 0);

    // Second dimension.
    ASSERT_EQ(GlobalPtr.atIndex(0).narrow().atIndex(0).getIndex(), 0);
    ASSERT_EQ(GlobalPtr.atIndex(0).narrow().atIndex(1).getIndex(), 1);
    ASSERT_EQ(GlobalPtr.atIndex(1).narrow().atIndex(0).getIndex(), 0);
    ASSERT_EQ(GlobalPtr.atIndex(1).narrow().atIndex(1).getIndex(), 1);
    ASSERT_EQ(GlobalPtr.atIndex(2).narrow().atIndex(0).getIndex(), 0);
    ASSERT_EQ(GlobalPtr.atIndex(2).narrow().atIndex(1).getIndex(), 1);

    // Second dimension, with narrow().
    ASSERT_EQ(GlobalPtr.atIndex(0).narrow().atIndex(0).narrow().getIndex(), 0);
    ASSERT_EQ(GlobalPtr.atIndex(0).narrow().atIndex(1).narrow().getIndex(), 0);
    ASSERT_EQ(GlobalPtr.atIndex(1).narrow().atIndex(0).narrow().getIndex(), 0);
    ASSERT_EQ(GlobalPtr.atIndex(1).narrow().atIndex(1).narrow().getIndex(), 0);
    ASSERT_EQ(GlobalPtr.atIndex(2).narrow().atIndex(0).narrow().getIndex(), 0);
    ASSERT_EQ(GlobalPtr.atIndex(2).narrow().atIndex(1).narrow().getIndex(), 0);
  }
}

TEST(Pointer, TypesPrimitive) {
  constexpr char Code[] = "constexpr int arr[3][2] = {\n"
                          " { 1, 2 },\n"
                          " { 3, 4 },\n"
                          " { 5, 6 },\n"
                          "};\n";

  auto AST = tooling::buildASTFromCodeWithArgs(
      Code, {"-fexperimental-new-constant-interpreter"});
  ASTContext &ASTCtx = AST->getASTContext();
  const VarDecl *D =
      selectFirst<VarDecl>("arr", match(varDecl().bind("arr"), ASTCtx));
  ASSERT_NE(D, nullptr);

  const auto &Ctx = AST->getASTContext().getInterpContext();
  Program &Prog = Ctx.getProgram();
  // Global is registered.
  ASSERT_TRUE(Prog.getGlobal(D));

  // Get a Pointer to the global.
  const Pointer &GlobalPtr = Prog.getPtrGlobal(*Prog.getGlobal(D));

  // Type of the global ptr should be int[3][2].
  {
    QualType T = GlobalPtr.getType();
    ASSERT_TRUE(T->isArrayType());
    const auto *ArrTy = cast<ConstantArrayType>(T->getAsArrayTypeUnsafe());
    ASSERT_NE(ArrTy, nullptr);
    ASSERT_EQ(ArrTy->getZExtSize(), (uint64_t)3);

    QualType ElemTy = ArrTy->getElementType();
    const auto *ElemArrTy =
        cast<ConstantArrayType>(ElemTy->getAsArrayTypeUnsafe());
    ASSERT_NE(ElemArrTy, nullptr);
    ASSERT_EQ(ElemArrTy->getZExtSize(), (uint64_t)2);
  }

  // Type of the elements of the first dimension should be int[2].
  {
    for (unsigned I = 0; I != 3; ++I) {
      Pointer Elem = GlobalPtr.atIndex(I).narrow();
      QualType T = Elem.getType();
      const auto *ArrTy = cast<ConstantArrayType>(T->getAsArrayTypeUnsafe());
      ASSERT_NE(ArrTy, nullptr);
      ASSERT_EQ(ArrTy->getZExtSize(), (uint64_t)2);
    }
  }

  // Inner dimension is just int.
  {
    for (unsigned I = 0; I != 3; ++I) {
      for (unsigned J = 0; J != 2; ++J) {
        Pointer Elem = GlobalPtr.atIndex(I).narrow().atIndex(J);
        QualType T = Elem.getType();
        ASSERT_TRUE(T->isIntegerType());
      }
    }
  }

  {
    // First dimension.
    ASSERT_EQ(GlobalPtr.atIndex(0).getIndex(), 0);
    ASSERT_EQ(GlobalPtr.atIndex(1).getIndex(), 1);
    ASSERT_EQ(GlobalPtr.atIndex(2).getIndex(), 2);

    // First dimension, with narrow().
    ASSERT_EQ(GlobalPtr.atIndex(0).narrow().getIndex(), 0);
    ASSERT_EQ(GlobalPtr.atIndex(1).narrow().getIndex(), 0);
    ASSERT_EQ(GlobalPtr.atIndex(2).narrow().getIndex(), 0);

    // Second dimension.
    ASSERT_EQ(GlobalPtr.atIndex(0).narrow().atIndex(0).getIndex(), 0);
    ASSERT_EQ(GlobalPtr.atIndex(0).narrow().atIndex(1).getIndex(), 1);
    ASSERT_EQ(GlobalPtr.atIndex(1).narrow().atIndex(0).getIndex(), 0);
    ASSERT_EQ(GlobalPtr.atIndex(1).narrow().atIndex(1).getIndex(), 1);
    ASSERT_EQ(GlobalPtr.atIndex(2).narrow().atIndex(0).getIndex(), 0);
    ASSERT_EQ(GlobalPtr.atIndex(2).narrow().atIndex(1).getIndex(), 1);
  }

  {
    // narrow() does not affect pointers into primitive arrays.
    ASSERT_EQ(GlobalPtr.atIndex(0).narrow().atIndex(1),
              GlobalPtr.atIndex(0).narrow().atIndex(1).narrow());
    ASSERT_EQ(GlobalPtr.atIndex(0).narrow().atIndex(1),
              GlobalPtr.atIndex(0).narrow().atIndex(1).expand());

    ASSERT_EQ(GlobalPtr.atIndex(0).narrow(),
              GlobalPtr.atIndex(0).narrow().atIndex(1).getArray());

    ASSERT_EQ(GlobalPtr, GlobalPtr.atIndex(2).getArray());
    ASSERT_EQ(GlobalPtr, GlobalPtr.atIndex(2).narrow().expand().getArray());
  }
}
