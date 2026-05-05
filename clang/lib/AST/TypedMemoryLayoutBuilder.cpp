#include "clang/AST/TypedMemoryLayoutBuilder.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/RecordLayout.h"
#include "llvm/ADT/SmallSet.h"
#include <cassert>

using namespace clang;

static TypedMemoryFieldsVec CollectFields(const ASTContext &Context,
                                          QualType QT, const Decl *Parent);

class TypedMemoryRecordVisitor
    : public ConstDeclVisitor<TypedMemoryRecordVisitor> {
  const ASTContext &Context;

public:
  TypedMemoryFieldsVec Fields;

  TypedMemoryRecordVisitor(const ASTContext &Ctx) : Context(Ctx) {}

  void VisitRecordDecl(const RecordDecl *D) {
    if (D->isInvalidDecl()) {
      Fields.clear();
      return;
    }

    if (const CXXRecordDecl *CXXRD = dyn_cast_or_null<CXXRecordDecl>(D)) {
      Visit(CXXRD);
    } else {
      for (auto F : D->fields())
        Visit(F);
    }
  }

  void VisitCXXRecordDecl(const CXXRecordDecl *D) {
    processCXXDecl(D, 0, true);
  }

  void VisitFieldDecl(const FieldDecl *D) { processFieldDecl(D, 0); }

  void processCXXDecl(const CXXRecordDecl *CXXRD, int64_t BaseClassOffset,
                      bool ProcessVBases) {
    if (CXXRD->isInvalidDecl()) {
      Fields.clear();
      return;
    }

    const ASTRecordLayout &RL = Context.getASTRecordLayout(CXXRD);
    if ((CXXRD->isDynamicClass() && !RL.getPrimaryBase()) || RL.hasOwnVFPtr()) {
      // add the vtable pointer
      Fields.emplace_back(Context.VoidPtrTy, BaseClassOffset,
                          Context.getTypeSize(Context.VoidPtrTy),
                          /* Parent */ nullptr);
    }

    // process non-virtual bases
    for (const CXXBaseSpecifier &BaseSpec : CXXRD->bases()) {
      if (!BaseSpec.isVirtual()) {
        const CXXRecordDecl *Base = BaseSpec.getType()->getAsCXXRecordDecl();
        int64_t OffsetInBits = RL.getBaseClassOffset(Base).getQuantity() * 8;
        processCXXDecl(Base, BaseClassOffset + OffsetInBits,
                       /*ProcessVBases=*/false);
      }
    }

    // process fields
    for (auto F : CXXRD->fields())
      processFieldDecl(F, BaseClassOffset);

    // process virtual bases
    if (ProcessVBases) {
      for (const CXXBaseSpecifier &BaseSpec : CXXRD->vbases()) {
        const CXXRecordDecl *VBase = BaseSpec.getType()->getAsCXXRecordDecl();
        int64_t OffsetInBits = RL.getVBaseClassOffset(VBase).getQuantity() * 8;
        processCXXDecl(VBase, BaseClassOffset + OffsetInBits,
                       /*ProcessVBases=*/false);
      }
    }
  }

  void processFieldDecl(const FieldDecl *D, int64_t BaseClassOffset) {
    if (D->isInvalidDecl() || D->getParent()->isInvalidDecl()) {
      Fields.clear();
      return;
    }

    const ASTRecordLayout *RL = &Context.getASTRecordLayout(D->getParent());
    const QualType &QT = D->getType();
    uint64_t FieldOffset =
        RL->getFieldOffset(D->getFieldIndex()) + BaseClassOffset;
    if (D->isBitField()) {
      Fields.emplace_back(QT, FieldOffset, D->getBitWidthValue(),
                          dyn_cast<Decl>(D));
    } else {
      for (auto &F : CollectFields(Context, QT, dyn_cast<Decl>(D))) {
        Fields.emplace_back(F.Type, F.Offset + FieldOffset, F.Width,
                            F.TypeOrFieldDecl);
      }
    }
  }
};

static TypedMemoryFieldsVec CollectFields(const ASTContext &Context,
                                          QualType QT, const Decl *Parent) {
  TypedMemoryFieldsVec Fields;
  if (QT->isRecordType()) {
    if (!QT->getAsRecordDecl()->isInvalidDecl()) {
      TypedMemoryRecordVisitor RV(Context);
      RV.Visit(QT->getAs<RecordType>()->getDecl());
      Fields = std::move(RV.Fields);
    }
  } else if (QT->isConstantArrayType()) {
    const ConstantArrayType *CAT = Context.getAsConstantArrayType(QT);
    QualType ElemType = CAT->getElementType();
    unsigned ElemWidth = Context.getTypeSize(ElemType);
    uint64_t Count = CAT->getSize().getZExtValue();
    assert(Context.getTypeSize(QT) >= Count * ElemWidth);
    auto ElemFields = CollectFields(Context, ElemType, /* Parent */ nullptr);
    for (uint64_t I = 0; I < Count; I++) {
      for (auto &EF : ElemFields) {
        Fields.emplace_back(EF.Type, EF.Offset + I * ElemWidth, EF.Width,
                            Parent);
      }
    }
  } else {
    Fields.emplace_back(QT, 0, Context.getTypeSize(QT), Parent);
  }
  return Fields;
}

TypedMemoryLayoutBuilder::TypedMemoryLayoutBuilder(const ASTContext &Ctx,
                                                   QualType QT)
    : Context(Ctx), QType(QT), Layout({}) {}

void TypedMemoryLayoutBuilder::build() {
  assert(Layout.Fields.empty());
  Layout.Width = Context.getTypeSize(QType);
  Layout.Fields = CollectFields(Context, QType, /* Parent */ nullptr);
}
