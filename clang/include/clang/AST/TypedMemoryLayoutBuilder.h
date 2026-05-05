#ifndef LLVM_CLANG_AST_TYPEDMEMORYLAYOUTBUILDER_H
#define LLVM_CLANG_AST_TYPEDMEMORYLAYOUTBUILDER_H

#include "clang/AST/Decl.h"
#include "clang/AST/Type.h"
#include <vector>

namespace clang {

struct TypedMemoryLayoutField {
  QualType Type;
  uint64_t Offset;
  uint64_t Width;
  const Decl *TypeOrFieldDecl;

  TypedMemoryLayoutField(QualType T, uint64_t O, uint64_t W, const Decl *D)
      : Type(T), Offset(O), Width(W), TypeOrFieldDecl(D) {}
};

using TypedMemoryFieldsVec = std::vector<TypedMemoryLayoutField>;

struct TypedMemoryLayout {
  uint64_t Width;
  TypedMemoryFieldsVec Fields;
};

class TypedMemoryLayoutBuilder {

public:
  TypedMemoryLayoutBuilder(const TypedMemoryLayoutBuilder &) = delete;
  TypedMemoryLayoutBuilder &
  operator=(const TypedMemoryLayoutBuilder &) = delete;
  TypedMemoryLayoutBuilder(const ASTContext &Ctx, QualType QT);
  ~TypedMemoryLayoutBuilder() = default;

  void build();

  const TypedMemoryLayout &getLayout() const { return Layout; }
  const TypedMemoryFieldsVec &getFields() const { return Layout.Fields; }

protected:
  const ASTContext &Context;
  const QualType QType;
  TypedMemoryLayout Layout;
};

} // namespace clang

#endif
