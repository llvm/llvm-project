#ifndef LLVM_CLANG_BASIC_ATTRIBUTESCOPEINFO_H
#define LLVM_CLANG_BASIC_ATTRIBUTESCOPEINFO_H

#include "clang/Basic/SourceLocation.h"

namespace clang {

class IdentifierInfo;

class AttributeScopeInfo {
public:
  AttributeScopeInfo() = default;

  AttributeScopeInfo(const IdentifierInfo *Name, SourceLocation NameLoc)
      : Name(Name), NameLoc(NameLoc) {}

  AttributeScopeInfo(const IdentifierInfo *Name, SourceLocation NameLoc,
                     SourceLocation CommonScopeLoc)
      : Name(Name), NameLoc(NameLoc), CommonScopeLoc(CommonScopeLoc) {}

  const IdentifierInfo *getName() const { return Name; }
  SourceLocation getNameLoc() const { return NameLoc; }

  bool isValid() const { return Name != nullptr; }
  bool isExplicit() const { return CommonScopeLoc.isInvalid(); }

private:
  const IdentifierInfo *Name = nullptr;
  SourceLocation NameLoc;
  SourceLocation CommonScopeLoc;
};

} // namespace clang

#endif // LLVM_CLANG_BASIC_ATTRIBUTESCOPEINFO_H
