// TODO: ADD HEADER

#include "CIRGenCXXABI.h"

#include "clang/AST/GlobalDecl.h"
#include "clang/AST/Mangle.h"

using namespace cir;
using namespace clang;

CIRGenCXXABI::~CIRGenCXXABI() {}

CIRGenCXXABI::AddedStructorArgCounts CIRGenCXXABI::addImplicitConstructorArgs(
    CIRGenFunction &CGF, const clang::CXXConstructorDecl *D,
    clang::CXXCtorType Type, bool ForVirtualBase, bool Delegating,
    CallArgList &Args) {
  auto AddedArgs =
      getImplicitConstructorArgs(CGF, D, Type, ForVirtualBase, Delegating);
  for (size_t i = 0; i < AddedArgs.Prefix.size(); ++i)
    Args.insert(Args.begin() + 1 + i,
                CallArg(RValue::get(AddedArgs.Prefix[i].Value),
                        AddedArgs.Prefix[i].Type));
  for (const auto &arg : AddedArgs.Suffix)
    Args.add(RValue::get(arg.Value), arg.Type);
  return AddedStructorArgCounts(AddedArgs.Prefix.size(),
                                AddedArgs.Suffix.size());
}

bool CIRGenCXXABI::NeedsVTTParameter(GlobalDecl GD) { return false; }
