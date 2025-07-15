# Clang Enhancement: `__fq_func__` and `__mangled_func__` Support

## 1) `Expr.h` → `llvm-project/clang/include/clang/AST/Expr.h`

```cpp
enum class PredefinedIdentKind {
  Func,
  Function,
  LFunction, // Same as Function, but as wide string.
  FuncDName,
  FuncSig,
  LFuncSig, // Same as FuncSig, but as wide string
  PrettyFunction,
  /// The same as PrettyFunction, except that the
  /// 'virtual' keyword is omitted for virtual member functions.
  PrettyFunctionNoVirtual,
  FQFunction,
  MangledFunction
};
```

## `2)Expr.cpp` -->`llvm-project/clang/lib/AST`
```
StringRef PredefinedExpr::getIdentKindName(PredefinedIdentKind IK) {
  switch (IK) {
  case PredefinedIdentKind::Func:
    return "__func__";
  case PredefinedIdentKind::Function:
    return "__FUNCTION__";
  case PredefinedIdentKind::FuncDName:
    return "__FUNCDNAME__";
  case PredefinedIdentKind::FQFunction:
    return "__fq_func__";
  case PredefinedIdentKind::MangledFunction:
    return "__mangled_func__";
  case PredefinedIdentKind::LFunction:
    return "L__FUNCTION__";
  case PredefinedIdentKind::PrettyFunction:
    return "__PRETTY_FUNCTION__";
  case PredefinedIdentKind::FuncSig:
    return "__FUNCSIG__";
  case PredefinedIdentKind::LFuncSig:
    return "L__FUNCSIG__";
  case PredefinedIdentKind::PrettyFunctionNoVirtual:
    break;
  }
  llvm_unreachable("Unknown ident kind for PredefinedExpr");
}
```
```
std::string PredefinedExpr::ComputeName(PredefinedIdentKind IK,
                                        const Decl *CurrentDecl,
                                        bool ForceElaboratedPrinting) {
  ASTContext &Context = CurrentDecl->getASTContext();

  if (IK == PredefinedIdentKind::FQFunction) {
  if (const auto *ND = dyn_cast<NamedDecl>(CurrentDecl))
    return ND->getQualifiedNameAsString();
  return "<unknown>";
}

if (IK == PredefinedIdentKind::MangledFunction) {
  if (const auto *ND = dyn_cast<NamedDecl>(CurrentDecl)) {
    std::unique_ptr<MangleContext> MC;
    MC.reset(Context.createMangleContext());
    SmallString<256> Buffer;
    llvm::raw_svector_ostream Out(Buffer);
    GlobalDecl GD;
    if (const CXXConstructorDecl *CD = dyn_cast<CXXConstructorDecl>(ND))
      GD = GlobalDecl(CD, Ctor_Base);
    else if (const CXXDestructorDecl *DD = dyn_cast<CXXDestructorDecl>(ND))
      GD = GlobalDecl(DD, Dtor_Base);
    else if (auto FD = dyn_cast<FunctionDecl>(ND)) {
      GD = FD->isReferenceableKernel() ? GlobalDecl(FD) : GlobalDecl(ND);
    } else
      GD = GlobalDecl(ND);
    MC->mangleName(GD, Out);
    return std::string(Buffer);
  }
  return "<unknown>";
}
// Remaining Code continues
```

## `3)TokenKinds.def` -->`llvm-project/clang/include/clang/Basic/TokenKinds.def`
```
KEYWORD(__fq_func__, KEYALL)
KEYWORD(__mangled_func__, KEYALL)
```

## `4)SemaExpr.cpp` -->`llvm-project/clang/lib/Sema/SemaExpr.cpp`
```
static PredefinedIdentKind getPredefinedExprKind(tok::TokenKind Kind) {
  switch (Kind) {
  default:
    llvm_unreachable("unexpected TokenKind");
  case tok::kw___func__:
    return PredefinedIdentKind::Func;
  case tok::kw___fq_func__:
    return PredefinedIdentKind::FQFunction;
  case tok::kw___mangled_func__:
    return PredefinedIdentKind::MangledFunction;
 // Code Continues
```

## `5)ParseExpr.cpp` -->`llvm-project/clang/lib/Parse/ParseExpr.cpp`
 ```
 case tok::kw_L__FUNCTION__:   
  case tok::kw_L__FUNCSIG__: 
  case tok::kw___PRETTY_FUNCTION__:
    //Add below lines
  case tok::kw___fq_func__:           
  case tok::kw___mangled_func__: 
  //end here
```

  
