/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Parsed attribute helpers                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: Attr.td                                                              *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#if !defined(WANT_DECL_MERGE_LOGIC) && !defined(WANT_STMT_MERGE_LOGIC)
static bool isStruct(const Decl *D) {
  if (const auto *S = dyn_cast<RecordDecl>(D))
    return !S->isUnion();
  return false;
}

static bool isSharedVar(const Decl *D) {
  if (const auto *S = dyn_cast<VarDecl>(D))
    return S->hasGlobalStorage() && !S->getTLSKind();
  return false;
}

static bool isGlobalVar(const Decl *D) {
  if (const auto *S = dyn_cast<VarDecl>(D))
    return S->hasGlobalStorage();
  return false;
}

static bool isHasFunctionProto(const Decl *D) {
  if (const auto *S = dyn_cast<Decl>(D))
    return (S->getFunctionType(true) != nullptr &&
                                       isa<FunctionProtoType>(S->getFunctionType())) ||
                                       isa<ObjCMethodDecl>(S) ||
                                       isa<BlockDecl>(S);
  return false;
}

static bool isFunctionLike(const Decl *D) {
  if (const auto *S = dyn_cast<Decl>(D))
    return S->getFunctionType(false) != nullptr;
  return false;
}

static bool isInlineFunction(const Decl *D) {
  if (const auto *S = dyn_cast<FunctionDecl>(D))
    return S->isInlineSpecified();
  return false;
}

static bool isLocalVar(const Decl *D) {
  if (const auto *S = dyn_cast<VarDecl>(D))
    return S->hasLocalStorage() && !isa<ParmVarDecl>(S);
  return false;
}

static bool isNonTLSGlobalVar(const Decl *D) {
  if (const auto *S = dyn_cast<VarDecl>(D))
    return S->hasGlobalStorage() &&
                               S->getTLSKind() == 0;
  return false;
}

static bool isHLSLEntry(const Decl *D) {
  if (const auto *S = dyn_cast<FunctionDecl>(D))
    return S->isExternallyVisible() && !isa<CXXMethodDecl>(S);
  return false;
}

static bool isHLSLBufferObj(const Decl *D) {
  if (const auto *S = dyn_cast<HLSLBufferDecl>(D))
    return isa<HLSLBufferDecl>(S);
  return false;
}

static bool isExternalGlobalVar(const Decl *D) {
  if (const auto *S = dyn_cast<VarDecl>(D))
    return S->hasGlobalStorage() &&
                               S->getStorageClass()!=StorageClass::SC_Static &&
                               !S->isLocalExternDecl();
  return false;
}

static bool isObjCInstanceMethod(const Decl *D) {
  if (const auto *S = dyn_cast<ObjCMethodDecl>(D))
    return S->isInstanceMethod();
  return false;
}

static bool isImplicitObjectParameter(const Decl *D) {
  if (const auto *S = dyn_cast<FunctionDecl>(D))
    return static_cast<void>(S), false;
  return false;
}

static bool isNonParmVar(const Decl *D) {
  if (const auto *S = dyn_cast<VarDecl>(D))
    return S->getKind() != Decl::ParmVar;
  return false;
}

static bool isClassTmpl(const Decl *D) {
  if (const auto *S = dyn_cast<CXXRecordDecl>(D))
    return S->getDescribedClassTemplate();
  return false;
}

static bool isFunctionTmpl(const Decl *D) {
  if (const auto *S = dyn_cast<FunctionDecl>(D))
    return S->getTemplatedKind() ==
                                 FunctionDecl::TK_FunctionTemplate;
  return false;
}

static bool isVarTmpl(const Decl *D) {
  if (const auto *S = dyn_cast<VarDecl>(D))
    return S->getDescribedVarTemplate();
  return false;
}

static bool isNonBitField(const Decl *D) {
  if (const auto *S = dyn_cast<FieldDecl>(D))
    return !S->isBitField();
  return false;
}

static bool isNonStaticCXXMethod(const Decl *D) {
  if (const auto *S = dyn_cast<CXXMethodDecl>(D))
    return !S->isStatic();
  return false;
}

static bool isBitField(const Decl *D) {
  if (const auto *S = dyn_cast<FieldDecl>(D))
    return S->isBitField();
  return false;
}

static bool isNonStaticNonConstCXXMethod(const Decl *D) {
  if (const auto *S = dyn_cast<CXXMethodDecl>(D))
    return !S->isStatic() && !S->isConst();
  return false;
}

static bool isNonLocalVar(const Decl *D) {
  if (const auto *S = dyn_cast<VarDecl>(D))
    return !S->hasLocalStorage();
  return false;
}

static bool isTLSVar(const Decl *D) {
  if (const auto *S = dyn_cast<VarDecl>(D))
    return S->getTLSKind() != 0;
  return false;
}

static bool isFunctionPointer(const Decl *D) {
  if (const auto *S = dyn_cast<Decl>(D))
    return S->isFunctionPointerType();
  return false;
}

static constexpr ParsedAttrInfo::Spelling AArch64SVEPcsSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "aarch64_sve_pcs"},
  {AttributeCommonInfo::AS_CXX11, "clang::aarch64_sve_pcs"},
  {AttributeCommonInfo::AS_C23, "clang::aarch64_sve_pcs"},
};
struct ParsedAttrInfoAArch64SVEPcs final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoAArch64SVEPcs() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_AArch64SVEPcs,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/AArch64SVEPcsSpellings,
    /*ArgNames=*/{}) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoAArch64SVEPcs Instance;
};
const ParsedAttrInfoAArch64SVEPcs ParsedAttrInfoAArch64SVEPcs::Instance;
static constexpr ParsedAttrInfo::Spelling AArch64VectorPcsSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "aarch64_vector_pcs"},
  {AttributeCommonInfo::AS_CXX11, "clang::aarch64_vector_pcs"},
  {AttributeCommonInfo::AS_C23, "clang::aarch64_vector_pcs"},
};
struct ParsedAttrInfoAArch64VectorPcs final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoAArch64VectorPcs() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_AArch64VectorPcs,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/AArch64VectorPcsSpellings,
    /*ArgNames=*/{}) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoAArch64VectorPcs Instance;
};
const ParsedAttrInfoAArch64VectorPcs ParsedAttrInfoAArch64VectorPcs::Instance;
static constexpr ParsedAttrInfo::Spelling AMDGPUFlatWorkGroupSizeSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "amdgpu_flat_work_group_size"},
  {AttributeCommonInfo::AS_CXX11, "clang::amdgpu_flat_work_group_size"},
};
static constexpr const char *AMDGPUFlatWorkGroupSizeArgNames[] = {
"Min","Max",};
struct ParsedAttrInfoAMDGPUFlatWorkGroupSize final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoAMDGPUFlatWorkGroupSize() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_AMDGPUFlatWorkGroupSize,
    /*NumArgs=*/2,
    /*OptArgs=*/0,
    /*NumArgMembers=*/2,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/AMDGPUFlatWorkGroupSizeSpellings,
    /*ArgNames=*/AMDGPUFlatWorkGroupSizeArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "kernel functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return (N == 0) || (N == 1) || false;
}

static const ParsedAttrInfoAMDGPUFlatWorkGroupSize Instance;
};
const ParsedAttrInfoAMDGPUFlatWorkGroupSize ParsedAttrInfoAMDGPUFlatWorkGroupSize::Instance;
static constexpr ParsedAttrInfo::Spelling AMDGPUKernelCallSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "amdgpu_kernel"},
  {AttributeCommonInfo::AS_CXX11, "clang::amdgpu_kernel"},
  {AttributeCommonInfo::AS_C23, "clang::amdgpu_kernel"},
};
struct ParsedAttrInfoAMDGPUKernelCall final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoAMDGPUKernelCall() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_AMDGPUKernelCall,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/AMDGPUKernelCallSpellings,
    /*ArgNames=*/{}) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoAMDGPUKernelCall Instance;
};
const ParsedAttrInfoAMDGPUKernelCall ParsedAttrInfoAMDGPUKernelCall::Instance;
static constexpr ParsedAttrInfo::Spelling AMDGPUMaxNumWorkGroupsSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "amdgpu_max_num_work_groups"},
  {AttributeCommonInfo::AS_CXX11, "clang::amdgpu_max_num_work_groups"},
};
static constexpr const char *AMDGPUMaxNumWorkGroupsArgNames[] = {
"MaxNumWorkGroupsX","MaxNumWorkGroupsY","MaxNumWorkGroupsZ",};
struct ParsedAttrInfoAMDGPUMaxNumWorkGroups final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoAMDGPUMaxNumWorkGroups() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_AMDGPUMaxNumWorkGroups,
    /*NumArgs=*/1,
    /*OptArgs=*/2,
    /*NumArgMembers=*/3,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/AMDGPUMaxNumWorkGroupsSpellings,
    /*ArgNames=*/AMDGPUMaxNumWorkGroupsArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "kernel functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return (N == 0) || (N == 1) || (N == 2) || false;
}

static const ParsedAttrInfoAMDGPUMaxNumWorkGroups Instance;
};
const ParsedAttrInfoAMDGPUMaxNumWorkGroups ParsedAttrInfoAMDGPUMaxNumWorkGroups::Instance;
static constexpr ParsedAttrInfo::Spelling AMDGPUNumSGPRSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "amdgpu_num_sgpr"},
  {AttributeCommonInfo::AS_CXX11, "clang::amdgpu_num_sgpr"},
};
static constexpr const char *AMDGPUNumSGPRArgNames[] = {
"NumSGPR",};
struct ParsedAttrInfoAMDGPUNumSGPR final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoAMDGPUNumSGPR() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_AMDGPUNumSGPR,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/AMDGPUNumSGPRSpellings,
    /*ArgNames=*/AMDGPUNumSGPRArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "kernel functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoAMDGPUNumSGPR Instance;
};
const ParsedAttrInfoAMDGPUNumSGPR ParsedAttrInfoAMDGPUNumSGPR::Instance;
static constexpr ParsedAttrInfo::Spelling AMDGPUNumVGPRSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "amdgpu_num_vgpr"},
  {AttributeCommonInfo::AS_CXX11, "clang::amdgpu_num_vgpr"},
};
static constexpr const char *AMDGPUNumVGPRArgNames[] = {
"NumVGPR",};
struct ParsedAttrInfoAMDGPUNumVGPR final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoAMDGPUNumVGPR() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_AMDGPUNumVGPR,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/AMDGPUNumVGPRSpellings,
    /*ArgNames=*/AMDGPUNumVGPRArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "kernel functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoAMDGPUNumVGPR Instance;
};
const ParsedAttrInfoAMDGPUNumVGPR ParsedAttrInfoAMDGPUNumVGPR::Instance;
static constexpr ParsedAttrInfo::Spelling AMDGPUWavesPerEUSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "amdgpu_waves_per_eu"},
  {AttributeCommonInfo::AS_CXX11, "clang::amdgpu_waves_per_eu"},
};
static constexpr const char *AMDGPUWavesPerEUArgNames[] = {
"Min","Max",};
struct ParsedAttrInfoAMDGPUWavesPerEU final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoAMDGPUWavesPerEU() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_AMDGPUWavesPerEU,
    /*NumArgs=*/1,
    /*OptArgs=*/1,
    /*NumArgMembers=*/2,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/AMDGPUWavesPerEUSpellings,
    /*ArgNames=*/AMDGPUWavesPerEUArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "kernel functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return (N == 0) || (N == 1) || false;
}

static const ParsedAttrInfoAMDGPUWavesPerEU Instance;
};
const ParsedAttrInfoAMDGPUWavesPerEU ParsedAttrInfoAMDGPUWavesPerEU::Instance;
static constexpr ParsedAttrInfo::Spelling InterruptSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "interrupt"},
  {AttributeCommonInfo::AS_CXX11, "gnu::interrupt"},
  {AttributeCommonInfo::AS_C23, "gnu::interrupt"},
};
static constexpr const char *InterruptArgNames[] = {
"Interrupt",};
struct ParsedAttrInfoInterrupt final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoInterrupt() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Interrupt,
    /*NumArgs=*/0,
    /*OptArgs=*/1,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/1,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/InterruptSpellings,
    /*ArgNames=*/InterruptArgNames) {}
bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && (T.getArch() == llvm::Triple::arm || T.getArch() == llvm::Triple::thumb || T.getArch() == llvm::Triple::armeb || T.getArch() == llvm::Triple::thumbeb || T.getArch() == llvm::Triple::avr || T.getArch() == llvm::Triple::x86 || T.getArch() == llvm::Triple::x86_64 || T.getArch() == llvm::Triple::m68k || T.getArch() == llvm::Triple::msp430 || T.getArch() == llvm::Triple::mips || T.getArch() == llvm::Triple::mipsel || T.getArch() == llvm::Triple::riscv32 || T.getArch() == llvm::Triple::riscv64);
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoInterrupt Instance;
};
const ParsedAttrInfoInterrupt ParsedAttrInfoInterrupt::Instance;
static constexpr ParsedAttrInfo::Spelling ARMInterruptSaveFPSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "interrupt_save_fp"},
};
static constexpr const char *ARMInterruptSaveFPArgNames[] = {
"Interrupt",};
struct ParsedAttrInfoARMInterruptSaveFP final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoARMInterruptSaveFP() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ARMInterruptSaveFP,
    /*NumArgs=*/0,
    /*OptArgs=*/1,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/ARMInterruptSaveFPSpellings,
    /*ArgNames=*/ARMInterruptSaveFPArgNames) {}
bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && (T.getArch() == llvm::Triple::arm || T.getArch() == llvm::Triple::thumb || T.getArch() == llvm::Triple::armeb || T.getArch() == llvm::Triple::thumbeb);
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoARMInterruptSaveFP Instance;
};
const ParsedAttrInfoARMInterruptSaveFP ParsedAttrInfoARMInterruptSaveFP::Instance;
struct ParsedAttrInfoARMSaveFP final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoARMSaveFP() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ARMSaveFP,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/{},
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && (T.getArch() == llvm::Triple::arm || T.getArch() == llvm::Triple::thumb || T.getArch() == llvm::Triple::armeb || T.getArch() == llvm::Triple::thumbeb);
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoARMSaveFP Instance;
};
const ParsedAttrInfoARMSaveFP ParsedAttrInfoARMSaveFP::Instance;
static constexpr ParsedAttrInfo::Spelling AVRSignalSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "signal"},
  {AttributeCommonInfo::AS_CXX11, "gnu::signal"},
  {AttributeCommonInfo::AS_C23, "gnu::signal"},
};
struct ParsedAttrInfoAVRSignal final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoAVRSignal() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_AVRSignal,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/AVRSignalSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && (T.getArch() == llvm::Triple::avr);
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoAVRSignal Instance;
};
const ParsedAttrInfoAVRSignal ParsedAttrInfoAVRSignal::Instance;
static constexpr ParsedAttrInfo::Spelling AbiTagSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "abi_tag"},
  {AttributeCommonInfo::AS_CXX11, "gnu::abi_tag"},
};
static constexpr const char *AbiTagArgNames[] = {
"Tags...",};
struct ParsedAttrInfoAbiTag final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoAbiTag() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_AbiTag,
    /*NumArgs=*/0,
    /*OptArgs=*/15,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/AbiTagSpellings,
    /*ArgNames=*/AbiTagArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isStruct(D) && !isa<VarDecl>(D) && !isa<FunctionDecl>(D) && !isa<NamespaceDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "structs, variables, functions, and namespaces";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record_not_is_union, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_namespace, /*IsSupported=*/LangOpts.CPlusPlus));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoAbiTag Instance;
};
const ParsedAttrInfoAbiTag ParsedAttrInfoAbiTag::Instance;
static constexpr ParsedAttrInfo::Spelling AcquireCapabilitySpellings[] = {
  {AttributeCommonInfo::AS_GNU, "acquire_capability"},
  {AttributeCommonInfo::AS_CXX11, "clang::acquire_capability"},
  {AttributeCommonInfo::AS_GNU, "acquire_shared_capability"},
  {AttributeCommonInfo::AS_CXX11, "clang::acquire_shared_capability"},
  {AttributeCommonInfo::AS_GNU, "exclusive_lock_function"},
  {AttributeCommonInfo::AS_GNU, "shared_lock_function"},
};
static constexpr const char *AcquireCapabilityArgNames[] = {
"Args...",};
struct ParsedAttrInfoAcquireCapability final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoAcquireCapability() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_AcquireCapability,
    /*NumArgs=*/0,
    /*OptArgs=*/15,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/AcquireCapabilitySpellings,
    /*ArgNames=*/AcquireCapabilityArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D) && !isa<ParmVarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions and parameters";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

unsigned spellingIndexToSemanticSpelling(const ParsedAttr &Attr) const override {
  enum Spelling {
    GNU_acquire_capability = 0,
    CXX11_clang_acquire_capability = 1,
    GNU_acquire_shared_capability = 2,
    CXX11_clang_acquire_shared_capability = 3,
    GNU_exclusive_lock_function = 4,
    GNU_shared_lock_function = 5,
  SpellingNotCalculated = 15

  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return GNU_acquire_capability;
    case 1: return CXX11_clang_acquire_capability;
    case 2: return GNU_acquire_shared_capability;
    case 3: return CXX11_clang_acquire_shared_capability;
    case 4: return GNU_exclusive_lock_function;
    case 5: return GNU_shared_lock_function;
  }
}

bool isParamExpr(size_t N) const override {
  return (N == 0) || false;
}

static const ParsedAttrInfoAcquireCapability Instance;
};
const ParsedAttrInfoAcquireCapability ParsedAttrInfoAcquireCapability::Instance;
static constexpr ParsedAttrInfo::Spelling AcquireHandleSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "acquire_handle"},
  {AttributeCommonInfo::AS_CXX11, "clang::acquire_handle"},
  {AttributeCommonInfo::AS_C23, "clang::acquire_handle"},
};
static constexpr const char *AcquireHandleArgNames[] = {
"HandleType",};
struct ParsedAttrInfoAcquireHandle final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoAcquireHandle() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_AcquireHandle,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/AcquireHandleSpellings,
    /*ArgNames=*/AcquireHandleArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D) && !isa<TypedefNameDecl>(D) && !isa<ParmVarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions, typedefs, and parameters";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_type_alias, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable_is_parameter, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoAcquireHandle Instance;
};
const ParsedAttrInfoAcquireHandle ParsedAttrInfoAcquireHandle::Instance;
static constexpr ParsedAttrInfo::Spelling AcquiredAfterSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "acquired_after"},
};
static constexpr const char *AcquiredAfterArgNames[] = {
"Args...",};
struct ParsedAttrInfoAcquiredAfter final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoAcquiredAfter() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_AcquiredAfter,
    /*NumArgs=*/0,
    /*OptArgs=*/15,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/AcquiredAfterSpellings,
    /*ArgNames=*/AcquiredAfterArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FieldDecl>(D) && !isSharedVar(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "non-static data members and global variables";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool isParamExpr(size_t N) const override {
  return (N == 0) || false;
}

static const ParsedAttrInfoAcquiredAfter Instance;
};
const ParsedAttrInfoAcquiredAfter ParsedAttrInfoAcquiredAfter::Instance;
static constexpr ParsedAttrInfo::Spelling AcquiredBeforeSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "acquired_before"},
};
static constexpr const char *AcquiredBeforeArgNames[] = {
"Args...",};
struct ParsedAttrInfoAcquiredBefore final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoAcquiredBefore() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_AcquiredBefore,
    /*NumArgs=*/0,
    /*OptArgs=*/15,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/AcquiredBeforeSpellings,
    /*ArgNames=*/AcquiredBeforeArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FieldDecl>(D) && !isSharedVar(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "non-static data members and global variables";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool isParamExpr(size_t N) const override {
  return (N == 0) || false;
}

static const ParsedAttrInfoAcquiredBefore Instance;
};
const ParsedAttrInfoAcquiredBefore ParsedAttrInfoAcquiredBefore::Instance;
static constexpr ParsedAttrInfo::Spelling AddressSpaceSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "address_space"},
  {AttributeCommonInfo::AS_CXX11, "clang::address_space"},
  {AttributeCommonInfo::AS_C23, "clang::address_space"},
};
static constexpr const char *AddressSpaceArgNames[] = {
"AddressSpace",};
struct ParsedAttrInfoAddressSpace final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoAddressSpace() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_AddressSpace,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/AddressSpaceSpellings,
    /*ArgNames=*/AddressSpaceArgNames) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoAddressSpace Instance;
};
const ParsedAttrInfoAddressSpace ParsedAttrInfoAddressSpace::Instance;
static constexpr ParsedAttrInfo::Spelling AliasSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "alias"},
  {AttributeCommonInfo::AS_CXX11, "gnu::alias"},
  {AttributeCommonInfo::AS_C23, "gnu::alias"},
};
static constexpr const char *AliasArgNames[] = {
"Aliasee",};
struct ParsedAttrInfoAlias final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoAlias() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Alias,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/AliasSpellings,
    /*ArgNames=*/AliasArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D) && !isGlobalVar(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions and global variables";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable_is_global, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoAlias Instance;
};
const ParsedAttrInfoAlias ParsedAttrInfoAlias::Instance;
static constexpr ParsedAttrInfo::Spelling AlignValueSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "align_value"},
};
static constexpr const char *AlignValueArgNames[] = {
"Alignment",};
struct ParsedAttrInfoAlignValue final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoAlignValue() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_AlignValue,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/AlignValueSpellings,
    /*ArgNames=*/AlignValueArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<VarDecl>(D) && !isa<TypedefNameDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "variables and typedefs";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_type_alias, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return (N == 0) || false;
}

static const ParsedAttrInfoAlignValue Instance;
};
const ParsedAttrInfoAlignValue ParsedAttrInfoAlignValue::Instance;
static constexpr ParsedAttrInfo::Spelling AlignedSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "aligned"},
  {AttributeCommonInfo::AS_CXX11, "gnu::aligned"},
  {AttributeCommonInfo::AS_C23, "gnu::aligned"},
  {AttributeCommonInfo::AS_Declspec, "align"},
  {AttributeCommonInfo::AS_Keyword, "alignas"},
  {AttributeCommonInfo::AS_Keyword, "_Alignas"},
};
static constexpr const char *AlignedArgNames[] = {
"Alignment",};
struct ParsedAttrInfoAligned final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoAligned() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Aligned,
    /*NumArgs=*/0,
    /*OptArgs=*/1,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/AlignedSpellings,
    /*ArgNames=*/AlignedArgNames) {}
unsigned spellingIndexToSemanticSpelling(const ParsedAttr &Attr) const override {
  enum Spelling {
    GNU_aligned = 0,
    CXX11_gnu_aligned = 1,
    C23_gnu_aligned = 2,
    Declspec_align = 3,
    Keyword_alignas = 4,
    Keyword_Alignas = 5,
  SpellingNotCalculated = 15

  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return GNU_aligned;
    case 1: return CXX11_gnu_aligned;
    case 2: return C23_gnu_aligned;
    case 3: return Declspec_align;
    case 4: return Keyword_alignas;
    case 5: return Keyword_Alignas;
  }
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoAligned Instance;
};
const ParsedAttrInfoAligned ParsedAttrInfoAligned::Instance;
static constexpr ParsedAttrInfo::Spelling AllocAlignSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "alloc_align"},
  {AttributeCommonInfo::AS_CXX11, "gnu::alloc_align"},
  {AttributeCommonInfo::AS_C23, "gnu::alloc_align"},
};
static constexpr const char *AllocAlignArgNames[] = {
"ParamIndex",};
struct ParsedAttrInfoAllocAlign final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoAllocAlign() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_AllocAlign,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/AllocAlignSpellings,
    /*ArgNames=*/AllocAlignArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isHasFunctionProto(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "non-K&R-style functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoAllocAlign Instance;
};
const ParsedAttrInfoAllocAlign ParsedAttrInfoAllocAlign::Instance;
static constexpr ParsedAttrInfo::Spelling AllocSizeSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "alloc_size"},
  {AttributeCommonInfo::AS_CXX11, "gnu::alloc_size"},
  {AttributeCommonInfo::AS_C23, "gnu::alloc_size"},
};
static constexpr const char *AllocSizeArgNames[] = {
"ElemSizeParam","NumElemsParam",};
struct ParsedAttrInfoAllocSize final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoAllocSize() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_AllocSize,
    /*NumArgs=*/1,
    /*OptArgs=*/1,
    /*NumArgMembers=*/2,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/AllocSizeSpellings,
    /*ArgNames=*/AllocSizeArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isHasFunctionProto(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "non-K&R-style functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoAllocSize Instance;
};
const ParsedAttrInfoAllocSize ParsedAttrInfoAllocSize::Instance;
static constexpr ParsedAttrInfo::Spelling AllocatingSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "allocating"},
  {AttributeCommonInfo::AS_CXX11, "clang::allocating"},
  {AttributeCommonInfo::AS_C23, "clang::allocating"},
};
struct ParsedAttrInfoAllocating final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoAllocating() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Allocating,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/AllocatingSpellings,
    /*ArgNames=*/{}) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoAllocating Instance;
};
const ParsedAttrInfoAllocating ParsedAttrInfoAllocating::Instance;
static constexpr ParsedAttrInfo::Spelling AlwaysDestroySpellings[] = {
  {AttributeCommonInfo::AS_GNU, "always_destroy"},
  {AttributeCommonInfo::AS_CXX11, "clang::always_destroy"},
};
struct ParsedAttrInfoAlwaysDestroy final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoAlwaysDestroy() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_AlwaysDestroy,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/AlwaysDestroySpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<VarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "variables";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

  using ParsedAttrInfo::diagMutualExclusion;

  bool diagMutualExclusion(Sema &S, const ParsedAttr &AL, const Decl *D) const override {
    if (const auto *A = D->getAttr<NoDestroyAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    return true;
  }

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoAlwaysDestroy Instance;
};
const ParsedAttrInfoAlwaysDestroy ParsedAttrInfoAlwaysDestroy::Instance;
static constexpr ParsedAttrInfo::Spelling AlwaysInlineSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "always_inline"},
  {AttributeCommonInfo::AS_CXX11, "gnu::always_inline"},
  {AttributeCommonInfo::AS_C23, "gnu::always_inline"},
  {AttributeCommonInfo::AS_CXX11, "clang::always_inline"},
  {AttributeCommonInfo::AS_C23, "clang::always_inline"},
  {AttributeCommonInfo::AS_Keyword, "__forceinline"},
};
struct ParsedAttrInfoAlwaysInline final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoAlwaysInline() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_AlwaysInline,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/1,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/AlwaysInlineSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions and statements";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &Attr, const Stmt *St) const override {
  if (!isa<Stmt>(St)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions and statements";
    return false;
  }
  return true;
}

  using ParsedAttrInfo::diagMutualExclusion;

  bool diagMutualExclusion(Sema &S, const ParsedAttr &AL, const Decl *D) const override {
    if (const auto *A = D->getAttr<NotTailCalledAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    return true;
  }

unsigned spellingIndexToSemanticSpelling(const ParsedAttr &Attr) const override {
  enum Spelling {
    GNU_always_inline = 0,
    CXX11_gnu_always_inline = 1,
    C23_gnu_always_inline = 2,
    CXX11_clang_always_inline = 3,
    C23_clang_always_inline = 4,
    Keyword_forceinline = 5,
  SpellingNotCalculated = 15

  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return GNU_always_inline;
    case 1: return CXX11_gnu_always_inline;
    case 2: return C23_gnu_always_inline;
    case 3: return CXX11_clang_always_inline;
    case 4: return C23_clang_always_inline;
    case 5: return Keyword_forceinline;
  }
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoAlwaysInline Instance;
};
const ParsedAttrInfoAlwaysInline ParsedAttrInfoAlwaysInline::Instance;
static constexpr ParsedAttrInfo::Spelling AnalyzerNoReturnSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "analyzer_noreturn"},
};
struct ParsedAttrInfoAnalyzerNoReturn final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoAnalyzerNoReturn() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_AnalyzerNoReturn,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/AnalyzerNoReturnSpellings,
    /*ArgNames=*/{}) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoAnalyzerNoReturn Instance;
};
const ParsedAttrInfoAnalyzerNoReturn ParsedAttrInfoAnalyzerNoReturn::Instance;
static constexpr ParsedAttrInfo::Spelling AnnotateSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "annotate"},
  {AttributeCommonInfo::AS_CXX11, "clang::annotate"},
  {AttributeCommonInfo::AS_C23, "clang::annotate"},
};
static constexpr const char *AnnotateArgNames[] = {
"Annotation","Args...",};
struct ParsedAttrInfoAnnotate final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoAnnotate() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Annotate,
    /*NumArgs=*/1,
    /*OptArgs=*/15,
    /*NumArgMembers=*/2,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/1,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/AnnotateSpellings,
    /*ArgNames=*/AnnotateArgNames) {}
bool isParamExpr(size_t N) const override {
  return (N == 1) || false;
}

static const ParsedAttrInfoAnnotate Instance;
};
const ParsedAttrInfoAnnotate ParsedAttrInfoAnnotate::Instance;
static constexpr ParsedAttrInfo::Spelling AnnotateTypeSpellings[] = {
  {AttributeCommonInfo::AS_CXX11, "clang::annotate_type"},
  {AttributeCommonInfo::AS_C23, "clang::annotate_type"},
};
static constexpr const char *AnnotateTypeArgNames[] = {
"Annotation","Args...",};
struct ParsedAttrInfoAnnotateType final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoAnnotateType() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_AnnotateType,
    /*NumArgs=*/1,
    /*OptArgs=*/15,
    /*NumArgMembers=*/2,
    /*HasCustomParsing=*/1,
    /*AcceptsExprPack=*/1,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/AnnotateTypeSpellings,
    /*ArgNames=*/AnnotateTypeArgNames) {}
bool isParamExpr(size_t N) const override {
  return (N == 1) || false;
}

static const ParsedAttrInfoAnnotateType Instance;
};
const ParsedAttrInfoAnnotateType ParsedAttrInfoAnnotateType::Instance;
static constexpr ParsedAttrInfo::Spelling AnyX86NoCallerSavedRegistersSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "no_caller_saved_registers"},
  {AttributeCommonInfo::AS_CXX11, "gnu::no_caller_saved_registers"},
  {AttributeCommonInfo::AS_C23, "gnu::no_caller_saved_registers"},
};
struct ParsedAttrInfoAnyX86NoCallerSavedRegisters final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoAnyX86NoCallerSavedRegisters() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_AnyX86NoCallerSavedRegisters,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/AnyX86NoCallerSavedRegistersSpellings,
    /*ArgNames=*/{}) {}
bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && (T.getArch() == llvm::Triple::x86 || T.getArch() == llvm::Triple::x86_64);
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) AnyX86NoCallerSavedRegistersAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoAnyX86NoCallerSavedRegisters Instance;
};
const ParsedAttrInfoAnyX86NoCallerSavedRegisters ParsedAttrInfoAnyX86NoCallerSavedRegisters::Instance;
static constexpr ParsedAttrInfo::Spelling AnyX86NoCfCheckSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "nocf_check"},
  {AttributeCommonInfo::AS_CXX11, "gnu::nocf_check"},
  {AttributeCommonInfo::AS_C23, "gnu::nocf_check"},
};
struct ParsedAttrInfoAnyX86NoCfCheck final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoAnyX86NoCfCheck() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_AnyX86NoCfCheck,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/AnyX86NoCfCheckSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isFunctionLike(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions and function pointers";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && (T.getArch() == llvm::Triple::x86 || T.getArch() == llvm::Triple::x86_64);
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_hasType_functionType, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoAnyX86NoCfCheck Instance;
};
const ParsedAttrInfoAnyX86NoCfCheck ParsedAttrInfoAnyX86NoCfCheck::Instance;
static constexpr ParsedAttrInfo::Spelling ArcWeakrefUnavailableSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "objc_arc_weak_reference_unavailable"},
  {AttributeCommonInfo::AS_CXX11, "clang::objc_arc_weak_reference_unavailable"},
  {AttributeCommonInfo::AS_C23, "clang::objc_arc_weak_reference_unavailable"},
};
struct ParsedAttrInfoArcWeakrefUnavailable final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoArcWeakrefUnavailable() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ArcWeakrefUnavailable,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/ArcWeakrefUnavailableSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ObjCInterfaceDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "Objective-C interfaces";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_interface, /*IsSupported=*/LangOpts.ObjC));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) ArcWeakrefUnavailableAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoArcWeakrefUnavailable Instance;
};
const ParsedAttrInfoArcWeakrefUnavailable ParsedAttrInfoArcWeakrefUnavailable::Instance;
static constexpr ParsedAttrInfo::Spelling ArgumentWithTypeTagSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "argument_with_type_tag"},
  {AttributeCommonInfo::AS_CXX11, "clang::argument_with_type_tag"},
  {AttributeCommonInfo::AS_C23, "clang::argument_with_type_tag"},
  {AttributeCommonInfo::AS_GNU, "pointer_with_type_tag"},
  {AttributeCommonInfo::AS_CXX11, "clang::pointer_with_type_tag"},
  {AttributeCommonInfo::AS_C23, "clang::pointer_with_type_tag"},
};
static constexpr const char *ArgumentWithTypeTagArgNames[] = {
"ArgumentKind","ArgumentIdx","TypeTagIdx",};
struct ParsedAttrInfoArgumentWithTypeTag final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoArgumentWithTypeTag() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ArgumentWithTypeTag,
    /*NumArgs=*/3,
    /*OptArgs=*/0,
    /*NumArgMembers=*/3,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/ArgumentWithTypeTagSpellings,
    /*ArgNames=*/ArgumentWithTypeTagArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isHasFunctionProto(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "non-K&R-style functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

unsigned spellingIndexToSemanticSpelling(const ParsedAttr &Attr) const override {
  enum Spelling {
    GNU_argument_with_type_tag = 0,
    CXX11_clang_argument_with_type_tag = 1,
    C23_clang_argument_with_type_tag = 2,
    GNU_pointer_with_type_tag = 3,
    CXX11_clang_pointer_with_type_tag = 4,
    C23_clang_pointer_with_type_tag = 5,
  SpellingNotCalculated = 15

  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return GNU_argument_with_type_tag;
    case 1: return CXX11_clang_argument_with_type_tag;
    case 2: return C23_clang_argument_with_type_tag;
    case 3: return GNU_pointer_with_type_tag;
    case 4: return CXX11_clang_pointer_with_type_tag;
    case 5: return C23_clang_pointer_with_type_tag;
  }
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoArgumentWithTypeTag Instance;
};
const ParsedAttrInfoArgumentWithTypeTag ParsedAttrInfoArgumentWithTypeTag::Instance;
static constexpr ParsedAttrInfo::Spelling ArmAgnosticSpellings[] = {
  {AttributeCommonInfo::AS_Keyword, "__arm_agnostic"},
};
static constexpr const char *ArmAgnosticArgNames[] = {
"AgnosticArgs...",};
struct ParsedAttrInfoArmAgnostic final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoArmAgnostic() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ArmAgnostic,
    /*NumArgs=*/0,
    /*OptArgs=*/15,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/ArmAgnosticSpellings,
    /*ArgNames=*/ArmAgnosticArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isHasFunctionProto(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "non-K&R-style functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && (T.getArch() == llvm::Triple::aarch64 || T.getArch() == llvm::Triple::aarch64_be || T.getArch() == llvm::Triple::aarch64_32);
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoArmAgnostic Instance;
};
const ParsedAttrInfoArmAgnostic ParsedAttrInfoArmAgnostic::Instance;
static constexpr ParsedAttrInfo::Spelling ArmBuiltinAliasSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "__clang_arm_builtin_alias"},
  {AttributeCommonInfo::AS_CXX11, "clang::__clang_arm_builtin_alias"},
  {AttributeCommonInfo::AS_C23, "clang::__clang_arm_builtin_alias"},
};
static constexpr const char *ArmBuiltinAliasArgNames[] = {
"BuiltinName",};
struct ParsedAttrInfoArmBuiltinAlias final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoArmBuiltinAlias() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ArmBuiltinAlias,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/ArmBuiltinAliasSpellings,
    /*ArgNames=*/ArmBuiltinAliasArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && (T.getArch() == llvm::Triple::arm || T.getArch() == llvm::Triple::thumb || T.getArch() == llvm::Triple::armeb || T.getArch() == llvm::Triple::thumbeb || T.getArch() == llvm::Triple::aarch64 || T.getArch() == llvm::Triple::aarch64_be || T.getArch() == llvm::Triple::aarch64_32);
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoArmBuiltinAlias Instance;
};
const ParsedAttrInfoArmBuiltinAlias ParsedAttrInfoArmBuiltinAlias::Instance;
static constexpr ParsedAttrInfo::Spelling ArmInSpellings[] = {
  {AttributeCommonInfo::AS_Keyword, "__arm_in"},
};
static constexpr const char *ArmInArgNames[] = {
"InArgs...",};
struct ParsedAttrInfoArmIn final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoArmIn() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ArmIn,
    /*NumArgs=*/0,
    /*OptArgs=*/15,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/ArmInSpellings,
    /*ArgNames=*/ArmInArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isHasFunctionProto(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "non-K&R-style functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && (T.getArch() == llvm::Triple::aarch64 || T.getArch() == llvm::Triple::aarch64_be || T.getArch() == llvm::Triple::aarch64_32);
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoArmIn Instance;
};
const ParsedAttrInfoArmIn ParsedAttrInfoArmIn::Instance;
static constexpr ParsedAttrInfo::Spelling ArmInOutSpellings[] = {
  {AttributeCommonInfo::AS_Keyword, "__arm_inout"},
};
static constexpr const char *ArmInOutArgNames[] = {
"InOutArgs...",};
struct ParsedAttrInfoArmInOut final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoArmInOut() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ArmInOut,
    /*NumArgs=*/0,
    /*OptArgs=*/15,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/ArmInOutSpellings,
    /*ArgNames=*/ArmInOutArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isHasFunctionProto(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "non-K&R-style functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && (T.getArch() == llvm::Triple::aarch64 || T.getArch() == llvm::Triple::aarch64_be || T.getArch() == llvm::Triple::aarch64_32);
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoArmInOut Instance;
};
const ParsedAttrInfoArmInOut ParsedAttrInfoArmInOut::Instance;
static constexpr ParsedAttrInfo::Spelling ArmLocallyStreamingSpellings[] = {
  {AttributeCommonInfo::AS_Keyword, "__arm_locally_streaming"},
};
struct ParsedAttrInfoArmLocallyStreaming final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoArmLocallyStreaming() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ArmLocallyStreaming,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/ArmLocallyStreamingSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && (T.getArch() == llvm::Triple::aarch64 || T.getArch() == llvm::Triple::aarch64_be || T.getArch() == llvm::Triple::aarch64_32);
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoArmLocallyStreaming Instance;
};
const ParsedAttrInfoArmLocallyStreaming ParsedAttrInfoArmLocallyStreaming::Instance;
static constexpr ParsedAttrInfo::Spelling ArmMveStrictPolymorphismSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "__clang_arm_mve_strict_polymorphism"},
  {AttributeCommonInfo::AS_CXX11, "clang::__clang_arm_mve_strict_polymorphism"},
  {AttributeCommonInfo::AS_C23, "clang::__clang_arm_mve_strict_polymorphism"},
};
struct ParsedAttrInfoArmMveStrictPolymorphism final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoArmMveStrictPolymorphism() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ArmMveStrictPolymorphism,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/ArmMveStrictPolymorphismSpellings,
    /*ArgNames=*/{}) {}
bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && (T.getArch() == llvm::Triple::arm || T.getArch() == llvm::Triple::thumb || T.getArch() == llvm::Triple::armeb || T.getArch() == llvm::Triple::thumbeb);
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoArmMveStrictPolymorphism Instance;
};
const ParsedAttrInfoArmMveStrictPolymorphism ParsedAttrInfoArmMveStrictPolymorphism::Instance;
static constexpr ParsedAttrInfo::Spelling ArmNewSpellings[] = {
  {AttributeCommonInfo::AS_Keyword, "__arm_new"},
};
static constexpr const char *ArmNewArgNames[] = {
"NewArgs...",};
struct ParsedAttrInfoArmNew final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoArmNew() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ArmNew,
    /*NumArgs=*/0,
    /*OptArgs=*/15,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/ArmNewSpellings,
    /*ArgNames=*/ArmNewArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && (T.getArch() == llvm::Triple::aarch64 || T.getArch() == llvm::Triple::aarch64_be || T.getArch() == llvm::Triple::aarch64_32);
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoArmNew Instance;
};
const ParsedAttrInfoArmNew ParsedAttrInfoArmNew::Instance;
static constexpr ParsedAttrInfo::Spelling ArmOutSpellings[] = {
  {AttributeCommonInfo::AS_Keyword, "__arm_out"},
};
static constexpr const char *ArmOutArgNames[] = {
"OutArgs...",};
struct ParsedAttrInfoArmOut final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoArmOut() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ArmOut,
    /*NumArgs=*/0,
    /*OptArgs=*/15,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/ArmOutSpellings,
    /*ArgNames=*/ArmOutArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isHasFunctionProto(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "non-K&R-style functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && (T.getArch() == llvm::Triple::aarch64 || T.getArch() == llvm::Triple::aarch64_be || T.getArch() == llvm::Triple::aarch64_32);
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoArmOut Instance;
};
const ParsedAttrInfoArmOut ParsedAttrInfoArmOut::Instance;
static constexpr ParsedAttrInfo::Spelling ArmPreservesSpellings[] = {
  {AttributeCommonInfo::AS_Keyword, "__arm_preserves"},
};
static constexpr const char *ArmPreservesArgNames[] = {
"PreserveArgs...",};
struct ParsedAttrInfoArmPreserves final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoArmPreserves() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ArmPreserves,
    /*NumArgs=*/0,
    /*OptArgs=*/15,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/ArmPreservesSpellings,
    /*ArgNames=*/ArmPreservesArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isHasFunctionProto(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "non-K&R-style functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && (T.getArch() == llvm::Triple::aarch64 || T.getArch() == llvm::Triple::aarch64_be || T.getArch() == llvm::Triple::aarch64_32);
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoArmPreserves Instance;
};
const ParsedAttrInfoArmPreserves ParsedAttrInfoArmPreserves::Instance;
static constexpr ParsedAttrInfo::Spelling ArmStreamingSpellings[] = {
  {AttributeCommonInfo::AS_Keyword, "__arm_streaming"},
};
struct ParsedAttrInfoArmStreaming final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoArmStreaming() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ArmStreaming,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/ArmStreamingSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isHasFunctionProto(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "non-K&R-style functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && (T.getArch() == llvm::Triple::aarch64 || T.getArch() == llvm::Triple::aarch64_be || T.getArch() == llvm::Triple::aarch64_32);
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoArmStreaming Instance;
};
const ParsedAttrInfoArmStreaming ParsedAttrInfoArmStreaming::Instance;
static constexpr ParsedAttrInfo::Spelling ArmStreamingCompatibleSpellings[] = {
  {AttributeCommonInfo::AS_Keyword, "__arm_streaming_compatible"},
};
struct ParsedAttrInfoArmStreamingCompatible final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoArmStreamingCompatible() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ArmStreamingCompatible,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/ArmStreamingCompatibleSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isHasFunctionProto(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "non-K&R-style functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && (T.getArch() == llvm::Triple::aarch64 || T.getArch() == llvm::Triple::aarch64_be || T.getArch() == llvm::Triple::aarch64_32);
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoArmStreamingCompatible Instance;
};
const ParsedAttrInfoArmStreamingCompatible ParsedAttrInfoArmStreamingCompatible::Instance;
static constexpr ParsedAttrInfo::Spelling ArmSveVectorBitsSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "arm_sve_vector_bits"},
};
static constexpr const char *ArmSveVectorBitsArgNames[] = {
"NumBits",};
struct ParsedAttrInfoArmSveVectorBits final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoArmSveVectorBits() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ArmSveVectorBits,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/ArmSveVectorBitsSpellings,
    /*ArgNames=*/ArmSveVectorBitsArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<TypedefNameDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "typedefs";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoArmSveVectorBits Instance;
};
const ParsedAttrInfoArmSveVectorBits ParsedAttrInfoArmSveVectorBits::Instance;
static constexpr ParsedAttrInfo::Spelling ArtificialSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "artificial"},
  {AttributeCommonInfo::AS_CXX11, "gnu::artificial"},
  {AttributeCommonInfo::AS_C23, "gnu::artificial"},
};
struct ParsedAttrInfoArtificial final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoArtificial() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Artificial,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/ArtificialSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isInlineFunction(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "inline functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) ArtificialAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoArtificial Instance;
};
const ParsedAttrInfoArtificial ParsedAttrInfoArtificial::Instance;
static constexpr ParsedAttrInfo::Spelling AssertCapabilitySpellings[] = {
  {AttributeCommonInfo::AS_GNU, "assert_capability"},
  {AttributeCommonInfo::AS_CXX11, "clang::assert_capability"},
  {AttributeCommonInfo::AS_GNU, "assert_shared_capability"},
  {AttributeCommonInfo::AS_CXX11, "clang::assert_shared_capability"},
  {AttributeCommonInfo::AS_GNU, "assert_exclusive_lock"},
  {AttributeCommonInfo::AS_GNU, "assert_shared_lock"},
};
static constexpr const char *AssertCapabilityArgNames[] = {
"Args...",};
struct ParsedAttrInfoAssertCapability final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoAssertCapability() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_AssertCapability,
    /*NumArgs=*/0,
    /*OptArgs=*/15,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/AssertCapabilitySpellings,
    /*ArgNames=*/AssertCapabilityArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

unsigned spellingIndexToSemanticSpelling(const ParsedAttr &Attr) const override {
  enum Spelling {
    GNU_assert_capability = 0,
    CXX11_clang_assert_capability = 1,
    GNU_assert_shared_capability = 2,
    CXX11_clang_assert_shared_capability = 3,
    GNU_assert_exclusive_lock = 4,
    GNU_assert_shared_lock = 5,
  SpellingNotCalculated = 15

  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return GNU_assert_capability;
    case 1: return CXX11_clang_assert_capability;
    case 2: return GNU_assert_shared_capability;
    case 3: return CXX11_clang_assert_shared_capability;
    case 4: return GNU_assert_exclusive_lock;
    case 5: return GNU_assert_shared_lock;
  }
}

bool isParamExpr(size_t N) const override {
  return (N == 0) || false;
}

static const ParsedAttrInfoAssertCapability Instance;
};
const ParsedAttrInfoAssertCapability ParsedAttrInfoAssertCapability::Instance;
static constexpr ParsedAttrInfo::Spelling AssumeAlignedSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "assume_aligned"},
  {AttributeCommonInfo::AS_CXX11, "gnu::assume_aligned"},
  {AttributeCommonInfo::AS_C23, "gnu::assume_aligned"},
};
static constexpr const char *AssumeAlignedArgNames[] = {
"Alignment","Offset",};
struct ParsedAttrInfoAssumeAligned final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoAssumeAligned() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_AssumeAligned,
    /*NumArgs=*/1,
    /*OptArgs=*/1,
    /*NumArgMembers=*/2,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/AssumeAlignedSpellings,
    /*ArgNames=*/AssumeAlignedArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ObjCMethodDecl>(D) && !isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "Objective-C methods and functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return (N == 0) || (N == 1) || false;
}

static const ParsedAttrInfoAssumeAligned Instance;
};
const ParsedAttrInfoAssumeAligned ParsedAttrInfoAssumeAligned::Instance;
static constexpr ParsedAttrInfo::Spelling AtomicSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "atomic"},
  {AttributeCommonInfo::AS_CXX11, "clang::atomic"},
  {AttributeCommonInfo::AS_C23, "clang::atomic"},
};
static constexpr const char *AtomicArgNames[] = {
"AtomicOptions...",};
struct ParsedAttrInfoAtomic final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoAtomic() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Atomic,
    /*NumArgs=*/0,
    /*OptArgs=*/15,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/1,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/AtomicSpellings,
    /*ArgNames=*/AtomicArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &AL, const Decl *D) const override {
  S.Diag(AL.getLoc(), diag::err_attribute_invalid_on_decl)
    << AL << AL.isRegularKeywordAttribute() << D->getLocation();
  return false;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &Attr, const Stmt *St) const override {
  if (!isa<CompoundStmt>(St)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "compound statements";
    return false;
  }
  return true;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoAtomic Instance;
};
const ParsedAttrInfoAtomic ParsedAttrInfoAtomic::Instance;
static constexpr ParsedAttrInfo::Spelling AvailabilitySpellings[] = {
  {AttributeCommonInfo::AS_GNU, "availability"},
  {AttributeCommonInfo::AS_CXX11, "clang::availability"},
  {AttributeCommonInfo::AS_C23, "clang::availability"},
};
static constexpr const char *AvailabilityArgNames[] = {
"platform","introduced","deprecated","obsoleted","unavailable","message","strict","replacement","priority","environment",};
struct ParsedAttrInfoAvailability final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoAvailability() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Availability,
    /*NumArgs=*/10,
    /*OptArgs=*/0,
    /*NumArgMembers=*/10,
    /*HasCustomParsing=*/1,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/AvailabilitySpellings,
    /*ArgNames=*/AvailabilityArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<NamedDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "named declarations";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_enum, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_enum_constant, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_field, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_namespace, /*IsSupported=*/LangOpts.CPlusPlus));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_category, /*IsSupported=*/LangOpts.ObjC));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_implementation, /*IsSupported=*/LangOpts.ObjC));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_interface, /*IsSupported=*/LangOpts.ObjC));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_property, /*IsSupported=*/LangOpts.ObjC));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_protocol, /*IsSupported=*/LangOpts.ObjC));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_type_alias, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoAvailability Instance;
};
const ParsedAttrInfoAvailability ParsedAttrInfoAvailability::Instance;
static constexpr ParsedAttrInfo::Spelling AvailableOnlyInDefaultEvalMethodSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "available_only_in_default_eval_method"},
  {AttributeCommonInfo::AS_CXX11, "clang::available_only_in_default_eval_method"},
  {AttributeCommonInfo::AS_C23, "clang::available_only_in_default_eval_method"},
};
struct ParsedAttrInfoAvailableOnlyInDefaultEvalMethod final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoAvailableOnlyInDefaultEvalMethod() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_AvailableOnlyInDefaultEvalMethod,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/AvailableOnlyInDefaultEvalMethodSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<TypedefNameDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "typedefs";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_type_alias, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoAvailableOnlyInDefaultEvalMethod Instance;
};
const ParsedAttrInfoAvailableOnlyInDefaultEvalMethod ParsedAttrInfoAvailableOnlyInDefaultEvalMethod::Instance;
static constexpr ParsedAttrInfo::Spelling BPFFastCallSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "bpf_fastcall"},
  {AttributeCommonInfo::AS_CXX11, "clang::bpf_fastcall"},
  {AttributeCommonInfo::AS_C23, "clang::bpf_fastcall"},
};
struct ParsedAttrInfoBPFFastCall final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoBPFFastCall() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_BPFFastCall,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/BPFFastCallSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isFunctionLike(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions and function pointers";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return (!LangOpts.CPlusPlus);
}

bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && (T.getArch() == llvm::Triple::bpfel || T.getArch() == llvm::Triple::bpfeb);
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_hasType_functionType, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) BPFFastCallAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoBPFFastCall Instance;
};
const ParsedAttrInfoBPFFastCall ParsedAttrInfoBPFFastCall::Instance;
static constexpr ParsedAttrInfo::Spelling BPFPreserveAccessIndexSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "preserve_access_index"},
  {AttributeCommonInfo::AS_CXX11, "clang::preserve_access_index"},
  {AttributeCommonInfo::AS_C23, "clang::preserve_access_index"},
};
struct ParsedAttrInfoBPFPreserveAccessIndex final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoBPFPreserveAccessIndex() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_BPFPreserveAccessIndex,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/BPFPreserveAccessIndexSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<RecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "structs, unions, and classes";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return (!LangOpts.CPlusPlus);
}

bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && (T.getArch() == llvm::Triple::bpfel || T.getArch() == llvm::Triple::bpfeb);
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoBPFPreserveAccessIndex Instance;
};
const ParsedAttrInfoBPFPreserveAccessIndex ParsedAttrInfoBPFPreserveAccessIndex::Instance;
static constexpr ParsedAttrInfo::Spelling BPFPreserveStaticOffsetSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "preserve_static_offset"},
  {AttributeCommonInfo::AS_CXX11, "clang::preserve_static_offset"},
  {AttributeCommonInfo::AS_C23, "clang::preserve_static_offset"},
};
struct ParsedAttrInfoBPFPreserveStaticOffset final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoBPFPreserveStaticOffset() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_BPFPreserveStaticOffset,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/BPFPreserveStaticOffsetSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<RecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "structs, unions, and classes";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return (!LangOpts.CPlusPlus);
}

bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && (T.getArch() == llvm::Triple::bpfel || T.getArch() == llvm::Triple::bpfeb);
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoBPFPreserveStaticOffset Instance;
};
const ParsedAttrInfoBPFPreserveStaticOffset ParsedAttrInfoBPFPreserveStaticOffset::Instance;
static constexpr ParsedAttrInfo::Spelling BTFDeclTagSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "btf_decl_tag"},
  {AttributeCommonInfo::AS_CXX11, "clang::btf_decl_tag"},
  {AttributeCommonInfo::AS_C23, "clang::btf_decl_tag"},
};
static constexpr const char *BTFDeclTagArgNames[] = {
"BTFDeclTag",};
struct ParsedAttrInfoBTFDeclTag final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoBTFDeclTag() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_BTFDeclTag,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/BTFDeclTagSpellings,
    /*ArgNames=*/BTFDeclTagArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<VarDecl>(D) && !isa<FunctionDecl>(D) && !isa<RecordDecl>(D) && !isa<FieldDecl>(D) && !isa<TypedefNameDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "variables, functions, structs, unions, classes, non-static data members, and typedefs";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return (!LangOpts.CPlusPlus);
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_field, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_type_alias, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoBTFDeclTag Instance;
};
const ParsedAttrInfoBTFDeclTag ParsedAttrInfoBTFDeclTag::Instance;
static constexpr ParsedAttrInfo::Spelling BTFTypeTagSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "btf_type_tag"},
  {AttributeCommonInfo::AS_CXX11, "clang::btf_type_tag"},
  {AttributeCommonInfo::AS_C23, "clang::btf_type_tag"},
};
static constexpr const char *BTFTypeTagArgNames[] = {
"BTFTypeTag",};
struct ParsedAttrInfoBTFTypeTag final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoBTFTypeTag() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_BTFTypeTag,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/BTFTypeTagSpellings,
    /*ArgNames=*/BTFTypeTagArgNames) {}
bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return (!LangOpts.CPlusPlus);
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoBTFTypeTag Instance;
};
const ParsedAttrInfoBTFTypeTag ParsedAttrInfoBTFTypeTag::Instance;
static constexpr ParsedAttrInfo::Spelling BlockingSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "blocking"},
  {AttributeCommonInfo::AS_CXX11, "clang::blocking"},
  {AttributeCommonInfo::AS_C23, "clang::blocking"},
};
struct ParsedAttrInfoBlocking final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoBlocking() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Blocking,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/BlockingSpellings,
    /*ArgNames=*/{}) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoBlocking Instance;
};
const ParsedAttrInfoBlocking ParsedAttrInfoBlocking::Instance;
static constexpr ParsedAttrInfo::Spelling BlocksSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "blocks"},
  {AttributeCommonInfo::AS_CXX11, "clang::blocks"},
  {AttributeCommonInfo::AS_C23, "clang::blocks"},
};
static constexpr const char *BlocksArgNames[] = {
"Type",};
struct ParsedAttrInfoBlocks final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoBlocks() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Blocks,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/BlocksSpellings,
    /*ArgNames=*/BlocksArgNames) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoBlocks Instance;
};
const ParsedAttrInfoBlocks ParsedAttrInfoBlocks::Instance;
static constexpr ParsedAttrInfo::Spelling BuiltinAliasSpellings[] = {
  {AttributeCommonInfo::AS_CXX11, "clang::builtin_alias"},
  {AttributeCommonInfo::AS_C23, "clang::builtin_alias"},
  {AttributeCommonInfo::AS_GNU, "clang_builtin_alias"},
};
static constexpr const char *BuiltinAliasArgNames[] = {
"BuiltinName",};
struct ParsedAttrInfoBuiltinAlias final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoBuiltinAlias() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_BuiltinAlias,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/BuiltinAliasSpellings,
    /*ArgNames=*/BuiltinAliasArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

unsigned spellingIndexToSemanticSpelling(const ParsedAttr &Attr) const override {
  enum Spelling {
    CXX11_clang_builtin_alias = 0,
    C23_clang_builtin_alias = 1,
    GNU_clang_builtin_alias = 2,
  SpellingNotCalculated = 15

  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return CXX11_clang_builtin_alias;
    case 1: return C23_clang_builtin_alias;
    case 2: return GNU_clang_builtin_alias;
  }
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoBuiltinAlias Instance;
};
const ParsedAttrInfoBuiltinAlias ParsedAttrInfoBuiltinAlias::Instance;
static constexpr ParsedAttrInfo::Spelling CDeclSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "cdecl"},
  {AttributeCommonInfo::AS_CXX11, "gnu::cdecl"},
  {AttributeCommonInfo::AS_C23, "gnu::cdecl"},
  {AttributeCommonInfo::AS_Keyword, "__cdecl"},
  {AttributeCommonInfo::AS_Keyword, "_cdecl"},
};
struct ParsedAttrInfoCDecl final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoCDecl() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_CDecl,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/CDeclSpellings,
    /*ArgNames=*/{}) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoCDecl Instance;
};
const ParsedAttrInfoCDecl ParsedAttrInfoCDecl::Instance;
static constexpr ParsedAttrInfo::Spelling CFAuditedTransferSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "cf_audited_transfer"},
  {AttributeCommonInfo::AS_CXX11, "clang::cf_audited_transfer"},
  {AttributeCommonInfo::AS_C23, "clang::cf_audited_transfer"},
};
struct ParsedAttrInfoCFAuditedTransfer final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoCFAuditedTransfer() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_CFAuditedTransfer,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/CFAuditedTransferSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

  using ParsedAttrInfo::diagMutualExclusion;

  bool diagMutualExclusion(Sema &S, const ParsedAttr &AL, const Decl *D) const override {
    if (const auto *A = D->getAttr<CFUnknownTransferAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    return true;
  }

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) CFAuditedTransferAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoCFAuditedTransfer Instance;
};
const ParsedAttrInfoCFAuditedTransfer ParsedAttrInfoCFAuditedTransfer::Instance;
static constexpr ParsedAttrInfo::Spelling CFConsumedSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "cf_consumed"},
  {AttributeCommonInfo::AS_CXX11, "clang::cf_consumed"},
  {AttributeCommonInfo::AS_C23, "clang::cf_consumed"},
};
struct ParsedAttrInfoCFConsumed final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoCFConsumed() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_CFConsumed,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/CFConsumedSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ParmVarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "parameters";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable_is_parameter, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoCFConsumed Instance;
};
const ParsedAttrInfoCFConsumed ParsedAttrInfoCFConsumed::Instance;
static constexpr ParsedAttrInfo::Spelling CFGuardSpellings[] = {
  {AttributeCommonInfo::AS_Declspec, "guard"},
  {AttributeCommonInfo::AS_GNU, "guard"},
  {AttributeCommonInfo::AS_CXX11, "clang::guard"},
  {AttributeCommonInfo::AS_C23, "clang::guard"},
};
static constexpr const char *CFGuardArgNames[] = {
"Guard",};
struct ParsedAttrInfoCFGuard final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoCFGuard() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_CFGuard,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/CFGuardSpellings,
    /*ArgNames=*/CFGuardArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && (T.getOS() == llvm::Triple::Win32);
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoCFGuard Instance;
};
const ParsedAttrInfoCFGuard ParsedAttrInfoCFGuard::Instance;
static constexpr ParsedAttrInfo::Spelling CFICanonicalJumpTableSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "cfi_canonical_jump_table"},
  {AttributeCommonInfo::AS_CXX11, "clang::cfi_canonical_jump_table"},
  {AttributeCommonInfo::AS_C23, "clang::cfi_canonical_jump_table"},
};
struct ParsedAttrInfoCFICanonicalJumpTable final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoCFICanonicalJumpTable() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_CFICanonicalJumpTable,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/CFICanonicalJumpTableSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) CFICanonicalJumpTableAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoCFICanonicalJumpTable Instance;
};
const ParsedAttrInfoCFICanonicalJumpTable ParsedAttrInfoCFICanonicalJumpTable::Instance;
static constexpr ParsedAttrInfo::Spelling CFReturnsNotRetainedSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "cf_returns_not_retained"},
  {AttributeCommonInfo::AS_CXX11, "clang::cf_returns_not_retained"},
  {AttributeCommonInfo::AS_C23, "clang::cf_returns_not_retained"},
};
struct ParsedAttrInfoCFReturnsNotRetained final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoCFReturnsNotRetained() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_CFReturnsNotRetained,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/CFReturnsNotRetainedSpellings,
    /*ArgNames=*/{}) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoCFReturnsNotRetained Instance;
};
const ParsedAttrInfoCFReturnsNotRetained ParsedAttrInfoCFReturnsNotRetained::Instance;
static constexpr ParsedAttrInfo::Spelling CFReturnsRetainedSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "cf_returns_retained"},
  {AttributeCommonInfo::AS_CXX11, "clang::cf_returns_retained"},
  {AttributeCommonInfo::AS_C23, "clang::cf_returns_retained"},
};
struct ParsedAttrInfoCFReturnsRetained final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoCFReturnsRetained() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_CFReturnsRetained,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/CFReturnsRetainedSpellings,
    /*ArgNames=*/{}) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoCFReturnsRetained Instance;
};
const ParsedAttrInfoCFReturnsRetained ParsedAttrInfoCFReturnsRetained::Instance;
static constexpr ParsedAttrInfo::Spelling CFUnknownTransferSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "cf_unknown_transfer"},
  {AttributeCommonInfo::AS_CXX11, "clang::cf_unknown_transfer"},
  {AttributeCommonInfo::AS_C23, "clang::cf_unknown_transfer"},
};
struct ParsedAttrInfoCFUnknownTransfer final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoCFUnknownTransfer() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_CFUnknownTransfer,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/CFUnknownTransferSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

  using ParsedAttrInfo::diagMutualExclusion;

  bool diagMutualExclusion(Sema &S, const ParsedAttr &AL, const Decl *D) const override {
    if (const auto *A = D->getAttr<CFAuditedTransferAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    return true;
  }

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) CFUnknownTransferAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoCFUnknownTransfer Instance;
};
const ParsedAttrInfoCFUnknownTransfer ParsedAttrInfoCFUnknownTransfer::Instance;
static constexpr ParsedAttrInfo::Spelling CPUDispatchSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "cpu_dispatch"},
  {AttributeCommonInfo::AS_CXX11, "clang::cpu_dispatch"},
  {AttributeCommonInfo::AS_C23, "clang::cpu_dispatch"},
  {AttributeCommonInfo::AS_Declspec, "cpu_dispatch"},
};
static constexpr const char *CPUDispatchArgNames[] = {
"Cpus...",};
struct ParsedAttrInfoCPUDispatch final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoCPUDispatch() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_CPUDispatch,
    /*NumArgs=*/0,
    /*OptArgs=*/15,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/CPUDispatchSpellings,
    /*ArgNames=*/CPUDispatchArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

  using ParsedAttrInfo::diagMutualExclusion;

  bool diagMutualExclusion(Sema &S, const ParsedAttr &AL, const Decl *D) const override {
    if (const auto *A = D->getAttr<TargetClonesAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    if (const auto *A = D->getAttr<TargetVersionAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    if (const auto *A = D->getAttr<TargetAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    if (const auto *A = D->getAttr<CPUSpecificAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    return true;
  }

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoCPUDispatch Instance;
};
const ParsedAttrInfoCPUDispatch ParsedAttrInfoCPUDispatch::Instance;
static constexpr ParsedAttrInfo::Spelling CPUSpecificSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "cpu_specific"},
  {AttributeCommonInfo::AS_CXX11, "clang::cpu_specific"},
  {AttributeCommonInfo::AS_C23, "clang::cpu_specific"},
  {AttributeCommonInfo::AS_Declspec, "cpu_specific"},
};
static constexpr const char *CPUSpecificArgNames[] = {
"Cpus...",};
struct ParsedAttrInfoCPUSpecific final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoCPUSpecific() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_CPUSpecific,
    /*NumArgs=*/0,
    /*OptArgs=*/15,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/CPUSpecificSpellings,
    /*ArgNames=*/CPUSpecificArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

  using ParsedAttrInfo::diagMutualExclusion;

  bool diagMutualExclusion(Sema &S, const ParsedAttr &AL, const Decl *D) const override {
    if (const auto *A = D->getAttr<TargetClonesAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    if (const auto *A = D->getAttr<TargetVersionAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    if (const auto *A = D->getAttr<TargetAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    if (const auto *A = D->getAttr<CPUDispatchAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    return true;
  }

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoCPUSpecific Instance;
};
const ParsedAttrInfoCPUSpecific ParsedAttrInfoCPUSpecific::Instance;
static constexpr ParsedAttrInfo::Spelling CUDAConstantSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "constant"},
  {AttributeCommonInfo::AS_Declspec, "__constant__"},
};
struct ParsedAttrInfoCUDAConstant final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoCUDAConstant() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_CUDAConstant,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/CUDAConstantSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<VarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "variables";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

  using ParsedAttrInfo::diagMutualExclusion;

  bool diagMutualExclusion(Sema &S, const ParsedAttr &AL, const Decl *D) const override {
    if (const auto *A = D->getAttr<CUDASharedAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    if (const auto *A = D->getAttr<HIPManagedAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    return true;
  }

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.CUDA;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoCUDAConstant Instance;
};
const ParsedAttrInfoCUDAConstant ParsedAttrInfoCUDAConstant::Instance;
static constexpr ParsedAttrInfo::Spelling CUDADeviceSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "device"},
  {AttributeCommonInfo::AS_Declspec, "__device__"},
};
struct ParsedAttrInfoCUDADevice final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoCUDADevice() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_CUDADevice,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/CUDADeviceSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D) && !isa<VarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions and variables";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

  using ParsedAttrInfo::diagMutualExclusion;

  bool diagMutualExclusion(Sema &S, const ParsedAttr &AL, const Decl *D) const override {
    if (const auto *A = D->getAttr<CUDAGlobalAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    return true;
  }

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.CUDA;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoCUDADevice Instance;
};
const ParsedAttrInfoCUDADevice ParsedAttrInfoCUDADevice::Instance;
static constexpr ParsedAttrInfo::Spelling CUDADeviceBuiltinSurfaceTypeSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "device_builtin_surface_type"},
  {AttributeCommonInfo::AS_Declspec, "__device_builtin_surface_type__"},
};
struct ParsedAttrInfoCUDADeviceBuiltinSurfaceType final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoCUDADeviceBuiltinSurfaceType() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_CUDADeviceBuiltinSurfaceType,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/CUDADeviceBuiltinSurfaceTypeSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<CXXRecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "classes";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

  using ParsedAttrInfo::diagMutualExclusion;

  bool diagMutualExclusion(Sema &S, const ParsedAttr &AL, const Decl *D) const override {
    if (const auto *A = D->getAttr<CUDADeviceBuiltinTextureTypeAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    return true;
  }

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.CUDA;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) CUDADeviceBuiltinSurfaceTypeAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoCUDADeviceBuiltinSurfaceType Instance;
};
const ParsedAttrInfoCUDADeviceBuiltinSurfaceType ParsedAttrInfoCUDADeviceBuiltinSurfaceType::Instance;
static constexpr ParsedAttrInfo::Spelling CUDADeviceBuiltinTextureTypeSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "device_builtin_texture_type"},
  {AttributeCommonInfo::AS_Declspec, "__device_builtin_texture_type__"},
};
struct ParsedAttrInfoCUDADeviceBuiltinTextureType final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoCUDADeviceBuiltinTextureType() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_CUDADeviceBuiltinTextureType,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/CUDADeviceBuiltinTextureTypeSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<CXXRecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "classes";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

  using ParsedAttrInfo::diagMutualExclusion;

  bool diagMutualExclusion(Sema &S, const ParsedAttr &AL, const Decl *D) const override {
    if (const auto *A = D->getAttr<CUDADeviceBuiltinSurfaceTypeAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    return true;
  }

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.CUDA;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) CUDADeviceBuiltinTextureTypeAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoCUDADeviceBuiltinTextureType Instance;
};
const ParsedAttrInfoCUDADeviceBuiltinTextureType ParsedAttrInfoCUDADeviceBuiltinTextureType::Instance;
static constexpr ParsedAttrInfo::Spelling CUDAGlobalSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "global"},
  {AttributeCommonInfo::AS_Declspec, "__global__"},
};
struct ParsedAttrInfoCUDAGlobal final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoCUDAGlobal() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_CUDAGlobal,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/CUDAGlobalSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

  using ParsedAttrInfo::diagMutualExclusion;

  bool diagMutualExclusion(Sema &S, const ParsedAttr &AL, const Decl *D) const override {
    if (const auto *A = D->getAttr<CUDADeviceAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    if (const auto *A = D->getAttr<CUDAHostAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    return true;
  }

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.CUDA;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoCUDAGlobal Instance;
};
const ParsedAttrInfoCUDAGlobal ParsedAttrInfoCUDAGlobal::Instance;
static constexpr ParsedAttrInfo::Spelling CUDAGridConstantSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "grid_constant"},
  {AttributeCommonInfo::AS_Declspec, "__grid_constant__"},
};
struct ParsedAttrInfoCUDAGridConstant final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoCUDAGridConstant() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_CUDAGridConstant,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/CUDAGridConstantSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ParmVarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "parameters";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.CUDA;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable_is_parameter, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoCUDAGridConstant Instance;
};
const ParsedAttrInfoCUDAGridConstant ParsedAttrInfoCUDAGridConstant::Instance;
static constexpr ParsedAttrInfo::Spelling CUDAHostSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "host"},
  {AttributeCommonInfo::AS_Declspec, "__host__"},
};
struct ParsedAttrInfoCUDAHost final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoCUDAHost() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_CUDAHost,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/CUDAHostSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

  using ParsedAttrInfo::diagMutualExclusion;

  bool diagMutualExclusion(Sema &S, const ParsedAttr &AL, const Decl *D) const override {
    if (const auto *A = D->getAttr<CUDAGlobalAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    return true;
  }

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.CUDA;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) CUDAHostAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoCUDAHost Instance;
};
const ParsedAttrInfoCUDAHost ParsedAttrInfoCUDAHost::Instance;
struct ParsedAttrInfoCUDAInvalidTarget final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoCUDAInvalidTarget() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_CUDAInvalidTarget,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/{},
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.CUDA;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoCUDAInvalidTarget Instance;
};
const ParsedAttrInfoCUDAInvalidTarget ParsedAttrInfoCUDAInvalidTarget::Instance;
static constexpr ParsedAttrInfo::Spelling CUDALaunchBoundsSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "launch_bounds"},
  {AttributeCommonInfo::AS_Declspec, "__launch_bounds__"},
};
static constexpr const char *CUDALaunchBoundsArgNames[] = {
"MaxThreads","MinBlocks","MaxBlocks",};
struct ParsedAttrInfoCUDALaunchBounds final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoCUDALaunchBounds() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_CUDALaunchBounds,
    /*NumArgs=*/1,
    /*OptArgs=*/2,
    /*NumArgMembers=*/3,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/CUDALaunchBoundsSpellings,
    /*ArgNames=*/CUDALaunchBoundsArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ObjCMethodDecl>(D) && !isFunctionLike(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "Objective-C methods, functions, and function pointers";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.CUDA;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_hasType_functionType, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return (N == 0) || (N == 1) || (N == 2) || false;
}

static const ParsedAttrInfoCUDALaunchBounds Instance;
};
const ParsedAttrInfoCUDALaunchBounds ParsedAttrInfoCUDALaunchBounds::Instance;
static constexpr ParsedAttrInfo::Spelling CUDASharedSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "shared"},
  {AttributeCommonInfo::AS_Declspec, "__shared__"},
};
struct ParsedAttrInfoCUDAShared final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoCUDAShared() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_CUDAShared,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/CUDASharedSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<VarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "variables";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

  using ParsedAttrInfo::diagMutualExclusion;

  bool diagMutualExclusion(Sema &S, const ParsedAttr &AL, const Decl *D) const override {
    if (const auto *A = D->getAttr<CUDAConstantAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    if (const auto *A = D->getAttr<HIPManagedAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    return true;
  }

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.CUDA;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoCUDAShared Instance;
};
const ParsedAttrInfoCUDAShared ParsedAttrInfoCUDAShared::Instance;
static constexpr ParsedAttrInfo::Spelling CXX11NoReturnSpellings[] = {
  {AttributeCommonInfo::AS_CXX11, "noreturn"},
  {AttributeCommonInfo::AS_C23, "noreturn"},
  {AttributeCommonInfo::AS_C23, "_Noreturn"},
};
struct ParsedAttrInfoCXX11NoReturn final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoCXX11NoReturn() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_CXX11NoReturn,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/CXX11NoReturnSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

unsigned spellingIndexToSemanticSpelling(const ParsedAttr &Attr) const override {
  enum Spelling {
    CXX11_noreturn = 0,
    C23_noreturn = 1,
    C23_Noreturn = 2,
  SpellingNotCalculated = 15

  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return CXX11_noreturn;
    case 1: return C23_noreturn;
    case 2: return C23_Noreturn;
  }
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoCXX11NoReturn Instance;
};
const ParsedAttrInfoCXX11NoReturn ParsedAttrInfoCXX11NoReturn::Instance;
static constexpr ParsedAttrInfo::Spelling CXXAssumeSpellings[] = {
  {AttributeCommonInfo::AS_CXX11, "assume"},
  {AttributeCommonInfo::AS_GNU, "assume"},
  {AttributeCommonInfo::AS_CXX11, "clang::assume"},
  {AttributeCommonInfo::AS_C23, "clang::assume"},
};
static constexpr const char *CXXAssumeArgNames[] = {
"Assumption",};
struct ParsedAttrInfoCXXAssume final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoCXXAssume() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_CXXAssume,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/1,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/1,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/CXXAssumeSpellings,
    /*ArgNames=*/CXXAssumeArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &AL, const Decl *D) const override {
  S.Diag(AL.getLoc(), diag::err_attribute_invalid_on_decl)
    << AL << AL.isRegularKeywordAttribute() << D->getLocation();
  return false;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &Attr, const Stmt *St) const override {
  if (!isa<NullStmt>(St)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "empty statements";
    return false;
  }
  return true;
}

bool isParamExpr(size_t N) const override {
  return (N == 0) || false;
}

static const ParsedAttrInfoCXXAssume Instance;
};
const ParsedAttrInfoCXXAssume ParsedAttrInfoCXXAssume::Instance;
static constexpr ParsedAttrInfo::Spelling CallableWhenSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "callable_when"},
  {AttributeCommonInfo::AS_CXX11, "clang::callable_when"},
};
static constexpr const char *CallableWhenArgNames[] = {
"CallableStates...",};
struct ParsedAttrInfoCallableWhen final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoCallableWhen() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_CallableWhen,
    /*NumArgs=*/0,
    /*OptArgs=*/15,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/CallableWhenSpellings,
    /*ArgNames=*/CallableWhenArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<CXXMethodDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function_is_member, /*IsSupported=*/LangOpts.CPlusPlus));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoCallableWhen Instance;
};
const ParsedAttrInfoCallableWhen ParsedAttrInfoCallableWhen::Instance;
static constexpr ParsedAttrInfo::Spelling CallbackSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "callback"},
  {AttributeCommonInfo::AS_CXX11, "clang::callback"},
  {AttributeCommonInfo::AS_C23, "clang::callback"},
};
static constexpr const char *CallbackArgNames[] = {
"Encoding...",};
struct ParsedAttrInfoCallback final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoCallback() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Callback,
    /*NumArgs=*/0,
    /*OptArgs=*/15,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/CallbackSpellings,
    /*ArgNames=*/CallbackArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoCallback Instance;
};
const ParsedAttrInfoCallback ParsedAttrInfoCallback::Instance;
static constexpr ParsedAttrInfo::Spelling CalledOnceSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "called_once"},
  {AttributeCommonInfo::AS_CXX11, "clang::called_once"},
  {AttributeCommonInfo::AS_C23, "clang::called_once"},
};
struct ParsedAttrInfoCalledOnce final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoCalledOnce() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_CalledOnce,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/CalledOnceSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ParmVarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "parameters";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.ObjC;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable_is_parameter, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoCalledOnce Instance;
};
const ParsedAttrInfoCalledOnce ParsedAttrInfoCalledOnce::Instance;
static constexpr ParsedAttrInfo::Spelling CapabilitySpellings[] = {
  {AttributeCommonInfo::AS_GNU, "capability"},
  {AttributeCommonInfo::AS_CXX11, "clang::capability"},
  {AttributeCommonInfo::AS_GNU, "shared_capability"},
  {AttributeCommonInfo::AS_CXX11, "clang::shared_capability"},
};
static constexpr const char *CapabilityArgNames[] = {
"Name",};
struct ParsedAttrInfoCapability final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoCapability() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Capability,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/CapabilitySpellings,
    /*ArgNames=*/CapabilityArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<RecordDecl>(D) && !isa<TypedefNameDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "structs, unions, classes, and typedefs";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

unsigned spellingIndexToSemanticSpelling(const ParsedAttr &Attr) const override {
  enum Spelling {
    GNU_capability = 0,
    CXX11_clang_capability = 1,
    GNU_shared_capability = 2,
    CXX11_clang_shared_capability = 3,
  SpellingNotCalculated = 15

  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return GNU_capability;
    case 1: return CXX11_clang_capability;
    case 2: return GNU_shared_capability;
    case 3: return CXX11_clang_shared_capability;
  }
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_type_alias, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoCapability Instance;
};
const ParsedAttrInfoCapability ParsedAttrInfoCapability::Instance;
static constexpr ParsedAttrInfo::Spelling CarriesDependencySpellings[] = {
  {AttributeCommonInfo::AS_GNU, "carries_dependency"},
  {AttributeCommonInfo::AS_CXX11, "carries_dependency"},
};
struct ParsedAttrInfoCarriesDependency final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoCarriesDependency() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_CarriesDependency,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/CarriesDependencySpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ParmVarDecl>(D) && !isa<ObjCMethodDecl>(D) && !isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "parameters, Objective-C methods, and functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable_is_parameter, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoCarriesDependency Instance;
};
const ParsedAttrInfoCarriesDependency ParsedAttrInfoCarriesDependency::Instance;
static constexpr ParsedAttrInfo::Spelling CleanupSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "cleanup"},
  {AttributeCommonInfo::AS_CXX11, "gnu::cleanup"},
  {AttributeCommonInfo::AS_C23, "gnu::cleanup"},
};
static constexpr const char *CleanupArgNames[] = {
"FunctionDecl",};
struct ParsedAttrInfoCleanup final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoCleanup() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Cleanup,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/CleanupSpellings,
    /*ArgNames=*/CleanupArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isLocalVar(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "local variables";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable_is_local, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoCleanup Instance;
};
const ParsedAttrInfoCleanup ParsedAttrInfoCleanup::Instance;
static constexpr ParsedAttrInfo::Spelling ClspvLibclcBuiltinSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "clspv_libclc_builtin"},
  {AttributeCommonInfo::AS_CXX11, "clang::clspv_libclc_builtin"},
  {AttributeCommonInfo::AS_C23, "clang::clspv_libclc_builtin"},
};
struct ParsedAttrInfoClspvLibclcBuiltin final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoClspvLibclcBuiltin() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ClspvLibclcBuiltin,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/ClspvLibclcBuiltinSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) ClspvLibclcBuiltinAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoClspvLibclcBuiltin Instance;
};
const ParsedAttrInfoClspvLibclcBuiltin ParsedAttrInfoClspvLibclcBuiltin::Instance;
static constexpr ParsedAttrInfo::Spelling CmseNSCallSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "cmse_nonsecure_call"},
};
struct ParsedAttrInfoCmseNSCall final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoCmseNSCall() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_CmseNSCall,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/CmseNSCallSpellings,
    /*ArgNames=*/{}) {}
bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.Cmse;
}

bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && (T.getArch() == llvm::Triple::arm || T.getArch() == llvm::Triple::thumb || T.getArch() == llvm::Triple::armeb || T.getArch() == llvm::Triple::thumbeb);
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoCmseNSCall Instance;
};
const ParsedAttrInfoCmseNSCall ParsedAttrInfoCmseNSCall::Instance;
static constexpr ParsedAttrInfo::Spelling CmseNSEntrySpellings[] = {
  {AttributeCommonInfo::AS_GNU, "cmse_nonsecure_entry"},
};
struct ParsedAttrInfoCmseNSEntry final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoCmseNSEntry() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_CmseNSEntry,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/CmseNSEntrySpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.Cmse;
}

bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && (T.getArch() == llvm::Triple::arm || T.getArch() == llvm::Triple::thumb || T.getArch() == llvm::Triple::armeb || T.getArch() == llvm::Triple::thumbeb);
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoCmseNSEntry Instance;
};
const ParsedAttrInfoCmseNSEntry ParsedAttrInfoCmseNSEntry::Instance;
static constexpr ParsedAttrInfo::Spelling CodeAlignSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "code_align"},
  {AttributeCommonInfo::AS_CXX11, "clang::code_align"},
  {AttributeCommonInfo::AS_C23, "clang::code_align"},
};
static constexpr const char *CodeAlignArgNames[] = {
"Alignment",};
struct ParsedAttrInfoCodeAlign final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoCodeAlign() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_CodeAlign,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/1,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/CodeAlignSpellings,
    /*ArgNames=*/CodeAlignArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &AL, const Decl *D) const override {
  S.Diag(AL.getLoc(), diag::err_attribute_invalid_on_decl)
    << AL << AL.isRegularKeywordAttribute() << D->getLocation();
  return false;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &Attr, const Stmt *St) const override {
  if (!isa<ForStmt>(St) && !isa<CXXForRangeStmt>(St) && !isa<WhileStmt>(St) && !isa<DoStmt>(St)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "'for', 'while', and 'do' statements";
    return false;
  }
  return true;
}

bool isParamExpr(size_t N) const override {
  return (N == 0) || false;
}

static const ParsedAttrInfoCodeAlign Instance;
};
const ParsedAttrInfoCodeAlign ParsedAttrInfoCodeAlign::Instance;
static constexpr ParsedAttrInfo::Spelling CodeModelSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "model"},
  {AttributeCommonInfo::AS_CXX11, "gnu::model"},
  {AttributeCommonInfo::AS_C23, "gnu::model"},
};
static constexpr const char *CodeModelArgNames[] = {
"Model",};
struct ParsedAttrInfoCodeModel final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoCodeModel() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_CodeModel,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/CodeModelSpellings,
    /*ArgNames=*/CodeModelArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isNonTLSGlobalVar(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "non-TLS global variables";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && (T.getArch() == llvm::Triple::loongarch32 || T.getArch() == llvm::Triple::loongarch64 || T.getArch() == llvm::Triple::x86_64 || T.getArch() == llvm::Triple::nvptx || T.getArch() == llvm::Triple::nvptx64 || T.getArch() == llvm::Triple::amdgcn || T.getArch() == llvm::Triple::r600 || T.getArch() == llvm::Triple::spirv || T.getArch() == llvm::Triple::spirv32 || T.getArch() == llvm::Triple::spirv64);
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoCodeModel Instance;
};
const ParsedAttrInfoCodeModel ParsedAttrInfoCodeModel::Instance;
static constexpr ParsedAttrInfo::Spelling CodeSegSpellings[] = {
  {AttributeCommonInfo::AS_Declspec, "code_seg"},
};
static constexpr const char *CodeSegArgNames[] = {
"Name",};
struct ParsedAttrInfoCodeSeg final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoCodeSeg() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_CodeSeg,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/CodeSegSpellings,
    /*ArgNames=*/CodeSegArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D) && !isa<CXXRecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions and classes";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoCodeSeg Instance;
};
const ParsedAttrInfoCodeSeg ParsedAttrInfoCodeSeg::Instance;
static constexpr ParsedAttrInfo::Spelling ColdSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "cold"},
  {AttributeCommonInfo::AS_CXX11, "gnu::cold"},
  {AttributeCommonInfo::AS_C23, "gnu::cold"},
};
struct ParsedAttrInfoCold final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoCold() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Cold,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/ColdSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

  using ParsedAttrInfo::diagMutualExclusion;

  bool diagMutualExclusion(Sema &S, const ParsedAttr &AL, const Decl *D) const override {
    if (const auto *A = D->getAttr<HotAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    return true;
  }

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) ColdAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoCold Instance;
};
const ParsedAttrInfoCold ParsedAttrInfoCold::Instance;
static constexpr ParsedAttrInfo::Spelling CommonSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "common"},
  {AttributeCommonInfo::AS_CXX11, "gnu::common"},
  {AttributeCommonInfo::AS_C23, "gnu::common"},
};
struct ParsedAttrInfoCommon final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoCommon() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Common,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/CommonSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<VarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "variables";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

  using ParsedAttrInfo::diagMutualExclusion;

  bool diagMutualExclusion(Sema &S, const ParsedAttr &AL, const Decl *D) const override {
    if (const auto *A = D->getAttr<InternalLinkageAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    return true;
  }

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoCommon Instance;
};
const ParsedAttrInfoCommon ParsedAttrInfoCommon::Instance;
static constexpr ParsedAttrInfo::Spelling ConstSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "const"},
  {AttributeCommonInfo::AS_CXX11, "gnu::const"},
  {AttributeCommonInfo::AS_C23, "gnu::const"},
  {AttributeCommonInfo::AS_GNU, "__const"},
  {AttributeCommonInfo::AS_CXX11, "gnu::__const"},
  {AttributeCommonInfo::AS_C23, "gnu::__const"},
};
struct ParsedAttrInfoConst final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoConst() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Const,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/ConstSpellings,
    /*ArgNames=*/{}) {}
AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) ConstAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoConst Instance;
};
const ParsedAttrInfoConst ParsedAttrInfoConst::Instance;
static constexpr ParsedAttrInfo::Spelling ConstInitSpellings[] = {
  {AttributeCommonInfo::AS_Keyword, "constinit"},
  {AttributeCommonInfo::AS_GNU, "require_constant_initialization"},
  {AttributeCommonInfo::AS_CXX11, "clang::require_constant_initialization"},
};
struct ParsedAttrInfoConstInit final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoConstInit() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ConstInit,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/ConstInitSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isGlobalVar(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "global variables";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.CPlusPlus;
}

unsigned spellingIndexToSemanticSpelling(const ParsedAttr &Attr) const override {
  enum Spelling {
    Keyword_constinit = 0,
    GNU_require_constant_initialization = 1,
    CXX11_clang_require_constant_initialization = 2,
  SpellingNotCalculated = 15

  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return Keyword_constinit;
    case 1: return GNU_require_constant_initialization;
    case 2: return CXX11_clang_require_constant_initialization;
  }
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable_is_global, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) ConstInitAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoConstInit Instance;
};
const ParsedAttrInfoConstInit ParsedAttrInfoConstInit::Instance;
static constexpr ParsedAttrInfo::Spelling ConstructorSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "constructor"},
  {AttributeCommonInfo::AS_CXX11, "gnu::constructor"},
  {AttributeCommonInfo::AS_C23, "gnu::constructor"},
};
static constexpr const char *ConstructorArgNames[] = {
"Priority",};
struct ParsedAttrInfoConstructor final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoConstructor() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Constructor,
    /*NumArgs=*/0,
    /*OptArgs=*/1,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/ConstructorSpellings,
    /*ArgNames=*/ConstructorArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoConstructor Instance;
};
const ParsedAttrInfoConstructor ParsedAttrInfoConstructor::Instance;
static constexpr ParsedAttrInfo::Spelling ConsumableSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "consumable"},
  {AttributeCommonInfo::AS_CXX11, "clang::consumable"},
};
static constexpr const char *ConsumableArgNames[] = {
"DefaultState",};
struct ParsedAttrInfoConsumable final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoConsumable() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Consumable,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/ConsumableSpellings,
    /*ArgNames=*/ConsumableArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<CXXRecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "classes";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoConsumable Instance;
};
const ParsedAttrInfoConsumable ParsedAttrInfoConsumable::Instance;
static constexpr ParsedAttrInfo::Spelling ConsumableAutoCastSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "consumable_auto_cast_state"},
  {AttributeCommonInfo::AS_CXX11, "clang::consumable_auto_cast_state"},
};
struct ParsedAttrInfoConsumableAutoCast final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoConsumableAutoCast() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ConsumableAutoCast,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/ConsumableAutoCastSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<CXXRecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "classes";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) ConsumableAutoCastAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoConsumableAutoCast Instance;
};
const ParsedAttrInfoConsumableAutoCast ParsedAttrInfoConsumableAutoCast::Instance;
static constexpr ParsedAttrInfo::Spelling ConsumableSetOnReadSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "consumable_set_state_on_read"},
  {AttributeCommonInfo::AS_CXX11, "clang::consumable_set_state_on_read"},
};
struct ParsedAttrInfoConsumableSetOnRead final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoConsumableSetOnRead() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ConsumableSetOnRead,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/ConsumableSetOnReadSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<CXXRecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "classes";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) ConsumableSetOnReadAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoConsumableSetOnRead Instance;
};
const ParsedAttrInfoConsumableSetOnRead ParsedAttrInfoConsumableSetOnRead::Instance;
static constexpr ParsedAttrInfo::Spelling ConvergentSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "convergent"},
  {AttributeCommonInfo::AS_CXX11, "clang::convergent"},
  {AttributeCommonInfo::AS_C23, "clang::convergent"},
};
struct ParsedAttrInfoConvergent final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoConvergent() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Convergent,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/ConvergentSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

  using ParsedAttrInfo::diagMutualExclusion;

  bool diagMutualExclusion(Sema &S, const ParsedAttr &AL, const Decl *D) const override {
    if (const auto *A = D->getAttr<NoConvergentAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    return true;
  }

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) ConvergentAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoConvergent Instance;
};
const ParsedAttrInfoConvergent ParsedAttrInfoConvergent::Instance;
static constexpr ParsedAttrInfo::Spelling CoroAwaitElidableSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "coro_await_elidable"},
  {AttributeCommonInfo::AS_CXX11, "clang::coro_await_elidable"},
  {AttributeCommonInfo::AS_C23, "clang::coro_await_elidable"},
};
struct ParsedAttrInfoCoroAwaitElidable final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoCoroAwaitElidable() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_CoroAwaitElidable,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/CoroAwaitElidableSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<CXXRecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "classes";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.CPlusPlus;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) CoroAwaitElidableAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoCoroAwaitElidable Instance;
};
const ParsedAttrInfoCoroAwaitElidable ParsedAttrInfoCoroAwaitElidable::Instance;
static constexpr ParsedAttrInfo::Spelling CoroAwaitElidableArgumentSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "coro_await_elidable_argument"},
  {AttributeCommonInfo::AS_CXX11, "clang::coro_await_elidable_argument"},
  {AttributeCommonInfo::AS_C23, "clang::coro_await_elidable_argument"},
};
struct ParsedAttrInfoCoroAwaitElidableArgument final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoCoroAwaitElidableArgument() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_CoroAwaitElidableArgument,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/CoroAwaitElidableArgumentSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ParmVarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "parameters";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.CPlusPlus;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable_is_parameter, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) CoroAwaitElidableArgumentAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoCoroAwaitElidableArgument Instance;
};
const ParsedAttrInfoCoroAwaitElidableArgument ParsedAttrInfoCoroAwaitElidableArgument::Instance;
static constexpr ParsedAttrInfo::Spelling CoroDisableLifetimeBoundSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "coro_disable_lifetimebound"},
  {AttributeCommonInfo::AS_CXX11, "clang::coro_disable_lifetimebound"},
  {AttributeCommonInfo::AS_C23, "clang::coro_disable_lifetimebound"},
};
struct ParsedAttrInfoCoroDisableLifetimeBound final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoCoroDisableLifetimeBound() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_CoroDisableLifetimeBound,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/CoroDisableLifetimeBoundSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.CPlusPlus;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) CoroDisableLifetimeBoundAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoCoroDisableLifetimeBound Instance;
};
const ParsedAttrInfoCoroDisableLifetimeBound ParsedAttrInfoCoroDisableLifetimeBound::Instance;
static constexpr ParsedAttrInfo::Spelling CoroLifetimeBoundSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "coro_lifetimebound"},
  {AttributeCommonInfo::AS_CXX11, "clang::coro_lifetimebound"},
  {AttributeCommonInfo::AS_C23, "clang::coro_lifetimebound"},
};
struct ParsedAttrInfoCoroLifetimeBound final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoCoroLifetimeBound() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_CoroLifetimeBound,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/CoroLifetimeBoundSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<CXXRecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "classes";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.CPlusPlus;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) CoroLifetimeBoundAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoCoroLifetimeBound Instance;
};
const ParsedAttrInfoCoroLifetimeBound ParsedAttrInfoCoroLifetimeBound::Instance;
static constexpr ParsedAttrInfo::Spelling CoroOnlyDestroyWhenCompleteSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "coro_only_destroy_when_complete"},
  {AttributeCommonInfo::AS_CXX11, "clang::coro_only_destroy_when_complete"},
  {AttributeCommonInfo::AS_C23, "clang::coro_only_destroy_when_complete"},
};
struct ParsedAttrInfoCoroOnlyDestroyWhenComplete final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoCoroOnlyDestroyWhenComplete() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_CoroOnlyDestroyWhenComplete,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/CoroOnlyDestroyWhenCompleteSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<CXXRecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "classes";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.CPlusPlus;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) CoroOnlyDestroyWhenCompleteAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoCoroOnlyDestroyWhenComplete Instance;
};
const ParsedAttrInfoCoroOnlyDestroyWhenComplete ParsedAttrInfoCoroOnlyDestroyWhenComplete::Instance;
static constexpr ParsedAttrInfo::Spelling CoroReturnTypeSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "coro_return_type"},
  {AttributeCommonInfo::AS_CXX11, "clang::coro_return_type"},
  {AttributeCommonInfo::AS_C23, "clang::coro_return_type"},
};
struct ParsedAttrInfoCoroReturnType final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoCoroReturnType() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_CoroReturnType,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/CoroReturnTypeSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<CXXRecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "classes";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.CPlusPlus;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) CoroReturnTypeAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoCoroReturnType Instance;
};
const ParsedAttrInfoCoroReturnType ParsedAttrInfoCoroReturnType::Instance;
static constexpr ParsedAttrInfo::Spelling CoroWrapperSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "coro_wrapper"},
  {AttributeCommonInfo::AS_CXX11, "clang::coro_wrapper"},
  {AttributeCommonInfo::AS_C23, "clang::coro_wrapper"},
};
struct ParsedAttrInfoCoroWrapper final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoCoroWrapper() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_CoroWrapper,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/CoroWrapperSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.CPlusPlus;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) CoroWrapperAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoCoroWrapper Instance;
};
const ParsedAttrInfoCoroWrapper ParsedAttrInfoCoroWrapper::Instance;
static constexpr ParsedAttrInfo::Spelling CountedBySpellings[] = {
  {AttributeCommonInfo::AS_GNU, "counted_by"},
  {AttributeCommonInfo::AS_CXX11, "clang::counted_by"},
  {AttributeCommonInfo::AS_C23, "clang::counted_by"},
};
static constexpr const char *CountedByArgNames[] = {
"Count","NestedLevel",};
struct ParsedAttrInfoCountedBy final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoCountedBy() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_CountedBy,
    /*NumArgs=*/1,
    /*OptArgs=*/1,
    /*NumArgMembers=*/2,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/CountedBySpellings,
    /*ArgNames=*/CountedByArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FieldDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "non-static data members";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return (!LangOpts.CPlusPlus);
}

bool isParamExpr(size_t N) const override {
  return (N == 0) || false;
}

static const ParsedAttrInfoCountedBy Instance;
};
const ParsedAttrInfoCountedBy ParsedAttrInfoCountedBy::Instance;
static constexpr ParsedAttrInfo::Spelling CountedByOrNullSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "counted_by_or_null"},
  {AttributeCommonInfo::AS_CXX11, "clang::counted_by_or_null"},
  {AttributeCommonInfo::AS_C23, "clang::counted_by_or_null"},
};
static constexpr const char *CountedByOrNullArgNames[] = {
"Count","NestedLevel",};
struct ParsedAttrInfoCountedByOrNull final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoCountedByOrNull() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_CountedByOrNull,
    /*NumArgs=*/1,
    /*OptArgs=*/1,
    /*NumArgMembers=*/2,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/CountedByOrNullSpellings,
    /*ArgNames=*/CountedByOrNullArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FieldDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "non-static data members";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return (!LangOpts.CPlusPlus);
}

bool isParamExpr(size_t N) const override {
  return (N == 0) || false;
}

static const ParsedAttrInfoCountedByOrNull Instance;
};
const ParsedAttrInfoCountedByOrNull ParsedAttrInfoCountedByOrNull::Instance;
static constexpr ParsedAttrInfo::Spelling DLLExportSpellings[] = {
  {AttributeCommonInfo::AS_Declspec, "dllexport"},
  {AttributeCommonInfo::AS_GNU, "dllexport"},
  {AttributeCommonInfo::AS_CXX11, "gnu::dllexport"},
  {AttributeCommonInfo::AS_C23, "gnu::dllexport"},
};
struct ParsedAttrInfoDLLExport final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoDLLExport() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_DLLExport,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/DLLExportSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D) && !isa<VarDecl>(D) && !isa<CXXRecordDecl>(D) && !isa<ObjCInterfaceDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions, variables, classes, and Objective-C interfaces";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && ( Target.getTriple().hasDLLImportExport() );
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_interface, /*IsSupported=*/LangOpts.ObjC));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoDLLExport Instance;
};
const ParsedAttrInfoDLLExport ParsedAttrInfoDLLExport::Instance;
struct ParsedAttrInfoDLLExportStaticLocal final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoDLLExportStaticLocal() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_DLLExportStaticLocal,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/{},
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && ( Target.getTriple().hasDLLImportExport() );
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoDLLExportStaticLocal Instance;
};
const ParsedAttrInfoDLLExportStaticLocal ParsedAttrInfoDLLExportStaticLocal::Instance;
static constexpr ParsedAttrInfo::Spelling DLLImportSpellings[] = {
  {AttributeCommonInfo::AS_Declspec, "dllimport"},
  {AttributeCommonInfo::AS_GNU, "dllimport"},
  {AttributeCommonInfo::AS_CXX11, "gnu::dllimport"},
  {AttributeCommonInfo::AS_C23, "gnu::dllimport"},
};
struct ParsedAttrInfoDLLImport final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoDLLImport() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_DLLImport,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/DLLImportSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D) && !isa<VarDecl>(D) && !isa<CXXRecordDecl>(D) && !isa<ObjCInterfaceDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions, variables, classes, and Objective-C interfaces";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && ( Target.getTriple().hasDLLImportExport() );
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_interface, /*IsSupported=*/LangOpts.ObjC));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoDLLImport Instance;
};
const ParsedAttrInfoDLLImport ParsedAttrInfoDLLImport::Instance;
struct ParsedAttrInfoDLLImportStaticLocal final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoDLLImportStaticLocal() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_DLLImportStaticLocal,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/{},
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && ( Target.getTriple().hasDLLImportExport() );
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoDLLImportStaticLocal Instance;
};
const ParsedAttrInfoDLLImportStaticLocal ParsedAttrInfoDLLImportStaticLocal::Instance;
static constexpr ParsedAttrInfo::Spelling DeprecatedSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "deprecated"},
  {AttributeCommonInfo::AS_CXX11, "gnu::deprecated"},
  {AttributeCommonInfo::AS_C23, "gnu::deprecated"},
  {AttributeCommonInfo::AS_Declspec, "deprecated"},
  {AttributeCommonInfo::AS_CXX11, "deprecated"},
  {AttributeCommonInfo::AS_C23, "deprecated"},
};
static constexpr const char *DeprecatedArgNames[] = {
"Message","Replacement",};
struct ParsedAttrInfoDeprecated final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoDeprecated() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Deprecated,
    /*NumArgs=*/0,
    /*OptArgs=*/2,
    /*NumArgMembers=*/2,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/DeprecatedSpellings,
    /*ArgNames=*/DeprecatedArgNames) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoDeprecated Instance;
};
const ParsedAttrInfoDeprecated ParsedAttrInfoDeprecated::Instance;
static constexpr ParsedAttrInfo::Spelling DestructorSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "destructor"},
  {AttributeCommonInfo::AS_CXX11, "gnu::destructor"},
  {AttributeCommonInfo::AS_C23, "gnu::destructor"},
};
static constexpr const char *DestructorArgNames[] = {
"Priority",};
struct ParsedAttrInfoDestructor final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoDestructor() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Destructor,
    /*NumArgs=*/0,
    /*OptArgs=*/1,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/DestructorSpellings,
    /*ArgNames=*/DestructorArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoDestructor Instance;
};
const ParsedAttrInfoDestructor ParsedAttrInfoDestructor::Instance;
static constexpr ParsedAttrInfo::Spelling DiagnoseAsBuiltinSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "diagnose_as_builtin"},
  {AttributeCommonInfo::AS_CXX11, "clang::diagnose_as_builtin"},
  {AttributeCommonInfo::AS_C23, "clang::diagnose_as_builtin"},
};
static constexpr const char *DiagnoseAsBuiltinArgNames[] = {
"Function","ArgIndices...",};
struct ParsedAttrInfoDiagnoseAsBuiltin final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoDiagnoseAsBuiltin() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_DiagnoseAsBuiltin,
    /*NumArgs=*/1,
    /*OptArgs=*/15,
    /*NumArgMembers=*/2,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/DiagnoseAsBuiltinSpellings,
    /*ArgNames=*/DiagnoseAsBuiltinArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoDiagnoseAsBuiltin Instance;
};
const ParsedAttrInfoDiagnoseAsBuiltin ParsedAttrInfoDiagnoseAsBuiltin::Instance;
static constexpr ParsedAttrInfo::Spelling DiagnoseIfSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "diagnose_if"},
};
static constexpr const char *DiagnoseIfArgNames[] = {
"Cond","Message","DefaultSeverity","WarningGroup",};
struct ParsedAttrInfoDiagnoseIf final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoDiagnoseIf() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_DiagnoseIf,
    /*NumArgs=*/3,
    /*OptArgs=*/1,
    /*NumArgMembers=*/4,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/DiagnoseIfSpellings,
    /*ArgNames=*/DiagnoseIfArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D) && !isa<ObjCMethodDecl>(D) && !isa<ObjCPropertyDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions, Objective-C methods, and Objective-C properties";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool isParamExpr(size_t N) const override {
  return (N == 0) || false;
}

static const ParsedAttrInfoDiagnoseIf Instance;
};
const ParsedAttrInfoDiagnoseIf ParsedAttrInfoDiagnoseIf::Instance;
static constexpr ParsedAttrInfo::Spelling DisableSanitizerInstrumentationSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "disable_sanitizer_instrumentation"},
  {AttributeCommonInfo::AS_CXX11, "clang::disable_sanitizer_instrumentation"},
  {AttributeCommonInfo::AS_C23, "clang::disable_sanitizer_instrumentation"},
};
struct ParsedAttrInfoDisableSanitizerInstrumentation final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoDisableSanitizerInstrumentation() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_DisableSanitizerInstrumentation,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/DisableSanitizerInstrumentationSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D) && !isa<ObjCMethodDecl>(D) && !isGlobalVar(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions, Objective-C methods, and global variables";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable_is_global, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) DisableSanitizerInstrumentationAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoDisableSanitizerInstrumentation Instance;
};
const ParsedAttrInfoDisableSanitizerInstrumentation ParsedAttrInfoDisableSanitizerInstrumentation::Instance;
static constexpr ParsedAttrInfo::Spelling DisableTailCallsSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "disable_tail_calls"},
  {AttributeCommonInfo::AS_CXX11, "clang::disable_tail_calls"},
  {AttributeCommonInfo::AS_C23, "clang::disable_tail_calls"},
};
struct ParsedAttrInfoDisableTailCalls final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoDisableTailCalls() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_DisableTailCalls,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/DisableTailCallsSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D) && !isa<ObjCMethodDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions and Objective-C methods";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

  using ParsedAttrInfo::diagMutualExclusion;

  bool diagMutualExclusion(Sema &S, const ParsedAttr &AL, const Decl *D) const override {
    if (const auto *A = D->getAttr<NakedAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    return true;
  }

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) DisableTailCallsAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoDisableTailCalls Instance;
};
const ParsedAttrInfoDisableTailCalls ParsedAttrInfoDisableTailCalls::Instance;
static constexpr ParsedAttrInfo::Spelling EmptyBasesSpellings[] = {
  {AttributeCommonInfo::AS_Declspec, "empty_bases"},
};
struct ParsedAttrInfoEmptyBases final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoEmptyBases() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_EmptyBases,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/EmptyBasesSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<CXXRecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "classes";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && (T.getArch() == llvm::Triple::x86 || T.getArch() == llvm::Triple::x86_64 || T.getArch() == llvm::Triple::arm || T.getArch() == llvm::Triple::thumb || T.getArch() == llvm::Triple::aarch64) && ( Target.getCXXABI().isMicrosoft() );
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) EmptyBasesAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoEmptyBases Instance;
};
const ParsedAttrInfoEmptyBases ParsedAttrInfoEmptyBases::Instance;
static constexpr ParsedAttrInfo::Spelling EnableIfSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "enable_if"},
};
static constexpr const char *EnableIfArgNames[] = {
"Cond","Message",};
struct ParsedAttrInfoEnableIf final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoEnableIf() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_EnableIf,
    /*NumArgs=*/2,
    /*OptArgs=*/0,
    /*NumArgMembers=*/2,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/EnableIfSpellings,
    /*ArgNames=*/EnableIfArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return (N == 0) || false;
}

static const ParsedAttrInfoEnableIf Instance;
};
const ParsedAttrInfoEnableIf ParsedAttrInfoEnableIf::Instance;
static constexpr ParsedAttrInfo::Spelling EnforceTCBSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "enforce_tcb"},
  {AttributeCommonInfo::AS_CXX11, "clang::enforce_tcb"},
  {AttributeCommonInfo::AS_C23, "clang::enforce_tcb"},
};
static constexpr const char *EnforceTCBArgNames[] = {
"TCBName",};
struct ParsedAttrInfoEnforceTCB final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoEnforceTCB() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_EnforceTCB,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/EnforceTCBSpellings,
    /*ArgNames=*/EnforceTCBArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D) && !isa<ObjCMethodDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions and Objective-C methods";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoEnforceTCB Instance;
};
const ParsedAttrInfoEnforceTCB ParsedAttrInfoEnforceTCB::Instance;
static constexpr ParsedAttrInfo::Spelling EnforceTCBLeafSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "enforce_tcb_leaf"},
  {AttributeCommonInfo::AS_CXX11, "clang::enforce_tcb_leaf"},
  {AttributeCommonInfo::AS_C23, "clang::enforce_tcb_leaf"},
};
static constexpr const char *EnforceTCBLeafArgNames[] = {
"TCBName",};
struct ParsedAttrInfoEnforceTCBLeaf final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoEnforceTCBLeaf() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_EnforceTCBLeaf,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/EnforceTCBLeafSpellings,
    /*ArgNames=*/EnforceTCBLeafArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D) && !isa<ObjCMethodDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions and Objective-C methods";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoEnforceTCBLeaf Instance;
};
const ParsedAttrInfoEnforceTCBLeaf ParsedAttrInfoEnforceTCBLeaf::Instance;
static constexpr ParsedAttrInfo::Spelling EnumExtensibilitySpellings[] = {
  {AttributeCommonInfo::AS_GNU, "enum_extensibility"},
  {AttributeCommonInfo::AS_CXX11, "clang::enum_extensibility"},
  {AttributeCommonInfo::AS_C23, "clang::enum_extensibility"},
};
static constexpr const char *EnumExtensibilityArgNames[] = {
"Extensibility",};
struct ParsedAttrInfoEnumExtensibility final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoEnumExtensibility() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_EnumExtensibility,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/EnumExtensibilitySpellings,
    /*ArgNames=*/EnumExtensibilityArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<EnumDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "enums";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_enum, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoEnumExtensibility Instance;
};
const ParsedAttrInfoEnumExtensibility ParsedAttrInfoEnumExtensibility::Instance;
static constexpr ParsedAttrInfo::Spelling ErrorSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "error"},
  {AttributeCommonInfo::AS_CXX11, "gnu::error"},
  {AttributeCommonInfo::AS_C23, "gnu::error"},
  {AttributeCommonInfo::AS_GNU, "warning"},
  {AttributeCommonInfo::AS_CXX11, "gnu::warning"},
  {AttributeCommonInfo::AS_C23, "gnu::warning"},
};
static constexpr const char *ErrorArgNames[] = {
"UserDiagnostic",};
struct ParsedAttrInfoError final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoError() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Error,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/ErrorSpellings,
    /*ArgNames=*/ErrorArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

unsigned spellingIndexToSemanticSpelling(const ParsedAttr &Attr) const override {
  enum Spelling {
    GNU_error = 0,
    CXX11_gnu_error = 1,
    C23_gnu_error = 2,
    GNU_warning = 3,
    CXX11_gnu_warning = 4,
    C23_gnu_warning = 5,
  SpellingNotCalculated = 15

  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return GNU_error;
    case 1: return CXX11_gnu_error;
    case 2: return C23_gnu_error;
    case 3: return GNU_warning;
    case 4: return CXX11_gnu_warning;
    case 5: return C23_gnu_warning;
  }
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoError Instance;
};
const ParsedAttrInfoError ParsedAttrInfoError::Instance;
static constexpr ParsedAttrInfo::Spelling ExcludeFromExplicitInstantiationSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "exclude_from_explicit_instantiation"},
  {AttributeCommonInfo::AS_CXX11, "clang::exclude_from_explicit_instantiation"},
  {AttributeCommonInfo::AS_C23, "clang::exclude_from_explicit_instantiation"},
};
struct ParsedAttrInfoExcludeFromExplicitInstantiation final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoExcludeFromExplicitInstantiation() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ExcludeFromExplicitInstantiation,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/ExcludeFromExplicitInstantiationSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<VarDecl>(D) && !isa<FunctionDecl>(D) && !isa<CXXRecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "variables, functions, and classes";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) ExcludeFromExplicitInstantiationAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoExcludeFromExplicitInstantiation Instance;
};
const ParsedAttrInfoExcludeFromExplicitInstantiation ParsedAttrInfoExcludeFromExplicitInstantiation::Instance;
static constexpr ParsedAttrInfo::Spelling ExplicitInitSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "require_explicit_initialization"},
  {AttributeCommonInfo::AS_CXX11, "clang::require_explicit_initialization"},
  {AttributeCommonInfo::AS_C23, "clang::require_explicit_initialization"},
};
struct ParsedAttrInfoExplicitInit final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoExplicitInit() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ExplicitInit,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/ExplicitInitSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FieldDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "non-static data members";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_field, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) ExplicitInitAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoExplicitInit Instance;
};
const ParsedAttrInfoExplicitInit ParsedAttrInfoExplicitInit::Instance;
static constexpr ParsedAttrInfo::Spelling ExtVectorTypeSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "ext_vector_type"},
  {AttributeCommonInfo::AS_CXX11, "clang::ext_vector_type"},
  {AttributeCommonInfo::AS_C23, "clang::ext_vector_type"},
};
static constexpr const char *ExtVectorTypeArgNames[] = {
"NumElements",};
struct ParsedAttrInfoExtVectorType final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoExtVectorType() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ExtVectorType,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/ExtVectorTypeSpellings,
    /*ArgNames=*/ExtVectorTypeArgNames) {}
bool isParamExpr(size_t N) const override {
  return (N == 0) || false;
}

static const ParsedAttrInfoExtVectorType Instance;
};
const ParsedAttrInfoExtVectorType ParsedAttrInfoExtVectorType::Instance;
static constexpr ParsedAttrInfo::Spelling ExternalSourceSymbolSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "external_source_symbol"},
  {AttributeCommonInfo::AS_CXX11, "clang::external_source_symbol"},
  {AttributeCommonInfo::AS_C23, "clang::external_source_symbol"},
};
static constexpr const char *ExternalSourceSymbolArgNames[] = {
"language","definedIn","generatedDeclaration","USR",};
struct ParsedAttrInfoExternalSourceSymbol final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoExternalSourceSymbol() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ExternalSourceSymbol,
    /*NumArgs=*/0,
    /*OptArgs=*/4,
    /*NumArgMembers=*/4,
    /*HasCustomParsing=*/1,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/ExternalSourceSymbolSpellings,
    /*ArgNames=*/ExternalSourceSymbolArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<NamedDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "named declarations";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_enum, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_enum_constant, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_field, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_namespace, /*IsSupported=*/LangOpts.CPlusPlus));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_category, /*IsSupported=*/LangOpts.ObjC));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_implementation, /*IsSupported=*/LangOpts.ObjC));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_interface, /*IsSupported=*/LangOpts.ObjC));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_property, /*IsSupported=*/LangOpts.ObjC));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_protocol, /*IsSupported=*/LangOpts.ObjC));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_type_alias, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoExternalSourceSymbol Instance;
};
const ParsedAttrInfoExternalSourceSymbol ParsedAttrInfoExternalSourceSymbol::Instance;
static constexpr ParsedAttrInfo::Spelling FallThroughSpellings[] = {
  {AttributeCommonInfo::AS_CXX11, "fallthrough"},
  {AttributeCommonInfo::AS_C23, "fallthrough"},
  {AttributeCommonInfo::AS_CXX11, "clang::fallthrough"},
  {AttributeCommonInfo::AS_GNU, "fallthrough"},
  {AttributeCommonInfo::AS_CXX11, "gnu::fallthrough"},
  {AttributeCommonInfo::AS_C23, "gnu::fallthrough"},
};
struct ParsedAttrInfoFallThrough final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoFallThrough() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_FallThrough,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/1,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/FallThroughSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &AL, const Decl *D) const override {
  S.Diag(AL.getLoc(), diag::err_attribute_invalid_on_decl)
    << AL << AL.isRegularKeywordAttribute() << D->getLocation();
  return false;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &Attr, const Stmt *St) const override {
  if (!isa<NullStmt>(St) && !isa<SwitchCase>(St)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "empty statements";
    return false;
  }
  return true;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoFallThrough Instance;
};
const ParsedAttrInfoFallThrough ParsedAttrInfoFallThrough::Instance;
static constexpr ParsedAttrInfo::Spelling FastCallSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "fastcall"},
  {AttributeCommonInfo::AS_CXX11, "gnu::fastcall"},
  {AttributeCommonInfo::AS_C23, "gnu::fastcall"},
  {AttributeCommonInfo::AS_Keyword, "__fastcall"},
  {AttributeCommonInfo::AS_Keyword, "_fastcall"},
};
struct ParsedAttrInfoFastCall final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoFastCall() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_FastCall,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/FastCallSpellings,
    /*ArgNames=*/{}) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoFastCall Instance;
};
const ParsedAttrInfoFastCall ParsedAttrInfoFastCall::Instance;
static constexpr ParsedAttrInfo::Spelling FlagEnumSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "flag_enum"},
  {AttributeCommonInfo::AS_CXX11, "clang::flag_enum"},
  {AttributeCommonInfo::AS_C23, "clang::flag_enum"},
};
struct ParsedAttrInfoFlagEnum final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoFlagEnum() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_FlagEnum,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/FlagEnumSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<EnumDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "enums";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_enum, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) FlagEnumAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoFlagEnum Instance;
};
const ParsedAttrInfoFlagEnum ParsedAttrInfoFlagEnum::Instance;
static constexpr ParsedAttrInfo::Spelling FlattenSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "flatten"},
  {AttributeCommonInfo::AS_CXX11, "gnu::flatten"},
  {AttributeCommonInfo::AS_C23, "gnu::flatten"},
};
struct ParsedAttrInfoFlatten final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoFlatten() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Flatten,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/FlattenSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) FlattenAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoFlatten Instance;
};
const ParsedAttrInfoFlatten ParsedAttrInfoFlatten::Instance;
static constexpr ParsedAttrInfo::Spelling FormatSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "format"},
  {AttributeCommonInfo::AS_CXX11, "gnu::format"},
  {AttributeCommonInfo::AS_C23, "gnu::format"},
};
static constexpr const char *FormatArgNames[] = {
"Type","FormatIdx","FirstArg",};
struct ParsedAttrInfoFormat final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoFormat() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Format,
    /*NumArgs=*/3,
    /*OptArgs=*/0,
    /*NumArgMembers=*/3,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/FormatSpellings,
    /*ArgNames=*/FormatArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ObjCMethodDecl>(D) && !isa<BlockDecl>(D) && !isHasFunctionProto(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "Objective-C methods, blocks, and non-K&R-style functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoFormat Instance;
};
const ParsedAttrInfoFormat ParsedAttrInfoFormat::Instance;
static constexpr ParsedAttrInfo::Spelling FormatArgSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "format_arg"},
  {AttributeCommonInfo::AS_CXX11, "gnu::format_arg"},
  {AttributeCommonInfo::AS_C23, "gnu::format_arg"},
};
static constexpr const char *FormatArgArgNames[] = {
"FormatIdx",};
struct ParsedAttrInfoFormatArg final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoFormatArg() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_FormatArg,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/FormatArgSpellings,
    /*ArgNames=*/FormatArgArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ObjCMethodDecl>(D) && !isHasFunctionProto(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "Objective-C methods and non-K&R-style functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoFormatArg Instance;
};
const ParsedAttrInfoFormatArg ParsedAttrInfoFormatArg::Instance;
static constexpr ParsedAttrInfo::Spelling FormatMatchesSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "format_matches"},
  {AttributeCommonInfo::AS_CXX11, "gnu::format_matches"},
  {AttributeCommonInfo::AS_C23, "gnu::format_matches"},
};
static constexpr const char *FormatMatchesArgNames[] = {
"Type","FormatIdx","ExpectedFormat",};
struct ParsedAttrInfoFormatMatches final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoFormatMatches() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_FormatMatches,
    /*NumArgs=*/3,
    /*OptArgs=*/0,
    /*NumArgMembers=*/3,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/FormatMatchesSpellings,
    /*ArgNames=*/FormatMatchesArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ObjCMethodDecl>(D) && !isa<BlockDecl>(D) && !isHasFunctionProto(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "Objective-C methods, blocks, and non-K&R-style functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool isParamExpr(size_t N) const override {
  return (N == 2) || false;
}

static const ParsedAttrInfoFormatMatches Instance;
};
const ParsedAttrInfoFormatMatches ParsedAttrInfoFormatMatches::Instance;
static constexpr ParsedAttrInfo::Spelling FunctionReturnThunksSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "function_return"},
  {AttributeCommonInfo::AS_CXX11, "gnu::function_return"},
  {AttributeCommonInfo::AS_C23, "gnu::function_return"},
};
static constexpr const char *FunctionReturnThunksArgNames[] = {
"ThunkType",};
struct ParsedAttrInfoFunctionReturnThunks final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoFunctionReturnThunks() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_FunctionReturnThunks,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/FunctionReturnThunksSpellings,
    /*ArgNames=*/FunctionReturnThunksArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && (T.getArch() == llvm::Triple::x86 || T.getArch() == llvm::Triple::x86_64);
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoFunctionReturnThunks Instance;
};
const ParsedAttrInfoFunctionReturnThunks ParsedAttrInfoFunctionReturnThunks::Instance;
static constexpr ParsedAttrInfo::Spelling GNUInlineSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "gnu_inline"},
  {AttributeCommonInfo::AS_CXX11, "gnu::gnu_inline"},
  {AttributeCommonInfo::AS_C23, "gnu::gnu_inline"},
};
struct ParsedAttrInfoGNUInline final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoGNUInline() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_GNUInline,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/GNUInlineSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoGNUInline Instance;
};
const ParsedAttrInfoGNUInline ParsedAttrInfoGNUInline::Instance;
static constexpr ParsedAttrInfo::Spelling GuardedBySpellings[] = {
  {AttributeCommonInfo::AS_GNU, "guarded_by"},
};
static constexpr const char *GuardedByArgNames[] = {
"Arg",};
struct ParsedAttrInfoGuardedBy final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoGuardedBy() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_GuardedBy,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/GuardedBySpellings,
    /*ArgNames=*/GuardedByArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FieldDecl>(D) && !isSharedVar(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "non-static data members and global variables";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool isParamExpr(size_t N) const override {
  return (N == 0) || false;
}

static const ParsedAttrInfoGuardedBy Instance;
};
const ParsedAttrInfoGuardedBy ParsedAttrInfoGuardedBy::Instance;
static constexpr ParsedAttrInfo::Spelling GuardedVarSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "guarded_var"},
  {AttributeCommonInfo::AS_CXX11, "clang::guarded_var"},
};
struct ParsedAttrInfoGuardedVar final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoGuardedVar() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_GuardedVar,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/GuardedVarSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FieldDecl>(D) && !isSharedVar(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "non-static data members and global variables";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) GuardedVarAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoGuardedVar Instance;
};
const ParsedAttrInfoGuardedVar ParsedAttrInfoGuardedVar::Instance;
static constexpr ParsedAttrInfo::Spelling HIPManagedSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "managed"},
  {AttributeCommonInfo::AS_Declspec, "__managed__"},
};
struct ParsedAttrInfoHIPManaged final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoHIPManaged() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_HIPManaged,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/HIPManagedSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<VarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "variables";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

  using ParsedAttrInfo::diagMutualExclusion;

  bool diagMutualExclusion(Sema &S, const ParsedAttr &AL, const Decl *D) const override {
    if (const auto *A = D->getAttr<CUDAConstantAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    if (const auto *A = D->getAttr<CUDASharedAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    return true;
  }

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.HIP;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoHIPManaged Instance;
};
const ParsedAttrInfoHIPManaged ParsedAttrInfoHIPManaged::Instance;
static constexpr ParsedAttrInfo::Spelling HLSLContainedTypeSpellings[] = {
  {AttributeCommonInfo::AS_CXX11, "hlsl::contained_type"},
};
static constexpr const char *HLSLContainedTypeArgNames[] = {
"Type",};
struct ParsedAttrInfoHLSLContainedType final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoHLSLContainedType() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_HLSLContainedType,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/HLSLContainedTypeSpellings,
    /*ArgNames=*/HLSLContainedTypeArgNames) {}
bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.HLSL;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoHLSLContainedType Instance;
};
const ParsedAttrInfoHLSLContainedType ParsedAttrInfoHLSLContainedType::Instance;
static constexpr ParsedAttrInfo::Spelling HLSLControlFlowHintSpellings[] = {
  {AttributeCommonInfo::AS_Microsoft, "branch"},
  {AttributeCommonInfo::AS_Microsoft, "flatten"},
};
struct ParsedAttrInfoHLSLControlFlowHint final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoHLSLControlFlowHint() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_HLSLControlFlowHint,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/1,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/HLSLControlFlowHintSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &AL, const Decl *D) const override {
  S.Diag(AL.getLoc(), diag::err_attribute_invalid_on_decl)
    << AL << AL.isRegularKeywordAttribute() << D->getLocation();
  return false;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &Attr, const Stmt *St) const override {
  if (!isa<IfStmt>(St) && !isa<SwitchStmt>(St)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "'if' and 'switch' statements";
    return false;
  }
  return true;
}

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.HLSL;
}

unsigned spellingIndexToSemanticSpelling(const ParsedAttr &Attr) const override {
  enum Spelling {
    Microsoft_branch = 0,
    Microsoft_flatten = 1,
  SpellingNotCalculated = 15

  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return Microsoft_branch;
    case 1: return Microsoft_flatten;
  }
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoHLSLControlFlowHint Instance;
};
const ParsedAttrInfoHLSLControlFlowHint ParsedAttrInfoHLSLControlFlowHint::Instance;
static constexpr ParsedAttrInfo::Spelling HLSLGroupSharedAddressSpaceSpellings[] = {
  {AttributeCommonInfo::AS_Keyword, "groupshared"},
};
struct ParsedAttrInfoHLSLGroupSharedAddressSpace final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoHLSLGroupSharedAddressSpace() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_HLSLGroupSharedAddressSpace,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/HLSLGroupSharedAddressSpaceSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<VarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "variables";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoHLSLGroupSharedAddressSpace Instance;
};
const ParsedAttrInfoHLSLGroupSharedAddressSpace ParsedAttrInfoHLSLGroupSharedAddressSpace::Instance;
static constexpr ParsedAttrInfo::Spelling HLSLLoopHintSpellings[] = {
  {AttributeCommonInfo::AS_Microsoft, "unroll"},
  {AttributeCommonInfo::AS_Microsoft, "loop"},
};
static constexpr const char *HLSLLoopHintArgNames[] = {
"directive",};
struct ParsedAttrInfoHLSLLoopHint final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoHLSLLoopHint() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_HLSLLoopHint,
    /*NumArgs=*/0,
    /*OptArgs=*/1,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/1,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/HLSLLoopHintSpellings,
    /*ArgNames=*/HLSLLoopHintArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &AL, const Decl *D) const override {
  S.Diag(AL.getLoc(), diag::err_attribute_invalid_on_decl)
    << AL << AL.isRegularKeywordAttribute() << D->getLocation();
  return false;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &Attr, const Stmt *St) const override {
  if (!isa<ForStmt>(St) && !isa<WhileStmt>(St) && !isa<DoStmt>(St)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "'for', 'while', and 'do' statements";
    return false;
  }
  return true;
}

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.HLSL;
}

unsigned spellingIndexToSemanticSpelling(const ParsedAttr &Attr) const override {
  enum Spelling {
    Microsoft_unroll = 0,
    Microsoft_loop = 1,
  SpellingNotCalculated = 15

  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return Microsoft_unroll;
    case 1: return Microsoft_loop;
  }
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoHLSLLoopHint Instance;
};
const ParsedAttrInfoHLSLLoopHint ParsedAttrInfoHLSLLoopHint::Instance;
static constexpr ParsedAttrInfo::Spelling HLSLNumThreadsSpellings[] = {
  {AttributeCommonInfo::AS_Microsoft, "numthreads"},
};
static constexpr const char *HLSLNumThreadsArgNames[] = {
"X","Y","Z",};
struct ParsedAttrInfoHLSLNumThreads final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoHLSLNumThreads() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_HLSLNumThreads,
    /*NumArgs=*/3,
    /*OptArgs=*/0,
    /*NumArgMembers=*/3,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/HLSLNumThreadsSpellings,
    /*ArgNames=*/HLSLNumThreadsArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isHLSLEntry(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "global functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.HLSL;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoHLSLNumThreads Instance;
};
const ParsedAttrInfoHLSLNumThreads ParsedAttrInfoHLSLNumThreads::Instance;
static constexpr ParsedAttrInfo::Spelling HLSLPackOffsetSpellings[] = {
  {AttributeCommonInfo::AS_HLSLAnnotation, "packoffset"},
};
static constexpr const char *HLSLPackOffsetArgNames[] = {
"Subcomponent","Component",};
struct ParsedAttrInfoHLSLPackOffset final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoHLSLPackOffset() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_HLSLPackOffset,
    /*NumArgs=*/2,
    /*OptArgs=*/0,
    /*NumArgMembers=*/2,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/HLSLPackOffsetSpellings,
    /*ArgNames=*/HLSLPackOffsetArgNames) {}
bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.HLSL;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoHLSLPackOffset Instance;
};
const ParsedAttrInfoHLSLPackOffset ParsedAttrInfoHLSLPackOffset::Instance;
static constexpr ParsedAttrInfo::Spelling HLSLParamModifierSpellings[] = {
  {AttributeCommonInfo::AS_Keyword, "in"},
  {AttributeCommonInfo::AS_Keyword, "inout"},
  {AttributeCommonInfo::AS_Keyword, "out"},
};
struct ParsedAttrInfoHLSLParamModifier final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoHLSLParamModifier() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_HLSLParamModifier,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/HLSLParamModifierSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ParmVarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "parameters";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

unsigned spellingIndexToSemanticSpelling(const ParsedAttr &Attr) const override {
  enum Spelling {
    Keyword_in = 0,
    Keyword_inout = 1,
    Keyword_out = 2,
  SpellingNotCalculated = 15

  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return Keyword_in;
    case 1: return Keyword_inout;
    case 2: return Keyword_out;
  }
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoHLSLParamModifier Instance;
};
const ParsedAttrInfoHLSLParamModifier ParsedAttrInfoHLSLParamModifier::Instance;
static constexpr ParsedAttrInfo::Spelling HLSLROVSpellings[] = {
  {AttributeCommonInfo::AS_CXX11, "hlsl::is_rov"},
};
struct ParsedAttrInfoHLSLROV final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoHLSLROV() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_HLSLROV,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/HLSLROVSpellings,
    /*ArgNames=*/{}) {}
bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.HLSL;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoHLSLROV Instance;
};
const ParsedAttrInfoHLSLROV ParsedAttrInfoHLSLROV::Instance;
static constexpr ParsedAttrInfo::Spelling HLSLRawBufferSpellings[] = {
  {AttributeCommonInfo::AS_CXX11, "hlsl::raw_buffer"},
};
struct ParsedAttrInfoHLSLRawBuffer final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoHLSLRawBuffer() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_HLSLRawBuffer,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/HLSLRawBufferSpellings,
    /*ArgNames=*/{}) {}
bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.HLSL;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoHLSLRawBuffer Instance;
};
const ParsedAttrInfoHLSLRawBuffer ParsedAttrInfoHLSLRawBuffer::Instance;
static constexpr ParsedAttrInfo::Spelling HLSLResourceBindingSpellings[] = {
  {AttributeCommonInfo::AS_HLSLAnnotation, "register"},
};
static constexpr const char *HLSLResourceBindingArgNames[] = {
"Slot","Space",};
struct ParsedAttrInfoHLSLResourceBinding final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoHLSLResourceBinding() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_HLSLResourceBinding,
    /*NumArgs=*/1,
    /*OptArgs=*/1,
    /*NumArgMembers=*/2,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/HLSLResourceBindingSpellings,
    /*ArgNames=*/HLSLResourceBindingArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isHLSLBufferObj(D) && !isExternalGlobalVar(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "cbuffer/tbuffer and external global variables";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.HLSL;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoHLSLResourceBinding Instance;
};
const ParsedAttrInfoHLSLResourceBinding ParsedAttrInfoHLSLResourceBinding::Instance;
static constexpr ParsedAttrInfo::Spelling HLSLResourceClassSpellings[] = {
  {AttributeCommonInfo::AS_CXX11, "hlsl::resource_class"},
};
static constexpr const char *HLSLResourceClassArgNames[] = {
"ResourceClass",};
struct ParsedAttrInfoHLSLResourceClass final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoHLSLResourceClass() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_HLSLResourceClass,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/HLSLResourceClassSpellings,
    /*ArgNames=*/HLSLResourceClassArgNames) {}
bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.HLSL;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoHLSLResourceClass Instance;
};
const ParsedAttrInfoHLSLResourceClass ParsedAttrInfoHLSLResourceClass::Instance;
static constexpr ParsedAttrInfo::Spelling HLSLSV_DispatchThreadIDSpellings[] = {
  {AttributeCommonInfo::AS_HLSLAnnotation, "sv_dispatchthreadid"},
};
struct ParsedAttrInfoHLSLSV_DispatchThreadID final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoHLSLSV_DispatchThreadID() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_HLSLSV_DispatchThreadID,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/HLSLSV_DispatchThreadIDSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ParmVarDecl>(D) && !isa<FieldDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "parameters and non-static data members";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.HLSL;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoHLSLSV_DispatchThreadID Instance;
};
const ParsedAttrInfoHLSLSV_DispatchThreadID ParsedAttrInfoHLSLSV_DispatchThreadID::Instance;
static constexpr ParsedAttrInfo::Spelling HLSLSV_GroupIDSpellings[] = {
  {AttributeCommonInfo::AS_HLSLAnnotation, "sv_groupid"},
};
struct ParsedAttrInfoHLSLSV_GroupID final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoHLSLSV_GroupID() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_HLSLSV_GroupID,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/HLSLSV_GroupIDSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ParmVarDecl>(D) && !isa<FieldDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "parameters and non-static data members";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.HLSL;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoHLSLSV_GroupID Instance;
};
const ParsedAttrInfoHLSLSV_GroupID ParsedAttrInfoHLSLSV_GroupID::Instance;
static constexpr ParsedAttrInfo::Spelling HLSLSV_GroupIndexSpellings[] = {
  {AttributeCommonInfo::AS_HLSLAnnotation, "sv_groupindex"},
};
struct ParsedAttrInfoHLSLSV_GroupIndex final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoHLSLSV_GroupIndex() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_HLSLSV_GroupIndex,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/HLSLSV_GroupIndexSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ParmVarDecl>(D) && !isGlobalVar(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "parameters and global variables";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.HLSL;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoHLSLSV_GroupIndex Instance;
};
const ParsedAttrInfoHLSLSV_GroupIndex ParsedAttrInfoHLSLSV_GroupIndex::Instance;
static constexpr ParsedAttrInfo::Spelling HLSLSV_GroupThreadIDSpellings[] = {
  {AttributeCommonInfo::AS_HLSLAnnotation, "sv_groupthreadid"},
};
struct ParsedAttrInfoHLSLSV_GroupThreadID final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoHLSLSV_GroupThreadID() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_HLSLSV_GroupThreadID,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/HLSLSV_GroupThreadIDSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ParmVarDecl>(D) && !isa<FieldDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "parameters and non-static data members";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.HLSL;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoHLSLSV_GroupThreadID Instance;
};
const ParsedAttrInfoHLSLSV_GroupThreadID ParsedAttrInfoHLSLSV_GroupThreadID::Instance;
static constexpr ParsedAttrInfo::Spelling HLSLShaderSpellings[] = {
  {AttributeCommonInfo::AS_Microsoft, "shader"},
};
static constexpr const char *HLSLShaderArgNames[] = {
"Type",};
struct ParsedAttrInfoHLSLShader final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoHLSLShader() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_HLSLShader,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/HLSLShaderSpellings,
    /*ArgNames=*/HLSLShaderArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isHLSLEntry(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "global functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.HLSL;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoHLSLShader Instance;
};
const ParsedAttrInfoHLSLShader ParsedAttrInfoHLSLShader::Instance;
static constexpr ParsedAttrInfo::Spelling HLSLWaveSizeSpellings[] = {
  {AttributeCommonInfo::AS_Microsoft, "WaveSize"},
};
static constexpr const char *HLSLWaveSizeArgNames[] = {
"Min","Max","Preferred",};
struct ParsedAttrInfoHLSLWaveSize final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoHLSLWaveSize() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_HLSLWaveSize,
    /*NumArgs=*/1,
    /*OptArgs=*/2,
    /*NumArgMembers=*/3,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/HLSLWaveSizeSpellings,
    /*ArgNames=*/HLSLWaveSizeArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isHLSLEntry(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "global functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.HLSL;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoHLSLWaveSize Instance;
};
const ParsedAttrInfoHLSLWaveSize ParsedAttrInfoHLSLWaveSize::Instance;
static constexpr ParsedAttrInfo::Spelling HotSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "hot"},
  {AttributeCommonInfo::AS_CXX11, "gnu::hot"},
  {AttributeCommonInfo::AS_C23, "gnu::hot"},
};
struct ParsedAttrInfoHot final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoHot() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Hot,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/HotSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

  using ParsedAttrInfo::diagMutualExclusion;

  bool diagMutualExclusion(Sema &S, const ParsedAttr &AL, const Decl *D) const override {
    if (const auto *A = D->getAttr<ColdAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    return true;
  }

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) HotAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoHot Instance;
};
const ParsedAttrInfoHot ParsedAttrInfoHot::Instance;
static constexpr ParsedAttrInfo::Spelling HybridPatchableSpellings[] = {
  {AttributeCommonInfo::AS_Declspec, "hybrid_patchable"},
  {AttributeCommonInfo::AS_GNU, "hybrid_patchable"},
  {AttributeCommonInfo::AS_CXX11, "clang::hybrid_patchable"},
  {AttributeCommonInfo::AS_C23, "clang::hybrid_patchable"},
};
struct ParsedAttrInfoHybridPatchable final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoHybridPatchable() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_HybridPatchable,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/HybridPatchableSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && ( Target.getTriple().isWindowsArm64EC() );
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoHybridPatchable Instance;
};
const ParsedAttrInfoHybridPatchable ParsedAttrInfoHybridPatchable::Instance;
static constexpr ParsedAttrInfo::Spelling IBActionSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "ibaction"},
  {AttributeCommonInfo::AS_CXX11, "clang::ibaction"},
  {AttributeCommonInfo::AS_C23, "clang::ibaction"},
};
struct ParsedAttrInfoIBAction final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoIBAction() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_IBAction,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/IBActionSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isObjCInstanceMethod(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "Objective-C instance methods";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method_is_instance, /*IsSupported=*/LangOpts.ObjC));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) IBActionAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoIBAction Instance;
};
const ParsedAttrInfoIBAction ParsedAttrInfoIBAction::Instance;
static constexpr ParsedAttrInfo::Spelling IBOutletSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "iboutlet"},
  {AttributeCommonInfo::AS_CXX11, "clang::iboutlet"},
  {AttributeCommonInfo::AS_C23, "clang::iboutlet"},
};
struct ParsedAttrInfoIBOutlet final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoIBOutlet() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_IBOutlet,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/IBOutletSpellings,
    /*ArgNames=*/{}) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoIBOutlet Instance;
};
const ParsedAttrInfoIBOutlet ParsedAttrInfoIBOutlet::Instance;
static constexpr ParsedAttrInfo::Spelling IBOutletCollectionSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "iboutletcollection"},
  {AttributeCommonInfo::AS_CXX11, "clang::iboutletcollection"},
  {AttributeCommonInfo::AS_C23, "clang::iboutletcollection"},
};
static constexpr const char *IBOutletCollectionArgNames[] = {
"Interface",};
struct ParsedAttrInfoIBOutletCollection final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoIBOutletCollection() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_IBOutletCollection,
    /*NumArgs=*/0,
    /*OptArgs=*/1,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/IBOutletCollectionSpellings,
    /*ArgNames=*/IBOutletCollectionArgNames) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoIBOutletCollection Instance;
};
const ParsedAttrInfoIBOutletCollection ParsedAttrInfoIBOutletCollection::Instance;
static constexpr ParsedAttrInfo::Spelling IFuncSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "ifunc"},
  {AttributeCommonInfo::AS_CXX11, "gnu::ifunc"},
  {AttributeCommonInfo::AS_C23, "gnu::ifunc"},
};
static constexpr const char *IFuncArgNames[] = {
"Resolver",};
struct ParsedAttrInfoIFunc final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoIFunc() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_IFunc,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/IFuncSpellings,
    /*ArgNames=*/IFuncArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && ( Target.supportsIFunc() );
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoIFunc Instance;
};
const ParsedAttrInfoIFunc ParsedAttrInfoIFunc::Instance;
static constexpr ParsedAttrInfo::Spelling InitPrioritySpellings[] = {
  {AttributeCommonInfo::AS_GNU, "init_priority"},
  {AttributeCommonInfo::AS_CXX11, "gnu::init_priority"},
};
static constexpr const char *InitPriorityArgNames[] = {
"Priority",};
struct ParsedAttrInfoInitPriority final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoInitPriority() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_InitPriority,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/InitPrioritySpellings,
    /*ArgNames=*/InitPriorityArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<VarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "variables";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && ( !Target.getTriple().isOSzOS() );
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoInitPriority Instance;
};
const ParsedAttrInfoInitPriority ParsedAttrInfoInitPriority::Instance;
static constexpr ParsedAttrInfo::Spelling IntelOclBiccSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "intel_ocl_bicc"},
  {AttributeCommonInfo::AS_CXX11, "clang::intel_ocl_bicc"},
};
struct ParsedAttrInfoIntelOclBicc final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoIntelOclBicc() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_IntelOclBicc,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/IntelOclBiccSpellings,
    /*ArgNames=*/{}) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoIntelOclBicc Instance;
};
const ParsedAttrInfoIntelOclBicc ParsedAttrInfoIntelOclBicc::Instance;
static constexpr ParsedAttrInfo::Spelling InternalLinkageSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "internal_linkage"},
  {AttributeCommonInfo::AS_CXX11, "clang::internal_linkage"},
  {AttributeCommonInfo::AS_C23, "clang::internal_linkage"},
};
struct ParsedAttrInfoInternalLinkage final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoInternalLinkage() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_InternalLinkage,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/InternalLinkageSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<VarDecl>(D) && !isa<FunctionDecl>(D) && !isa<CXXRecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "variables, functions, and classes";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

  using ParsedAttrInfo::diagMutualExclusion;

  bool diagMutualExclusion(Sema &S, const ParsedAttr &AL, const Decl *D) const override {
    if (const auto *A = D->getAttr<CommonAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    return true;
  }

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoInternalLinkage Instance;
};
const ParsedAttrInfoInternalLinkage ParsedAttrInfoInternalLinkage::Instance;
static constexpr ParsedAttrInfo::Spelling LTOVisibilityPublicSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "lto_visibility_public"},
  {AttributeCommonInfo::AS_CXX11, "clang::lto_visibility_public"},
  {AttributeCommonInfo::AS_C23, "clang::lto_visibility_public"},
};
struct ParsedAttrInfoLTOVisibilityPublic final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoLTOVisibilityPublic() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_LTOVisibilityPublic,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/LTOVisibilityPublicSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<RecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "structs, unions, and classes";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) LTOVisibilityPublicAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoLTOVisibilityPublic Instance;
};
const ParsedAttrInfoLTOVisibilityPublic ParsedAttrInfoLTOVisibilityPublic::Instance;
static constexpr ParsedAttrInfo::Spelling LayoutVersionSpellings[] = {
  {AttributeCommonInfo::AS_Declspec, "layout_version"},
};
static constexpr const char *LayoutVersionArgNames[] = {
"Version",};
struct ParsedAttrInfoLayoutVersion final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoLayoutVersion() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_LayoutVersion,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/LayoutVersionSpellings,
    /*ArgNames=*/LayoutVersionArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<CXXRecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "classes";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && (T.getArch() == llvm::Triple::x86 || T.getArch() == llvm::Triple::x86_64 || T.getArch() == llvm::Triple::arm || T.getArch() == llvm::Triple::thumb || T.getArch() == llvm::Triple::aarch64) && ( Target.getCXXABI().isMicrosoft() );
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoLayoutVersion Instance;
};
const ParsedAttrInfoLayoutVersion ParsedAttrInfoLayoutVersion::Instance;
static constexpr ParsedAttrInfo::Spelling LeafSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "leaf"},
  {AttributeCommonInfo::AS_CXX11, "gnu::leaf"},
  {AttributeCommonInfo::AS_C23, "gnu::leaf"},
};
struct ParsedAttrInfoLeaf final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoLeaf() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Leaf,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/LeafSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) LeafAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoLeaf Instance;
};
const ParsedAttrInfoLeaf ParsedAttrInfoLeaf::Instance;
static constexpr ParsedAttrInfo::Spelling LifetimeBoundSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "lifetimebound"},
  {AttributeCommonInfo::AS_CXX11, "clang::lifetimebound"},
};
struct ParsedAttrInfoLifetimeBound final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoLifetimeBound() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_LifetimeBound,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/LifetimeBoundSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ParmVarDecl>(D) && !isImplicitObjectParameter(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "parameters and implicit object parameters";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) LifetimeBoundAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoLifetimeBound Instance;
};
const ParsedAttrInfoLifetimeBound ParsedAttrInfoLifetimeBound::Instance;
static constexpr ParsedAttrInfo::Spelling LifetimeCaptureBySpellings[] = {
  {AttributeCommonInfo::AS_GNU, "lifetime_capture_by"},
  {AttributeCommonInfo::AS_CXX11, "clang::lifetime_capture_by"},
};
static constexpr const char *LifetimeCaptureByArgNames[] = {
"Params...",};
struct ParsedAttrInfoLifetimeCaptureBy final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoLifetimeCaptureBy() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_LifetimeCaptureBy,
    /*NumArgs=*/0,
    /*OptArgs=*/15,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/LifetimeCaptureBySpellings,
    /*ArgNames=*/LifetimeCaptureByArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ParmVarDecl>(D) && !isImplicitObjectParameter(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "parameters and implicit object parameters";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoLifetimeCaptureBy Instance;
};
const ParsedAttrInfoLifetimeCaptureBy ParsedAttrInfoLifetimeCaptureBy::Instance;
static constexpr ParsedAttrInfo::Spelling LikelySpellings[] = {
  {AttributeCommonInfo::AS_CXX11, "likely"},
  {AttributeCommonInfo::AS_C23, "clang::likely"},
};
struct ParsedAttrInfoLikely final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoLikely() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Likely,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/1,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/LikelySpellings,
    /*ArgNames=*/{}) {}
  using ParsedAttrInfo::diagMutualExclusion;

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoLikely Instance;
};
const ParsedAttrInfoLikely ParsedAttrInfoLikely::Instance;
static constexpr ParsedAttrInfo::Spelling LoaderUninitializedSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "loader_uninitialized"},
  {AttributeCommonInfo::AS_CXX11, "clang::loader_uninitialized"},
  {AttributeCommonInfo::AS_C23, "clang::loader_uninitialized"},
};
struct ParsedAttrInfoLoaderUninitialized final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoLoaderUninitialized() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_LoaderUninitialized,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/LoaderUninitializedSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isGlobalVar(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "global variables";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable_is_global, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) LoaderUninitializedAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoLoaderUninitialized Instance;
};
const ParsedAttrInfoLoaderUninitialized ParsedAttrInfoLoaderUninitialized::Instance;
static constexpr ParsedAttrInfo::Spelling LockReturnedSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "lock_returned"},
};
static constexpr const char *LockReturnedArgNames[] = {
"Arg",};
struct ParsedAttrInfoLockReturned final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoLockReturned() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_LockReturned,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/LockReturnedSpellings,
    /*ArgNames=*/LockReturnedArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool isParamExpr(size_t N) const override {
  return (N == 0) || false;
}

static const ParsedAttrInfoLockReturned Instance;
};
const ParsedAttrInfoLockReturned ParsedAttrInfoLockReturned::Instance;
static constexpr ParsedAttrInfo::Spelling LockableSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "lockable"},
};
struct ParsedAttrInfoLockable final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoLockable() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Lockable,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/LockableSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<RecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "structs, unions, and classes";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoLockable Instance;
};
const ParsedAttrInfoLockable ParsedAttrInfoLockable::Instance;
static constexpr ParsedAttrInfo::Spelling LocksExcludedSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "locks_excluded"},
};
static constexpr const char *LocksExcludedArgNames[] = {
"Args...",};
struct ParsedAttrInfoLocksExcluded final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoLocksExcluded() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_LocksExcluded,
    /*NumArgs=*/0,
    /*OptArgs=*/15,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/LocksExcludedSpellings,
    /*ArgNames=*/LocksExcludedArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D) && !isa<ParmVarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions and parameters";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool isParamExpr(size_t N) const override {
  return (N == 0) || false;
}

static const ParsedAttrInfoLocksExcluded Instance;
};
const ParsedAttrInfoLocksExcluded ParsedAttrInfoLocksExcluded::Instance;
static constexpr ParsedAttrInfo::Spelling LoopHintSpellings[] = {
  {AttributeCommonInfo::AS_Pragma, "clang::loop"},
  {AttributeCommonInfo::AS_Pragma, "unroll"},
  {AttributeCommonInfo::AS_Pragma, "nounroll"},
  {AttributeCommonInfo::AS_Pragma, "unroll_and_jam"},
  {AttributeCommonInfo::AS_Pragma, "nounroll_and_jam"},
};
static constexpr const char *LoopHintArgNames[] = {
"Option","State","Value",};
struct ParsedAttrInfoLoopHint final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoLoopHint() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_LoopHint,
    /*NumArgs=*/3,
    /*OptArgs=*/0,
    /*NumArgMembers=*/3,
    /*HasCustomParsing=*/1,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/LoopHintSpellings,
    /*ArgNames=*/LoopHintArgNames) {}
unsigned spellingIndexToSemanticSpelling(const ParsedAttr &Attr) const override {
  enum Spelling {
    Pragma_clang_loop = 0,
    Pragma_unroll = 1,
    Pragma_nounroll = 2,
    Pragma_unroll_and_jam = 3,
    Pragma_nounroll_and_jam = 4,
  SpellingNotCalculated = 15

  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return Pragma_clang_loop;
    case 1: return Pragma_unroll;
    case 2: return Pragma_nounroll;
    case 3: return Pragma_unroll_and_jam;
    case 4: return Pragma_nounroll_and_jam;
  }
}

bool isParamExpr(size_t N) const override {
  return (N == 2) || false;
}

static const ParsedAttrInfoLoopHint Instance;
};
const ParsedAttrInfoLoopHint ParsedAttrInfoLoopHint::Instance;
static constexpr ParsedAttrInfo::Spelling M68kRTDSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "m68k_rtd"},
  {AttributeCommonInfo::AS_CXX11, "clang::m68k_rtd"},
  {AttributeCommonInfo::AS_C23, "clang::m68k_rtd"},
};
struct ParsedAttrInfoM68kRTD final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoM68kRTD() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_M68kRTD,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/M68kRTDSpellings,
    /*ArgNames=*/{}) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoM68kRTD Instance;
};
const ParsedAttrInfoM68kRTD ParsedAttrInfoM68kRTD::Instance;
static constexpr ParsedAttrInfo::Spelling MIGServerRoutineSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "mig_server_routine"},
  {AttributeCommonInfo::AS_CXX11, "clang::mig_server_routine"},
  {AttributeCommonInfo::AS_C23, "clang::mig_server_routine"},
};
struct ParsedAttrInfoMIGServerRoutine final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoMIGServerRoutine() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_MIGServerRoutine,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/MIGServerRoutineSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D) && !isa<ObjCMethodDecl>(D) && !isa<BlockDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions, Objective-C methods, and blocks";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_block, /*IsSupported=*/LangOpts.Blocks));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoMIGServerRoutine Instance;
};
const ParsedAttrInfoMIGServerRoutine ParsedAttrInfoMIGServerRoutine::Instance;
static constexpr ParsedAttrInfo::Spelling MSABISpellings[] = {
  {AttributeCommonInfo::AS_GNU, "ms_abi"},
  {AttributeCommonInfo::AS_CXX11, "gnu::ms_abi"},
  {AttributeCommonInfo::AS_C23, "gnu::ms_abi"},
};
struct ParsedAttrInfoMSABI final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoMSABI() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_MSABI,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/MSABISpellings,
    /*ArgNames=*/{}) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoMSABI Instance;
};
const ParsedAttrInfoMSABI ParsedAttrInfoMSABI::Instance;
static constexpr ParsedAttrInfo::Spelling MSAllocatorSpellings[] = {
  {AttributeCommonInfo::AS_Declspec, "allocator"},
};
struct ParsedAttrInfoMSAllocator final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoMSAllocator() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_MSAllocator,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/MSAllocatorSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoMSAllocator Instance;
};
const ParsedAttrInfoMSAllocator ParsedAttrInfoMSAllocator::Instance;
static constexpr ParsedAttrInfo::Spelling MSConstexprSpellings[] = {
  {AttributeCommonInfo::AS_CXX11, "msvc::constexpr"},
};
struct ParsedAttrInfoMSConstexpr final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoMSConstexpr() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_MSConstexpr,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/MSConstexprSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions and return statements";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &Attr, const Stmt *St) const override {
  if (!isa<ReturnStmt>(St)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions and return statements";
    return false;
  }
  return true;
}

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.MicrosoftExt;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoMSConstexpr Instance;
};
const ParsedAttrInfoMSConstexpr ParsedAttrInfoMSConstexpr::Instance;
static constexpr ParsedAttrInfo::Spelling MSInheritanceSpellings[] = {
  {AttributeCommonInfo::AS_Keyword, "__single_inheritance"},
  {AttributeCommonInfo::AS_Keyword, "__multiple_inheritance"},
  {AttributeCommonInfo::AS_Keyword, "__virtual_inheritance"},
  {AttributeCommonInfo::AS_Keyword, "__unspecified_inheritance"},
};
struct ParsedAttrInfoMSInheritance final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoMSInheritance() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_MSInheritance,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/MSInheritanceSpellings,
    /*ArgNames=*/{}) {}
bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.MicrosoftExt;
}

unsigned spellingIndexToSemanticSpelling(const ParsedAttr &Attr) const override {
  enum Spelling {
    Keyword_single_inheritance = 0,
    Keyword_multiple_inheritance = 1,
    Keyword_virtual_inheritance = 2,
    Keyword_unspecified_inheritance = 3,
  SpellingNotCalculated = 15

  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return Keyword_single_inheritance;
    case 1: return Keyword_multiple_inheritance;
    case 2: return Keyword_virtual_inheritance;
    case 3: return Keyword_unspecified_inheritance;
  }
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoMSInheritance Instance;
};
const ParsedAttrInfoMSInheritance ParsedAttrInfoMSInheritance::Instance;
static constexpr ParsedAttrInfo::Spelling MSNoVTableSpellings[] = {
  {AttributeCommonInfo::AS_Declspec, "novtable"},
};
struct ParsedAttrInfoMSNoVTable final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoMSNoVTable() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_MSNoVTable,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/MSNoVTableSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<CXXRecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "classes";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && (T.getArch() == llvm::Triple::x86 || T.getArch() == llvm::Triple::x86_64 || T.getArch() == llvm::Triple::arm || T.getArch() == llvm::Triple::thumb || T.getArch() == llvm::Triple::aarch64) && ( Target.getCXXABI().isMicrosoft() );
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) MSNoVTableAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoMSNoVTable Instance;
};
const ParsedAttrInfoMSNoVTable ParsedAttrInfoMSNoVTable::Instance;
static constexpr ParsedAttrInfo::Spelling MSStructSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "ms_struct"},
  {AttributeCommonInfo::AS_CXX11, "gnu::ms_struct"},
  {AttributeCommonInfo::AS_C23, "gnu::ms_struct"},
};
struct ParsedAttrInfoMSStruct final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoMSStruct() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_MSStruct,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/MSStructSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<RecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "structs, unions, and classes";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) MSStructAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoMSStruct Instance;
};
const ParsedAttrInfoMSStruct ParsedAttrInfoMSStruct::Instance;
static constexpr ParsedAttrInfo::Spelling MatrixTypeSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "matrix_type"},
  {AttributeCommonInfo::AS_CXX11, "clang::matrix_type"},
  {AttributeCommonInfo::AS_C23, "clang::matrix_type"},
};
static constexpr const char *MatrixTypeArgNames[] = {
"NumRows","NumColumns",};
struct ParsedAttrInfoMatrixType final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoMatrixType() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_MatrixType,
    /*NumArgs=*/2,
    /*OptArgs=*/0,
    /*NumArgMembers=*/2,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/MatrixTypeSpellings,
    /*ArgNames=*/MatrixTypeArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<TypedefNameDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "typedefs";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool isParamExpr(size_t N) const override {
  return (N == 0) || (N == 1) || false;
}

static const ParsedAttrInfoMatrixType Instance;
};
const ParsedAttrInfoMatrixType ParsedAttrInfoMatrixType::Instance;
static constexpr ParsedAttrInfo::Spelling MayAliasSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "may_alias"},
  {AttributeCommonInfo::AS_CXX11, "gnu::may_alias"},
  {AttributeCommonInfo::AS_C23, "gnu::may_alias"},
};
struct ParsedAttrInfoMayAlias final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoMayAlias() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_MayAlias,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/MayAliasSpellings,
    /*ArgNames=*/{}) {}
AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) MayAliasAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoMayAlias Instance;
};
const ParsedAttrInfoMayAlias ParsedAttrInfoMayAlias::Instance;
static constexpr ParsedAttrInfo::Spelling MaybeUndefSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "maybe_undef"},
  {AttributeCommonInfo::AS_CXX11, "clang::maybe_undef"},
  {AttributeCommonInfo::AS_C23, "clang::maybe_undef"},
};
struct ParsedAttrInfoMaybeUndef final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoMaybeUndef() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_MaybeUndef,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/MaybeUndefSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ParmVarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "parameters";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable_is_parameter, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) MaybeUndefAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoMaybeUndef Instance;
};
const ParsedAttrInfoMaybeUndef ParsedAttrInfoMaybeUndef::Instance;
static constexpr ParsedAttrInfo::Spelling MicroMipsSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "micromips"},
  {AttributeCommonInfo::AS_CXX11, "gnu::micromips"},
  {AttributeCommonInfo::AS_C23, "gnu::micromips"},
};
struct ParsedAttrInfoMicroMips final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoMicroMips() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_MicroMips,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/MicroMipsSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

  using ParsedAttrInfo::diagMutualExclusion;

  bool diagMutualExclusion(Sema &S, const ParsedAttr &AL, const Decl *D) const override {
    if (const auto *A = D->getAttr<Mips16Attr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    return true;
  }

bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && (T.getArch() == llvm::Triple::mips || T.getArch() == llvm::Triple::mipsel);
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) MicroMipsAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoMicroMips Instance;
};
const ParsedAttrInfoMicroMips ParsedAttrInfoMicroMips::Instance;
static constexpr ParsedAttrInfo::Spelling MinSizeSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "minsize"},
  {AttributeCommonInfo::AS_CXX11, "clang::minsize"},
  {AttributeCommonInfo::AS_C23, "clang::minsize"},
};
struct ParsedAttrInfoMinSize final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoMinSize() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_MinSize,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/MinSizeSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D) && !isa<ObjCMethodDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions and Objective-C methods";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoMinSize Instance;
};
const ParsedAttrInfoMinSize ParsedAttrInfoMinSize::Instance;
static constexpr ParsedAttrInfo::Spelling MinVectorWidthSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "min_vector_width"},
  {AttributeCommonInfo::AS_CXX11, "clang::min_vector_width"},
  {AttributeCommonInfo::AS_C23, "clang::min_vector_width"},
};
static constexpr const char *MinVectorWidthArgNames[] = {
"VectorWidth",};
struct ParsedAttrInfoMinVectorWidth final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoMinVectorWidth() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_MinVectorWidth,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/MinVectorWidthSpellings,
    /*ArgNames=*/MinVectorWidthArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoMinVectorWidth Instance;
};
const ParsedAttrInfoMinVectorWidth ParsedAttrInfoMinVectorWidth::Instance;
static constexpr ParsedAttrInfo::Spelling Mips16Spellings[] = {
  {AttributeCommonInfo::AS_GNU, "mips16"},
  {AttributeCommonInfo::AS_CXX11, "gnu::mips16"},
  {AttributeCommonInfo::AS_C23, "gnu::mips16"},
};
struct ParsedAttrInfoMips16 final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoMips16() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Mips16,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/Mips16Spellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

  using ParsedAttrInfo::diagMutualExclusion;

  bool diagMutualExclusion(Sema &S, const ParsedAttr &AL, const Decl *D) const override {
    if (const auto *A = D->getAttr<MipsInterruptAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    if (const auto *A = D->getAttr<MicroMipsAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    return true;
  }

bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && (T.getArch() == llvm::Triple::mips || T.getArch() == llvm::Triple::mipsel);
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) Mips16Attr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoMips16 Instance;
};
const ParsedAttrInfoMips16 ParsedAttrInfoMips16::Instance;
static constexpr ParsedAttrInfo::Spelling MipsLongCallSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "long_call"},
  {AttributeCommonInfo::AS_CXX11, "gnu::long_call"},
  {AttributeCommonInfo::AS_C23, "gnu::long_call"},
  {AttributeCommonInfo::AS_GNU, "far"},
  {AttributeCommonInfo::AS_CXX11, "gnu::far"},
  {AttributeCommonInfo::AS_C23, "gnu::far"},
};
struct ParsedAttrInfoMipsLongCall final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoMipsLongCall() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_MipsLongCall,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/MipsLongCallSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

  using ParsedAttrInfo::diagMutualExclusion;

  bool diagMutualExclusion(Sema &S, const ParsedAttr &AL, const Decl *D) const override {
    if (const auto *A = D->getAttr<MipsShortCallAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    return true;
  }

bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && (T.getArch() == llvm::Triple::mips || T.getArch() == llvm::Triple::mipsel || T.getArch() == llvm::Triple::mips64 || T.getArch() == llvm::Triple::mips64el);
}

unsigned spellingIndexToSemanticSpelling(const ParsedAttr &Attr) const override {
  enum Spelling {
    GNU_long_call = 0,
    CXX11_gnu_long_call = 1,
    C23_gnu_long_call = 2,
    GNU_far = 3,
    CXX11_gnu_far = 4,
    C23_gnu_far = 5,
  SpellingNotCalculated = 15

  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return GNU_long_call;
    case 1: return CXX11_gnu_long_call;
    case 2: return C23_gnu_long_call;
    case 3: return GNU_far;
    case 4: return CXX11_gnu_far;
    case 5: return C23_gnu_far;
  }
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) MipsLongCallAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoMipsLongCall Instance;
};
const ParsedAttrInfoMipsLongCall ParsedAttrInfoMipsLongCall::Instance;
static constexpr ParsedAttrInfo::Spelling MipsShortCallSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "short_call"},
  {AttributeCommonInfo::AS_CXX11, "gnu::short_call"},
  {AttributeCommonInfo::AS_C23, "gnu::short_call"},
  {AttributeCommonInfo::AS_GNU, "near"},
  {AttributeCommonInfo::AS_CXX11, "gnu::near"},
  {AttributeCommonInfo::AS_C23, "gnu::near"},
};
struct ParsedAttrInfoMipsShortCall final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoMipsShortCall() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_MipsShortCall,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/MipsShortCallSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

  using ParsedAttrInfo::diagMutualExclusion;

  bool diagMutualExclusion(Sema &S, const ParsedAttr &AL, const Decl *D) const override {
    if (const auto *A = D->getAttr<MipsLongCallAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    return true;
  }

bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && (T.getArch() == llvm::Triple::mips || T.getArch() == llvm::Triple::mipsel || T.getArch() == llvm::Triple::mips64 || T.getArch() == llvm::Triple::mips64el);
}

unsigned spellingIndexToSemanticSpelling(const ParsedAttr &Attr) const override {
  enum Spelling {
    GNU_short_call = 0,
    CXX11_gnu_short_call = 1,
    C23_gnu_short_call = 2,
    GNU_near = 3,
    CXX11_gnu_near = 4,
    C23_gnu_near = 5,
  SpellingNotCalculated = 15

  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return GNU_short_call;
    case 1: return CXX11_gnu_short_call;
    case 2: return C23_gnu_short_call;
    case 3: return GNU_near;
    case 4: return CXX11_gnu_near;
    case 5: return C23_gnu_near;
  }
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) MipsShortCallAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoMipsShortCall Instance;
};
const ParsedAttrInfoMipsShortCall ParsedAttrInfoMipsShortCall::Instance;
static constexpr ParsedAttrInfo::Spelling ModeSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "mode"},
  {AttributeCommonInfo::AS_CXX11, "gnu::mode"},
  {AttributeCommonInfo::AS_C23, "gnu::mode"},
};
static constexpr const char *ModeArgNames[] = {
"Mode",};
struct ParsedAttrInfoMode final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoMode() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Mode,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/ModeSpellings,
    /*ArgNames=*/ModeArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<VarDecl>(D) && !isa<EnumDecl>(D) && !isa<TypedefNameDecl>(D) && !isa<FieldDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "variables, enums, typedefs, and non-static data members";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoMode Instance;
};
const ParsedAttrInfoMode ParsedAttrInfoMode::Instance;
static constexpr ParsedAttrInfo::Spelling MustTailSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "musttail"},
  {AttributeCommonInfo::AS_CXX11, "clang::musttail"},
  {AttributeCommonInfo::AS_C23, "clang::musttail"},
};
struct ParsedAttrInfoMustTail final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoMustTail() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_MustTail,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/1,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/MustTailSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &AL, const Decl *D) const override {
  S.Diag(AL.getLoc(), diag::err_attribute_invalid_on_decl)
    << AL << AL.isRegularKeywordAttribute() << D->getLocation();
  return false;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &Attr, const Stmt *St) const override {
  if (!isa<ReturnStmt>(St)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "return statements";
    return false;
  }
  return true;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoMustTail Instance;
};
const ParsedAttrInfoMustTail ParsedAttrInfoMustTail::Instance;
static constexpr ParsedAttrInfo::Spelling NSConsumedSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "ns_consumed"},
  {AttributeCommonInfo::AS_CXX11, "clang::ns_consumed"},
  {AttributeCommonInfo::AS_C23, "clang::ns_consumed"},
};
struct ParsedAttrInfoNSConsumed final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoNSConsumed() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_NSConsumed,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/NSConsumedSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ParmVarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "parameters";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable_is_parameter, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoNSConsumed Instance;
};
const ParsedAttrInfoNSConsumed ParsedAttrInfoNSConsumed::Instance;
static constexpr ParsedAttrInfo::Spelling NSConsumesSelfSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "ns_consumes_self"},
  {AttributeCommonInfo::AS_CXX11, "clang::ns_consumes_self"},
  {AttributeCommonInfo::AS_C23, "clang::ns_consumes_self"},
};
struct ParsedAttrInfoNSConsumesSelf final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoNSConsumesSelf() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_NSConsumesSelf,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/NSConsumesSelfSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ObjCMethodDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "Objective-C methods";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) NSConsumesSelfAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoNSConsumesSelf Instance;
};
const ParsedAttrInfoNSConsumesSelf ParsedAttrInfoNSConsumesSelf::Instance;
static constexpr ParsedAttrInfo::Spelling NSErrorDomainSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "ns_error_domain"},
};
static constexpr const char *NSErrorDomainArgNames[] = {
"ErrorDomain",};
struct ParsedAttrInfoNSErrorDomain final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoNSErrorDomain() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_NSErrorDomain,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/NSErrorDomainSpellings,
    /*ArgNames=*/NSErrorDomainArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<EnumDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "enums";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_enum, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoNSErrorDomain Instance;
};
const ParsedAttrInfoNSErrorDomain ParsedAttrInfoNSErrorDomain::Instance;
static constexpr ParsedAttrInfo::Spelling NSReturnsAutoreleasedSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "ns_returns_autoreleased"},
  {AttributeCommonInfo::AS_CXX11, "clang::ns_returns_autoreleased"},
  {AttributeCommonInfo::AS_C23, "clang::ns_returns_autoreleased"},
};
struct ParsedAttrInfoNSReturnsAutoreleased final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoNSReturnsAutoreleased() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_NSReturnsAutoreleased,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/NSReturnsAutoreleasedSpellings,
    /*ArgNames=*/{}) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoNSReturnsAutoreleased Instance;
};
const ParsedAttrInfoNSReturnsAutoreleased ParsedAttrInfoNSReturnsAutoreleased::Instance;
static constexpr ParsedAttrInfo::Spelling NSReturnsNotRetainedSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "ns_returns_not_retained"},
  {AttributeCommonInfo::AS_CXX11, "clang::ns_returns_not_retained"},
  {AttributeCommonInfo::AS_C23, "clang::ns_returns_not_retained"},
};
struct ParsedAttrInfoNSReturnsNotRetained final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoNSReturnsNotRetained() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_NSReturnsNotRetained,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/NSReturnsNotRetainedSpellings,
    /*ArgNames=*/{}) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoNSReturnsNotRetained Instance;
};
const ParsedAttrInfoNSReturnsNotRetained ParsedAttrInfoNSReturnsNotRetained::Instance;
static constexpr ParsedAttrInfo::Spelling NSReturnsRetainedSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "ns_returns_retained"},
  {AttributeCommonInfo::AS_CXX11, "clang::ns_returns_retained"},
  {AttributeCommonInfo::AS_C23, "clang::ns_returns_retained"},
};
struct ParsedAttrInfoNSReturnsRetained final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoNSReturnsRetained() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_NSReturnsRetained,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/NSReturnsRetainedSpellings,
    /*ArgNames=*/{}) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoNSReturnsRetained Instance;
};
const ParsedAttrInfoNSReturnsRetained ParsedAttrInfoNSReturnsRetained::Instance;
static constexpr ParsedAttrInfo::Spelling NVPTXKernelSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "nvptx_kernel"},
  {AttributeCommonInfo::AS_CXX11, "clang::nvptx_kernel"},
  {AttributeCommonInfo::AS_C23, "clang::nvptx_kernel"},
};
struct ParsedAttrInfoNVPTXKernel final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoNVPTXKernel() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_NVPTXKernel,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/NVPTXKernelSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && (T.getArch() == llvm::Triple::nvptx || T.getArch() == llvm::Triple::nvptx64);
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoNVPTXKernel Instance;
};
const ParsedAttrInfoNVPTXKernel ParsedAttrInfoNVPTXKernel::Instance;
static constexpr ParsedAttrInfo::Spelling NakedSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "naked"},
  {AttributeCommonInfo::AS_CXX11, "gnu::naked"},
  {AttributeCommonInfo::AS_C23, "gnu::naked"},
  {AttributeCommonInfo::AS_Declspec, "naked"},
};
struct ParsedAttrInfoNaked final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoNaked() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Naked,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/NakedSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

  using ParsedAttrInfo::diagMutualExclusion;

  bool diagMutualExclusion(Sema &S, const ParsedAttr &AL, const Decl *D) const override {
    if (const auto *A = D->getAttr<DisableTailCallsAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    return true;
  }

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoNaked Instance;
};
const ParsedAttrInfoNaked ParsedAttrInfoNaked::Instance;
static constexpr ParsedAttrInfo::Spelling NeonPolyVectorTypeSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "neon_polyvector_type"},
  {AttributeCommonInfo::AS_CXX11, "clang::neon_polyvector_type"},
  {AttributeCommonInfo::AS_C23, "clang::neon_polyvector_type"},
};
static constexpr const char *NeonPolyVectorTypeArgNames[] = {
"NumElements",};
struct ParsedAttrInfoNeonPolyVectorType final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoNeonPolyVectorType() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_NeonPolyVectorType,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/NeonPolyVectorTypeSpellings,
    /*ArgNames=*/NeonPolyVectorTypeArgNames) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoNeonPolyVectorType Instance;
};
const ParsedAttrInfoNeonPolyVectorType ParsedAttrInfoNeonPolyVectorType::Instance;
static constexpr ParsedAttrInfo::Spelling NeonVectorTypeSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "neon_vector_type"},
  {AttributeCommonInfo::AS_CXX11, "clang::neon_vector_type"},
  {AttributeCommonInfo::AS_C23, "clang::neon_vector_type"},
};
static constexpr const char *NeonVectorTypeArgNames[] = {
"NumElements",};
struct ParsedAttrInfoNeonVectorType final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoNeonVectorType() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_NeonVectorType,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/NeonVectorTypeSpellings,
    /*ArgNames=*/NeonVectorTypeArgNames) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoNeonVectorType Instance;
};
const ParsedAttrInfoNeonVectorType ParsedAttrInfoNeonVectorType::Instance;
static constexpr ParsedAttrInfo::Spelling NoAliasSpellings[] = {
  {AttributeCommonInfo::AS_Declspec, "noalias"},
};
struct ParsedAttrInfoNoAlias final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoNoAlias() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_NoAlias,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/NoAliasSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) NoAliasAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoNoAlias Instance;
};
const ParsedAttrInfoNoAlias ParsedAttrInfoNoAlias::Instance;
static constexpr ParsedAttrInfo::Spelling NoBuiltinSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "no_builtin"},
  {AttributeCommonInfo::AS_CXX11, "clang::no_builtin"},
  {AttributeCommonInfo::AS_C23, "clang::no_builtin"},
};
static constexpr const char *NoBuiltinArgNames[] = {
"BuiltinNames...",};
struct ParsedAttrInfoNoBuiltin final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoNoBuiltin() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_NoBuiltin,
    /*NumArgs=*/0,
    /*OptArgs=*/15,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/NoBuiltinSpellings,
    /*ArgNames=*/NoBuiltinArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoNoBuiltin Instance;
};
const ParsedAttrInfoNoBuiltin ParsedAttrInfoNoBuiltin::Instance;
static constexpr ParsedAttrInfo::Spelling NoCommonSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "nocommon"},
  {AttributeCommonInfo::AS_CXX11, "gnu::nocommon"},
  {AttributeCommonInfo::AS_C23, "gnu::nocommon"},
};
struct ParsedAttrInfoNoCommon final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoNoCommon() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_NoCommon,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/NoCommonSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<VarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "variables";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) NoCommonAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoNoCommon Instance;
};
const ParsedAttrInfoNoCommon ParsedAttrInfoNoCommon::Instance;
static constexpr ParsedAttrInfo::Spelling NoConvergentSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "noconvergent"},
  {AttributeCommonInfo::AS_CXX11, "clang::noconvergent"},
  {AttributeCommonInfo::AS_C23, "clang::noconvergent"},
  {AttributeCommonInfo::AS_Declspec, "noconvergent"},
};
struct ParsedAttrInfoNoConvergent final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoNoConvergent() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_NoConvergent,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/NoConvergentSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions and statements";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &Attr, const Stmt *St) const override {
  if (!isa<Stmt>(St)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions and statements";
    return false;
  }
  return true;
}

  using ParsedAttrInfo::diagMutualExclusion;

  bool diagMutualExclusion(Sema &S, const ParsedAttr &AL, const Decl *D) const override {
    if (const auto *A = D->getAttr<ConvergentAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    return true;
  }

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.CUDA;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) NoConvergentAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoNoConvergent Instance;
};
const ParsedAttrInfoNoConvergent ParsedAttrInfoNoConvergent::Instance;
static constexpr ParsedAttrInfo::Spelling NoDebugSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "nodebug"},
  {AttributeCommonInfo::AS_CXX11, "gnu::nodebug"},
  {AttributeCommonInfo::AS_C23, "gnu::nodebug"},
};
struct ParsedAttrInfoNoDebug final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoNoDebug() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_NoDebug,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/NoDebugSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<TypedefNameDecl>(D) && !isFunctionLike(D) && !isa<ObjCMethodDecl>(D) && !isNonParmVar(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "typedefs, functions, function pointers, Objective-C methods, and variables";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_type_alias, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_hasType_functionType, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable_not_is_parameter, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoNoDebug Instance;
};
const ParsedAttrInfoNoDebug ParsedAttrInfoNoDebug::Instance;
static constexpr ParsedAttrInfo::Spelling NoDerefSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "noderef"},
  {AttributeCommonInfo::AS_CXX11, "clang::noderef"},
  {AttributeCommonInfo::AS_C23, "clang::noderef"},
};
struct ParsedAttrInfoNoDeref final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoNoDeref() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_NoDeref,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/NoDerefSpellings,
    /*ArgNames=*/{}) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoNoDeref Instance;
};
const ParsedAttrInfoNoDeref ParsedAttrInfoNoDeref::Instance;
static constexpr ParsedAttrInfo::Spelling NoDestroySpellings[] = {
  {AttributeCommonInfo::AS_GNU, "no_destroy"},
  {AttributeCommonInfo::AS_CXX11, "clang::no_destroy"},
};
struct ParsedAttrInfoNoDestroy final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoNoDestroy() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_NoDestroy,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/NoDestroySpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<VarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "variables";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

  using ParsedAttrInfo::diagMutualExclusion;

  bool diagMutualExclusion(Sema &S, const ParsedAttr &AL, const Decl *D) const override {
    if (const auto *A = D->getAttr<AlwaysDestroyAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    return true;
  }

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoNoDestroy Instance;
};
const ParsedAttrInfoNoDestroy ParsedAttrInfoNoDestroy::Instance;
static constexpr ParsedAttrInfo::Spelling NoDuplicateSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "noduplicate"},
  {AttributeCommonInfo::AS_CXX11, "clang::noduplicate"},
  {AttributeCommonInfo::AS_C23, "clang::noduplicate"},
};
struct ParsedAttrInfoNoDuplicate final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoNoDuplicate() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_NoDuplicate,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/NoDuplicateSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) NoDuplicateAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoNoDuplicate Instance;
};
const ParsedAttrInfoNoDuplicate ParsedAttrInfoNoDuplicate::Instance;
static constexpr ParsedAttrInfo::Spelling NoEscapeSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "noescape"},
  {AttributeCommonInfo::AS_CXX11, "clang::noescape"},
  {AttributeCommonInfo::AS_C23, "clang::noescape"},
};
struct ParsedAttrInfoNoEscape final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoNoEscape() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_NoEscape,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/NoEscapeSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ParmVarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "parameters";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable_is_parameter, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoNoEscape Instance;
};
const ParsedAttrInfoNoEscape ParsedAttrInfoNoEscape::Instance;
static constexpr ParsedAttrInfo::Spelling NoInlineSpellings[] = {
  {AttributeCommonInfo::AS_Keyword, "__noinline__"},
  {AttributeCommonInfo::AS_GNU, "noinline"},
  {AttributeCommonInfo::AS_CXX11, "gnu::noinline"},
  {AttributeCommonInfo::AS_C23, "gnu::noinline"},
  {AttributeCommonInfo::AS_CXX11, "clang::noinline"},
  {AttributeCommonInfo::AS_C23, "clang::noinline"},
  {AttributeCommonInfo::AS_CXX11, "msvc::noinline"},
  {AttributeCommonInfo::AS_C23, "msvc::noinline"},
  {AttributeCommonInfo::AS_Declspec, "noinline"},
};
struct ParsedAttrInfoNoInline final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoNoInline() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_NoInline,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/1,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/NoInlineSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions and statements";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &Attr, const Stmt *St) const override {
  if (!isa<Stmt>(St)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions and statements";
    return false;
  }
  return true;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) NoInlineAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoNoInline Instance;
};
const ParsedAttrInfoNoInline ParsedAttrInfoNoInline::Instance;
static constexpr ParsedAttrInfo::Spelling NoInstrumentFunctionSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "no_instrument_function"},
  {AttributeCommonInfo::AS_CXX11, "gnu::no_instrument_function"},
  {AttributeCommonInfo::AS_C23, "gnu::no_instrument_function"},
};
struct ParsedAttrInfoNoInstrumentFunction final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoNoInstrumentFunction() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_NoInstrumentFunction,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/NoInstrumentFunctionSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D) && !isa<ObjCMethodDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions and Objective-C methods";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) NoInstrumentFunctionAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoNoInstrumentFunction Instance;
};
const ParsedAttrInfoNoInstrumentFunction ParsedAttrInfoNoInstrumentFunction::Instance;
static constexpr ParsedAttrInfo::Spelling NoMergeSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "nomerge"},
  {AttributeCommonInfo::AS_CXX11, "clang::nomerge"},
  {AttributeCommonInfo::AS_C23, "clang::nomerge"},
};
struct ParsedAttrInfoNoMerge final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoNoMerge() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_NoMerge,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/1,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/NoMergeSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D) && !isa<VarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions, statements and variables";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &Attr, const Stmt *St) const override {
  if (!isa<Stmt>(St)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions, statements and variables";
    return false;
  }
  return true;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoNoMerge Instance;
};
const ParsedAttrInfoNoMerge ParsedAttrInfoNoMerge::Instance;
static constexpr ParsedAttrInfo::Spelling NoMicroMipsSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "nomicromips"},
  {AttributeCommonInfo::AS_CXX11, "gnu::nomicromips"},
  {AttributeCommonInfo::AS_C23, "gnu::nomicromips"},
};
struct ParsedAttrInfoNoMicroMips final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoNoMicroMips() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_NoMicroMips,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/NoMicroMipsSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && (T.getArch() == llvm::Triple::mips || T.getArch() == llvm::Triple::mipsel);
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) NoMicroMipsAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoNoMicroMips Instance;
};
const ParsedAttrInfoNoMicroMips ParsedAttrInfoNoMicroMips::Instance;
static constexpr ParsedAttrInfo::Spelling NoMips16Spellings[] = {
  {AttributeCommonInfo::AS_GNU, "nomips16"},
  {AttributeCommonInfo::AS_CXX11, "gnu::nomips16"},
  {AttributeCommonInfo::AS_C23, "gnu::nomips16"},
};
struct ParsedAttrInfoNoMips16 final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoNoMips16() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_NoMips16,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/NoMips16Spellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && (T.getArch() == llvm::Triple::mips || T.getArch() == llvm::Triple::mipsel);
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) NoMips16Attr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoNoMips16 Instance;
};
const ParsedAttrInfoNoMips16 ParsedAttrInfoNoMips16::Instance;
static constexpr ParsedAttrInfo::Spelling NoProfileFunctionSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "no_profile_instrument_function"},
  {AttributeCommonInfo::AS_CXX11, "gnu::no_profile_instrument_function"},
  {AttributeCommonInfo::AS_C23, "gnu::no_profile_instrument_function"},
};
struct ParsedAttrInfoNoProfileFunction final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoNoProfileFunction() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_NoProfileFunction,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/NoProfileFunctionSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) NoProfileFunctionAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoNoProfileFunction Instance;
};
const ParsedAttrInfoNoProfileFunction ParsedAttrInfoNoProfileFunction::Instance;
static constexpr ParsedAttrInfo::Spelling NoRandomizeLayoutSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "no_randomize_layout"},
  {AttributeCommonInfo::AS_CXX11, "gnu::no_randomize_layout"},
  {AttributeCommonInfo::AS_C23, "gnu::no_randomize_layout"},
};
struct ParsedAttrInfoNoRandomizeLayout final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoNoRandomizeLayout() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_NoRandomizeLayout,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/NoRandomizeLayoutSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<RecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "structs, unions, and classes";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

  using ParsedAttrInfo::diagMutualExclusion;

  bool diagMutualExclusion(Sema &S, const ParsedAttr &AL, const Decl *D) const override {
    if (const auto *A = D->getAttr<RandomizeLayoutAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    return true;
  }

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return (!LangOpts.CPlusPlus);
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoNoRandomizeLayout Instance;
};
const ParsedAttrInfoNoRandomizeLayout ParsedAttrInfoNoRandomizeLayout::Instance;
static constexpr ParsedAttrInfo::Spelling NoReturnSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "noreturn"},
  {AttributeCommonInfo::AS_CXX11, "gnu::noreturn"},
  {AttributeCommonInfo::AS_C23, "gnu::noreturn"},
  {AttributeCommonInfo::AS_Declspec, "noreturn"},
};
struct ParsedAttrInfoNoReturn final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoNoReturn() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_NoReturn,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/NoReturnSpellings,
    /*ArgNames=*/{}) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoNoReturn Instance;
};
const ParsedAttrInfoNoReturn ParsedAttrInfoNoReturn::Instance;
static constexpr ParsedAttrInfo::Spelling NoSanitizeSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "no_sanitize"},
  {AttributeCommonInfo::AS_CXX11, "clang::no_sanitize"},
  {AttributeCommonInfo::AS_CXX11, "gnu::no_sanitize"},
  {AttributeCommonInfo::AS_C23, "clang::no_sanitize"},
  {AttributeCommonInfo::AS_C23, "gnu::no_sanitize"},
};
static constexpr const char *NoSanitizeArgNames[] = {
"Sanitizers...",};
struct ParsedAttrInfoNoSanitize final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoNoSanitize() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_NoSanitize,
    /*NumArgs=*/0,
    /*OptArgs=*/15,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/NoSanitizeSpellings,
    /*ArgNames=*/NoSanitizeArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D) && !isa<ObjCMethodDecl>(D) && !isGlobalVar(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions, Objective-C methods, and global variables";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable_is_global, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoNoSanitize Instance;
};
const ParsedAttrInfoNoSanitize ParsedAttrInfoNoSanitize::Instance;
static constexpr ParsedAttrInfo::Spelling NoSanitizeSpecificSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "no_address_safety_analysis"},
  {AttributeCommonInfo::AS_CXX11, "gnu::no_address_safety_analysis"},
  {AttributeCommonInfo::AS_C23, "gnu::no_address_safety_analysis"},
  {AttributeCommonInfo::AS_GNU, "no_sanitize_address"},
  {AttributeCommonInfo::AS_CXX11, "gnu::no_sanitize_address"},
  {AttributeCommonInfo::AS_C23, "gnu::no_sanitize_address"},
  {AttributeCommonInfo::AS_GNU, "no_sanitize_thread"},
  {AttributeCommonInfo::AS_CXX11, "gnu::no_sanitize_thread"},
  {AttributeCommonInfo::AS_C23, "gnu::no_sanitize_thread"},
  {AttributeCommonInfo::AS_GNU, "no_sanitize_memory"},
  {AttributeCommonInfo::AS_CXX11, "clang::no_sanitize_memory"},
  {AttributeCommonInfo::AS_C23, "clang::no_sanitize_memory"},
};
struct ParsedAttrInfoNoSanitizeSpecific final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoNoSanitizeSpecific() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_NoSanitizeSpecific,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/NoSanitizeSpecificSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D) && !isGlobalVar(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions and global variables";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable_is_global, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoNoSanitizeSpecific Instance;
};
const ParsedAttrInfoNoSanitizeSpecific ParsedAttrInfoNoSanitizeSpecific::Instance;
static constexpr ParsedAttrInfo::Spelling NoSpecializationsSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "no_specializations"},
  {AttributeCommonInfo::AS_CXX11, "clang::no_specializations"},
};
static constexpr const char *NoSpecializationsArgNames[] = {
"Message",};
struct ParsedAttrInfoNoSpecializations final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoNoSpecializations() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_NoSpecializations,
    /*NumArgs=*/0,
    /*OptArgs=*/1,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/NoSpecializationsSpellings,
    /*ArgNames=*/NoSpecializationsArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isClassTmpl(D) && !isFunctionTmpl(D) && !isVarTmpl(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "class templates, function templates, and variable templates";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoNoSpecializations Instance;
};
const ParsedAttrInfoNoSpecializations ParsedAttrInfoNoSpecializations::Instance;
static constexpr ParsedAttrInfo::Spelling NoSpeculativeLoadHardeningSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "no_speculative_load_hardening"},
  {AttributeCommonInfo::AS_CXX11, "clang::no_speculative_load_hardening"},
  {AttributeCommonInfo::AS_C23, "clang::no_speculative_load_hardening"},
};
struct ParsedAttrInfoNoSpeculativeLoadHardening final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoNoSpeculativeLoadHardening() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_NoSpeculativeLoadHardening,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/NoSpeculativeLoadHardeningSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D) && !isa<ObjCMethodDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions and Objective-C methods";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

  using ParsedAttrInfo::diagMutualExclusion;

  bool diagMutualExclusion(Sema &S, const ParsedAttr &AL, const Decl *D) const override {
    if (const auto *A = D->getAttr<SpeculativeLoadHardeningAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    return true;
  }

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) NoSpeculativeLoadHardeningAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoNoSpeculativeLoadHardening Instance;
};
const ParsedAttrInfoNoSpeculativeLoadHardening ParsedAttrInfoNoSpeculativeLoadHardening::Instance;
static constexpr ParsedAttrInfo::Spelling NoSplitStackSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "no_split_stack"},
  {AttributeCommonInfo::AS_CXX11, "gnu::no_split_stack"},
  {AttributeCommonInfo::AS_C23, "gnu::no_split_stack"},
};
struct ParsedAttrInfoNoSplitStack final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoNoSplitStack() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_NoSplitStack,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/NoSplitStackSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) NoSplitStackAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoNoSplitStack Instance;
};
const ParsedAttrInfoNoSplitStack ParsedAttrInfoNoSplitStack::Instance;
static constexpr ParsedAttrInfo::Spelling NoStackProtectorSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "no_stack_protector"},
  {AttributeCommonInfo::AS_CXX11, "clang::no_stack_protector"},
  {AttributeCommonInfo::AS_C23, "clang::no_stack_protector"},
  {AttributeCommonInfo::AS_CXX11, "gnu::no_stack_protector"},
  {AttributeCommonInfo::AS_C23, "gnu::no_stack_protector"},
  {AttributeCommonInfo::AS_Declspec, "safebuffers"},
};
struct ParsedAttrInfoNoStackProtector final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoNoStackProtector() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_NoStackProtector,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/NoStackProtectorSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

unsigned spellingIndexToSemanticSpelling(const ParsedAttr &Attr) const override {
  enum Spelling {
    GNU_no_stack_protector = 0,
    CXX11_clang_no_stack_protector = 1,
    C23_clang_no_stack_protector = 2,
    CXX11_gnu_no_stack_protector = 3,
    C23_gnu_no_stack_protector = 4,
    Declspec_safebuffers = 5,
  SpellingNotCalculated = 15

  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return GNU_no_stack_protector;
    case 1: return CXX11_clang_no_stack_protector;
    case 2: return C23_clang_no_stack_protector;
    case 3: return CXX11_gnu_no_stack_protector;
    case 4: return C23_gnu_no_stack_protector;
    case 5: return Declspec_safebuffers;
  }
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) NoStackProtectorAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoNoStackProtector Instance;
};
const ParsedAttrInfoNoStackProtector ParsedAttrInfoNoStackProtector::Instance;
static constexpr ParsedAttrInfo::Spelling NoThreadSafetyAnalysisSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "no_thread_safety_analysis"},
  {AttributeCommonInfo::AS_CXX11, "clang::no_thread_safety_analysis"},
  {AttributeCommonInfo::AS_C23, "clang::no_thread_safety_analysis"},
};
struct ParsedAttrInfoNoThreadSafetyAnalysis final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoNoThreadSafetyAnalysis() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_NoThreadSafetyAnalysis,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/NoThreadSafetyAnalysisSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) NoThreadSafetyAnalysisAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoNoThreadSafetyAnalysis Instance;
};
const ParsedAttrInfoNoThreadSafetyAnalysis ParsedAttrInfoNoThreadSafetyAnalysis::Instance;
static constexpr ParsedAttrInfo::Spelling NoThrowSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "nothrow"},
  {AttributeCommonInfo::AS_CXX11, "gnu::nothrow"},
  {AttributeCommonInfo::AS_C23, "gnu::nothrow"},
  {AttributeCommonInfo::AS_Declspec, "nothrow"},
};
struct ParsedAttrInfoNoThrow final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoNoThrow() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_NoThrow,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/NoThrowSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isFunctionLike(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions and function pointers";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_hasType_functionType, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoNoThrow Instance;
};
const ParsedAttrInfoNoThrow ParsedAttrInfoNoThrow::Instance;
static constexpr ParsedAttrInfo::Spelling NoTrivialAutoVarInitSpellings[] = {
  {AttributeCommonInfo::AS_Declspec, "no_init_all"},
};
struct ParsedAttrInfoNoTrivialAutoVarInit final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoNoTrivialAutoVarInit() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_NoTrivialAutoVarInit,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/NoTrivialAutoVarInitSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D) && !isa<TagDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions and tag types";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) NoTrivialAutoVarInitAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoNoTrivialAutoVarInit Instance;
};
const ParsedAttrInfoNoTrivialAutoVarInit ParsedAttrInfoNoTrivialAutoVarInit::Instance;
static constexpr ParsedAttrInfo::Spelling NoUniqueAddressSpellings[] = {
  {AttributeCommonInfo::AS_CXX11, "no_unique_address"},
  {AttributeCommonInfo::AS_CXX11, "msvc::no_unique_address"},
};
struct ParsedAttrInfoNoUniqueAddress final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoNoUniqueAddress() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_NoUniqueAddress,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/NoUniqueAddressSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isNonBitField(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "non-bit-field non-static data members";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool spellingExistsInTarget(const TargetInfo &Target,
                            const unsigned SpellingListIndex) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return ((SpellingListIndex == 0) && true && ( Target.getCXXABI().isItaniumFamily() )) || ((SpellingListIndex == 1) && true && (T.getArch() == llvm::Triple::x86 || T.getArch() == llvm::Triple::x86_64 || T.getArch() == llvm::Triple::arm || T.getArch() == llvm::Triple::thumb || T.getArch() == llvm::Triple::aarch64) && ( Target.getCXXABI().isMicrosoft() ));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoNoUniqueAddress Instance;
};
const ParsedAttrInfoNoUniqueAddress ParsedAttrInfoNoUniqueAddress::Instance;
static constexpr ParsedAttrInfo::Spelling NoUwtableSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "nouwtable"},
  {AttributeCommonInfo::AS_CXX11, "clang::nouwtable"},
  {AttributeCommonInfo::AS_C23, "clang::nouwtable"},
};
struct ParsedAttrInfoNoUwtable final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoNoUwtable() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_NoUwtable,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/NoUwtableSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isFunctionLike(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions and function pointers";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_hasType_functionType, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) NoUwtableAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoNoUwtable Instance;
};
const ParsedAttrInfoNoUwtable ParsedAttrInfoNoUwtable::Instance;
static constexpr ParsedAttrInfo::Spelling NonAllocatingSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "nonallocating"},
  {AttributeCommonInfo::AS_CXX11, "clang::nonallocating"},
  {AttributeCommonInfo::AS_C23, "clang::nonallocating"},
};
static constexpr const char *NonAllocatingArgNames[] = {
"Cond",};
struct ParsedAttrInfoNonAllocating final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoNonAllocating() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_NonAllocating,
    /*NumArgs=*/0,
    /*OptArgs=*/1,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/NonAllocatingSpellings,
    /*ArgNames=*/NonAllocatingArgNames) {}
bool isParamExpr(size_t N) const override {
  return (N == 0) || false;
}

static const ParsedAttrInfoNonAllocating Instance;
};
const ParsedAttrInfoNonAllocating ParsedAttrInfoNonAllocating::Instance;
static constexpr ParsedAttrInfo::Spelling NonBlockingSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "nonblocking"},
  {AttributeCommonInfo::AS_CXX11, "clang::nonblocking"},
  {AttributeCommonInfo::AS_C23, "clang::nonblocking"},
};
static constexpr const char *NonBlockingArgNames[] = {
"Cond",};
struct ParsedAttrInfoNonBlocking final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoNonBlocking() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_NonBlocking,
    /*NumArgs=*/0,
    /*OptArgs=*/1,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/NonBlockingSpellings,
    /*ArgNames=*/NonBlockingArgNames) {}
bool isParamExpr(size_t N) const override {
  return (N == 0) || false;
}

static const ParsedAttrInfoNonBlocking Instance;
};
const ParsedAttrInfoNonBlocking ParsedAttrInfoNonBlocking::Instance;
static constexpr ParsedAttrInfo::Spelling NonNullSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "nonnull"},
  {AttributeCommonInfo::AS_CXX11, "gnu::nonnull"},
  {AttributeCommonInfo::AS_C23, "gnu::nonnull"},
};
static constexpr const char *NonNullArgNames[] = {
"Args...",};
struct ParsedAttrInfoNonNull final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoNonNull() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_NonNull,
    /*NumArgs=*/0,
    /*OptArgs=*/15,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/NonNullSpellings,
    /*ArgNames=*/NonNullArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ObjCMethodDecl>(D) && !isHasFunctionProto(D) && !isa<ParmVarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions, methods, and parameters";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoNonNull Instance;
};
const ParsedAttrInfoNonNull ParsedAttrInfoNonNull::Instance;
static constexpr ParsedAttrInfo::Spelling NotTailCalledSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "not_tail_called"},
  {AttributeCommonInfo::AS_CXX11, "clang::not_tail_called"},
  {AttributeCommonInfo::AS_C23, "clang::not_tail_called"},
};
struct ParsedAttrInfoNotTailCalled final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoNotTailCalled() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_NotTailCalled,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/NotTailCalledSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

  using ParsedAttrInfo::diagMutualExclusion;

  bool diagMutualExclusion(Sema &S, const ParsedAttr &AL, const Decl *D) const override {
    if (const auto *A = D->getAttr<AlwaysInlineAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    return true;
  }

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) NotTailCalledAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoNotTailCalled Instance;
};
const ParsedAttrInfoNotTailCalled ParsedAttrInfoNotTailCalled::Instance;
static constexpr ParsedAttrInfo::Spelling OMPAssumeSpellings[] = {
  {AttributeCommonInfo::AS_CXX11, "omp::assume"},
};
static constexpr const char *OMPAssumeArgNames[] = {
"Assumption",};
struct ParsedAttrInfoOMPAssume final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoOMPAssume() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_OMPAssume,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/OMPAssumeSpellings,
    /*ArgNames=*/OMPAssumeArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D) && !isa<ObjCMethodDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions and Objective-C methods";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoOMPAssume Instance;
};
const ParsedAttrInfoOMPAssume ParsedAttrInfoOMPAssume::Instance;
static constexpr ParsedAttrInfo::Spelling OSConsumedSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "os_consumed"},
  {AttributeCommonInfo::AS_CXX11, "clang::os_consumed"},
  {AttributeCommonInfo::AS_C23, "clang::os_consumed"},
};
struct ParsedAttrInfoOSConsumed final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoOSConsumed() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_OSConsumed,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/OSConsumedSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ParmVarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "parameters";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable_is_parameter, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoOSConsumed Instance;
};
const ParsedAttrInfoOSConsumed ParsedAttrInfoOSConsumed::Instance;
static constexpr ParsedAttrInfo::Spelling OSConsumesThisSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "os_consumes_this"},
  {AttributeCommonInfo::AS_CXX11, "clang::os_consumes_this"},
  {AttributeCommonInfo::AS_C23, "clang::os_consumes_this"},
};
struct ParsedAttrInfoOSConsumesThis final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoOSConsumesThis() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_OSConsumesThis,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/OSConsumesThisSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isNonStaticCXXMethod(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "non-static member functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) OSConsumesThisAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoOSConsumesThis Instance;
};
const ParsedAttrInfoOSConsumesThis ParsedAttrInfoOSConsumesThis::Instance;
static constexpr ParsedAttrInfo::Spelling OSReturnsNotRetainedSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "os_returns_not_retained"},
  {AttributeCommonInfo::AS_CXX11, "clang::os_returns_not_retained"},
  {AttributeCommonInfo::AS_C23, "clang::os_returns_not_retained"},
};
struct ParsedAttrInfoOSReturnsNotRetained final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoOSReturnsNotRetained() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_OSReturnsNotRetained,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/OSReturnsNotRetainedSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D) && !isa<ObjCMethodDecl>(D) && !isa<ObjCPropertyDecl>(D) && !isa<ParmVarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions, Objective-C methods, Objective-C properties, and parameters";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_property, /*IsSupported=*/LangOpts.ObjC));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable_is_parameter, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoOSReturnsNotRetained Instance;
};
const ParsedAttrInfoOSReturnsNotRetained ParsedAttrInfoOSReturnsNotRetained::Instance;
static constexpr ParsedAttrInfo::Spelling OSReturnsRetainedSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "os_returns_retained"},
  {AttributeCommonInfo::AS_CXX11, "clang::os_returns_retained"},
  {AttributeCommonInfo::AS_C23, "clang::os_returns_retained"},
};
struct ParsedAttrInfoOSReturnsRetained final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoOSReturnsRetained() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_OSReturnsRetained,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/OSReturnsRetainedSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D) && !isa<ObjCMethodDecl>(D) && !isa<ObjCPropertyDecl>(D) && !isa<ParmVarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions, Objective-C methods, Objective-C properties, and parameters";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_property, /*IsSupported=*/LangOpts.ObjC));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable_is_parameter, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoOSReturnsRetained Instance;
};
const ParsedAttrInfoOSReturnsRetained ParsedAttrInfoOSReturnsRetained::Instance;
static constexpr ParsedAttrInfo::Spelling OSReturnsRetainedOnNonZeroSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "os_returns_retained_on_non_zero"},
  {AttributeCommonInfo::AS_CXX11, "clang::os_returns_retained_on_non_zero"},
  {AttributeCommonInfo::AS_C23, "clang::os_returns_retained_on_non_zero"},
};
struct ParsedAttrInfoOSReturnsRetainedOnNonZero final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoOSReturnsRetainedOnNonZero() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_OSReturnsRetainedOnNonZero,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/OSReturnsRetainedOnNonZeroSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ParmVarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "parameters";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable_is_parameter, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoOSReturnsRetainedOnNonZero Instance;
};
const ParsedAttrInfoOSReturnsRetainedOnNonZero ParsedAttrInfoOSReturnsRetainedOnNonZero::Instance;
static constexpr ParsedAttrInfo::Spelling OSReturnsRetainedOnZeroSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "os_returns_retained_on_zero"},
  {AttributeCommonInfo::AS_CXX11, "clang::os_returns_retained_on_zero"},
  {AttributeCommonInfo::AS_C23, "clang::os_returns_retained_on_zero"},
};
struct ParsedAttrInfoOSReturnsRetainedOnZero final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoOSReturnsRetainedOnZero() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_OSReturnsRetainedOnZero,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/OSReturnsRetainedOnZeroSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ParmVarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "parameters";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable_is_parameter, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoOSReturnsRetainedOnZero Instance;
};
const ParsedAttrInfoOSReturnsRetainedOnZero ParsedAttrInfoOSReturnsRetainedOnZero::Instance;
static constexpr ParsedAttrInfo::Spelling ObjCBoxableSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "objc_boxable"},
  {AttributeCommonInfo::AS_CXX11, "clang::objc_boxable"},
  {AttributeCommonInfo::AS_C23, "clang::objc_boxable"},
};
struct ParsedAttrInfoObjCBoxable final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoObjCBoxable() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ObjCBoxable,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/ObjCBoxableSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<RecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "structs, unions, and classes";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoObjCBoxable Instance;
};
const ParsedAttrInfoObjCBoxable ParsedAttrInfoObjCBoxable::Instance;
static constexpr ParsedAttrInfo::Spelling ObjCBridgeSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "objc_bridge"},
  {AttributeCommonInfo::AS_CXX11, "clang::objc_bridge"},
  {AttributeCommonInfo::AS_C23, "clang::objc_bridge"},
};
static constexpr const char *ObjCBridgeArgNames[] = {
"BridgedType",};
struct ParsedAttrInfoObjCBridge final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoObjCBridge() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ObjCBridge,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/ObjCBridgeSpellings,
    /*ArgNames=*/ObjCBridgeArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<RecordDecl>(D) && !isa<TypedefNameDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "structs, unions, classes, and typedefs";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_type_alias, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoObjCBridge Instance;
};
const ParsedAttrInfoObjCBridge ParsedAttrInfoObjCBridge::Instance;
static constexpr ParsedAttrInfo::Spelling ObjCBridgeMutableSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "objc_bridge_mutable"},
  {AttributeCommonInfo::AS_CXX11, "clang::objc_bridge_mutable"},
  {AttributeCommonInfo::AS_C23, "clang::objc_bridge_mutable"},
};
static constexpr const char *ObjCBridgeMutableArgNames[] = {
"BridgedType",};
struct ParsedAttrInfoObjCBridgeMutable final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoObjCBridgeMutable() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ObjCBridgeMutable,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/ObjCBridgeMutableSpellings,
    /*ArgNames=*/ObjCBridgeMutableArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<RecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "structs, unions, and classes";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoObjCBridgeMutable Instance;
};
const ParsedAttrInfoObjCBridgeMutable ParsedAttrInfoObjCBridgeMutable::Instance;
static constexpr ParsedAttrInfo::Spelling ObjCBridgeRelatedSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "objc_bridge_related"},
  {AttributeCommonInfo::AS_CXX11, "clang::objc_bridge_related"},
  {AttributeCommonInfo::AS_C23, "clang::objc_bridge_related"},
};
static constexpr const char *ObjCBridgeRelatedArgNames[] = {
"RelatedClass","ClassMethod","InstanceMethod",};
struct ParsedAttrInfoObjCBridgeRelated final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoObjCBridgeRelated() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ObjCBridgeRelated,
    /*NumArgs=*/3,
    /*OptArgs=*/0,
    /*NumArgMembers=*/3,
    /*HasCustomParsing=*/1,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/ObjCBridgeRelatedSpellings,
    /*ArgNames=*/ObjCBridgeRelatedArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<RecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "structs, unions, and classes";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoObjCBridgeRelated Instance;
};
const ParsedAttrInfoObjCBridgeRelated ParsedAttrInfoObjCBridgeRelated::Instance;
static constexpr ParsedAttrInfo::Spelling ObjCClassStubSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "objc_class_stub"},
  {AttributeCommonInfo::AS_CXX11, "clang::objc_class_stub"},
  {AttributeCommonInfo::AS_C23, "clang::objc_class_stub"},
};
struct ParsedAttrInfoObjCClassStub final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoObjCClassStub() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ObjCClassStub,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/ObjCClassStubSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ObjCInterfaceDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "Objective-C interfaces";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return (LangOpts.ObjCRuntime.allowsClassStubs());
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_interface, /*IsSupported=*/LangOpts.ObjC));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) ObjCClassStubAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoObjCClassStub Instance;
};
const ParsedAttrInfoObjCClassStub ParsedAttrInfoObjCClassStub::Instance;
static constexpr ParsedAttrInfo::Spelling ObjCDesignatedInitializerSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "objc_designated_initializer"},
  {AttributeCommonInfo::AS_CXX11, "clang::objc_designated_initializer"},
  {AttributeCommonInfo::AS_C23, "clang::objc_designated_initializer"},
};
struct ParsedAttrInfoObjCDesignatedInitializer final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoObjCDesignatedInitializer() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ObjCDesignatedInitializer,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/ObjCDesignatedInitializerSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ObjCMethodDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "Objective-C methods";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoObjCDesignatedInitializer Instance;
};
const ParsedAttrInfoObjCDesignatedInitializer ParsedAttrInfoObjCDesignatedInitializer::Instance;
static constexpr ParsedAttrInfo::Spelling ObjCDirectSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "objc_direct"},
  {AttributeCommonInfo::AS_CXX11, "clang::objc_direct"},
  {AttributeCommonInfo::AS_C23, "clang::objc_direct"},
};
struct ParsedAttrInfoObjCDirect final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoObjCDirect() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ObjCDirect,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/ObjCDirectSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ObjCMethodDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "Objective-C methods";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.ObjC;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoObjCDirect Instance;
};
const ParsedAttrInfoObjCDirect ParsedAttrInfoObjCDirect::Instance;
static constexpr ParsedAttrInfo::Spelling ObjCDirectMembersSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "objc_direct_members"},
  {AttributeCommonInfo::AS_CXX11, "clang::objc_direct_members"},
  {AttributeCommonInfo::AS_C23, "clang::objc_direct_members"},
};
struct ParsedAttrInfoObjCDirectMembers final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoObjCDirectMembers() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ObjCDirectMembers,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/ObjCDirectMembersSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ObjCImplDecl>(D) && !isa<ObjCInterfaceDecl>(D) && !isa<ObjCCategoryDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "Objective-C implementation declarations, Objective-C interfaces, and Objective-C containers";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.ObjC;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_implementation, /*IsSupported=*/LangOpts.ObjC));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_interface, /*IsSupported=*/LangOpts.ObjC));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_category, /*IsSupported=*/LangOpts.ObjC));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoObjCDirectMembers Instance;
};
const ParsedAttrInfoObjCDirectMembers ParsedAttrInfoObjCDirectMembers::Instance;
static constexpr ParsedAttrInfo::Spelling ObjCExceptionSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "objc_exception"},
  {AttributeCommonInfo::AS_CXX11, "clang::objc_exception"},
  {AttributeCommonInfo::AS_C23, "clang::objc_exception"},
};
struct ParsedAttrInfoObjCException final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoObjCException() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ObjCException,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/ObjCExceptionSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ObjCInterfaceDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "Objective-C interfaces";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_interface, /*IsSupported=*/LangOpts.ObjC));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) ObjCExceptionAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoObjCException Instance;
};
const ParsedAttrInfoObjCException ParsedAttrInfoObjCException::Instance;
static constexpr ParsedAttrInfo::Spelling ObjCExplicitProtocolImplSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "objc_protocol_requires_explicit_implementation"},
  {AttributeCommonInfo::AS_CXX11, "clang::objc_protocol_requires_explicit_implementation"},
  {AttributeCommonInfo::AS_C23, "clang::objc_protocol_requires_explicit_implementation"},
};
struct ParsedAttrInfoObjCExplicitProtocolImpl final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoObjCExplicitProtocolImpl() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ObjCExplicitProtocolImpl,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/ObjCExplicitProtocolImplSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ObjCProtocolDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "Objective-C protocols";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_protocol, /*IsSupported=*/LangOpts.ObjC));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoObjCExplicitProtocolImpl Instance;
};
const ParsedAttrInfoObjCExplicitProtocolImpl ParsedAttrInfoObjCExplicitProtocolImpl::Instance;
static constexpr ParsedAttrInfo::Spelling ObjCExternallyRetainedSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "objc_externally_retained"},
  {AttributeCommonInfo::AS_CXX11, "clang::objc_externally_retained"},
  {AttributeCommonInfo::AS_C23, "clang::objc_externally_retained"},
};
struct ParsedAttrInfoObjCExternallyRetained final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoObjCExternallyRetained() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ObjCExternallyRetained,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/ObjCExternallyRetainedSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isNonParmVar(D) && !isa<FunctionDecl>(D) && !isa<BlockDecl>(D) && !isa<ObjCMethodDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "variables, functions, blocks, and Objective-C methods";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.ObjCAutoRefCount;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable_not_is_parameter, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_block, /*IsSupported=*/LangOpts.Blocks));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoObjCExternallyRetained Instance;
};
const ParsedAttrInfoObjCExternallyRetained ParsedAttrInfoObjCExternallyRetained::Instance;
static constexpr ParsedAttrInfo::Spelling ObjCGCSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "objc_gc"},
  {AttributeCommonInfo::AS_CXX11, "clang::objc_gc"},
  {AttributeCommonInfo::AS_C23, "clang::objc_gc"},
};
static constexpr const char *ObjCGCArgNames[] = {
"Kind",};
struct ParsedAttrInfoObjCGC final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoObjCGC() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ObjCGC,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/ObjCGCSpellings,
    /*ArgNames=*/ObjCGCArgNames) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoObjCGC Instance;
};
const ParsedAttrInfoObjCGC ParsedAttrInfoObjCGC::Instance;
static constexpr ParsedAttrInfo::Spelling ObjCIndependentClassSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "objc_independent_class"},
  {AttributeCommonInfo::AS_CXX11, "clang::objc_independent_class"},
  {AttributeCommonInfo::AS_C23, "clang::objc_independent_class"},
};
struct ParsedAttrInfoObjCIndependentClass final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoObjCIndependentClass() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ObjCIndependentClass,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/ObjCIndependentClassSpellings,
    /*ArgNames=*/{}) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoObjCIndependentClass Instance;
};
const ParsedAttrInfoObjCIndependentClass ParsedAttrInfoObjCIndependentClass::Instance;
static constexpr ParsedAttrInfo::Spelling ObjCInertUnsafeUnretainedSpellings[] = {
  {AttributeCommonInfo::AS_Keyword, "__unsafe_unretained"},
};
struct ParsedAttrInfoObjCInertUnsafeUnretained final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoObjCInertUnsafeUnretained() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ObjCInertUnsafeUnretained,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/ObjCInertUnsafeUnretainedSpellings,
    /*ArgNames=*/{}) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoObjCInertUnsafeUnretained Instance;
};
const ParsedAttrInfoObjCInertUnsafeUnretained ParsedAttrInfoObjCInertUnsafeUnretained::Instance;
static constexpr ParsedAttrInfo::Spelling ObjCKindOfSpellings[] = {
  {AttributeCommonInfo::AS_Keyword, "__kindof"},
};
struct ParsedAttrInfoObjCKindOf final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoObjCKindOf() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ObjCKindOf,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/ObjCKindOfSpellings,
    /*ArgNames=*/{}) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoObjCKindOf Instance;
};
const ParsedAttrInfoObjCKindOf ParsedAttrInfoObjCKindOf::Instance;
static constexpr ParsedAttrInfo::Spelling ObjCMethodFamilySpellings[] = {
  {AttributeCommonInfo::AS_GNU, "objc_method_family"},
  {AttributeCommonInfo::AS_CXX11, "clang::objc_method_family"},
  {AttributeCommonInfo::AS_C23, "clang::objc_method_family"},
};
static constexpr const char *ObjCMethodFamilyArgNames[] = {
"Family",};
struct ParsedAttrInfoObjCMethodFamily final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoObjCMethodFamily() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ObjCMethodFamily,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/ObjCMethodFamilySpellings,
    /*ArgNames=*/ObjCMethodFamilyArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ObjCMethodDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "Objective-C methods";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoObjCMethodFamily Instance;
};
const ParsedAttrInfoObjCMethodFamily ParsedAttrInfoObjCMethodFamily::Instance;
static constexpr ParsedAttrInfo::Spelling ObjCNSObjectSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "NSObject"},
  {AttributeCommonInfo::AS_CXX11, "clang::NSObject"},
  {AttributeCommonInfo::AS_C23, "clang::NSObject"},
};
struct ParsedAttrInfoObjCNSObject final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoObjCNSObject() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ObjCNSObject,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/ObjCNSObjectSpellings,
    /*ArgNames=*/{}) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoObjCNSObject Instance;
};
const ParsedAttrInfoObjCNSObject ParsedAttrInfoObjCNSObject::Instance;
static constexpr ParsedAttrInfo::Spelling ObjCNonLazyClassSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "objc_nonlazy_class"},
  {AttributeCommonInfo::AS_CXX11, "clang::objc_nonlazy_class"},
  {AttributeCommonInfo::AS_C23, "clang::objc_nonlazy_class"},
};
struct ParsedAttrInfoObjCNonLazyClass final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoObjCNonLazyClass() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ObjCNonLazyClass,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/ObjCNonLazyClassSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ObjCInterfaceDecl>(D) && !isa<ObjCImplDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "Objective-C interfaces and Objective-C implementation declarations";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.ObjC;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_interface, /*IsSupported=*/LangOpts.ObjC));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_implementation, /*IsSupported=*/LangOpts.ObjC));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) ObjCNonLazyClassAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoObjCNonLazyClass Instance;
};
const ParsedAttrInfoObjCNonLazyClass ParsedAttrInfoObjCNonLazyClass::Instance;
static constexpr ParsedAttrInfo::Spelling ObjCNonRuntimeProtocolSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "objc_non_runtime_protocol"},
  {AttributeCommonInfo::AS_CXX11, "clang::objc_non_runtime_protocol"},
  {AttributeCommonInfo::AS_C23, "clang::objc_non_runtime_protocol"},
};
struct ParsedAttrInfoObjCNonRuntimeProtocol final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoObjCNonRuntimeProtocol() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ObjCNonRuntimeProtocol,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/ObjCNonRuntimeProtocolSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ObjCProtocolDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "Objective-C protocols";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.ObjC;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_protocol, /*IsSupported=*/LangOpts.ObjC));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) ObjCNonRuntimeProtocolAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoObjCNonRuntimeProtocol Instance;
};
const ParsedAttrInfoObjCNonRuntimeProtocol ParsedAttrInfoObjCNonRuntimeProtocol::Instance;
static constexpr ParsedAttrInfo::Spelling ObjCOwnershipSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "objc_ownership"},
  {AttributeCommonInfo::AS_CXX11, "clang::objc_ownership"},
  {AttributeCommonInfo::AS_C23, "clang::objc_ownership"},
};
static constexpr const char *ObjCOwnershipArgNames[] = {
"Kind",};
struct ParsedAttrInfoObjCOwnership final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoObjCOwnership() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ObjCOwnership,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/ObjCOwnershipSpellings,
    /*ArgNames=*/ObjCOwnershipArgNames) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoObjCOwnership Instance;
};
const ParsedAttrInfoObjCOwnership ParsedAttrInfoObjCOwnership::Instance;
static constexpr ParsedAttrInfo::Spelling ObjCPreciseLifetimeSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "objc_precise_lifetime"},
  {AttributeCommonInfo::AS_CXX11, "clang::objc_precise_lifetime"},
  {AttributeCommonInfo::AS_C23, "clang::objc_precise_lifetime"},
};
struct ParsedAttrInfoObjCPreciseLifetime final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoObjCPreciseLifetime() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ObjCPreciseLifetime,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/ObjCPreciseLifetimeSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<VarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "variables";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoObjCPreciseLifetime Instance;
};
const ParsedAttrInfoObjCPreciseLifetime ParsedAttrInfoObjCPreciseLifetime::Instance;
static constexpr ParsedAttrInfo::Spelling ObjCRequiresPropertyDefsSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "objc_requires_property_definitions"},
  {AttributeCommonInfo::AS_CXX11, "clang::objc_requires_property_definitions"},
  {AttributeCommonInfo::AS_C23, "clang::objc_requires_property_definitions"},
};
struct ParsedAttrInfoObjCRequiresPropertyDefs final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoObjCRequiresPropertyDefs() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ObjCRequiresPropertyDefs,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/ObjCRequiresPropertyDefsSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ObjCInterfaceDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "Objective-C interfaces";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_interface, /*IsSupported=*/LangOpts.ObjC));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) ObjCRequiresPropertyDefsAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoObjCRequiresPropertyDefs Instance;
};
const ParsedAttrInfoObjCRequiresPropertyDefs ParsedAttrInfoObjCRequiresPropertyDefs::Instance;
static constexpr ParsedAttrInfo::Spelling ObjCRequiresSuperSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "objc_requires_super"},
  {AttributeCommonInfo::AS_CXX11, "clang::objc_requires_super"},
  {AttributeCommonInfo::AS_C23, "clang::objc_requires_super"},
};
struct ParsedAttrInfoObjCRequiresSuper final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoObjCRequiresSuper() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ObjCRequiresSuper,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/ObjCRequiresSuperSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ObjCMethodDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "Objective-C methods";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoObjCRequiresSuper Instance;
};
const ParsedAttrInfoObjCRequiresSuper ParsedAttrInfoObjCRequiresSuper::Instance;
static constexpr ParsedAttrInfo::Spelling ObjCReturnsInnerPointerSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "objc_returns_inner_pointer"},
  {AttributeCommonInfo::AS_CXX11, "clang::objc_returns_inner_pointer"},
  {AttributeCommonInfo::AS_C23, "clang::objc_returns_inner_pointer"},
};
struct ParsedAttrInfoObjCReturnsInnerPointer final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoObjCReturnsInnerPointer() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ObjCReturnsInnerPointer,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/ObjCReturnsInnerPointerSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ObjCMethodDecl>(D) && !isa<ObjCPropertyDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "Objective-C methods and Objective-C properties";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_property, /*IsSupported=*/LangOpts.ObjC));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoObjCReturnsInnerPointer Instance;
};
const ParsedAttrInfoObjCReturnsInnerPointer ParsedAttrInfoObjCReturnsInnerPointer::Instance;
static constexpr ParsedAttrInfo::Spelling ObjCRootClassSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "objc_root_class"},
  {AttributeCommonInfo::AS_CXX11, "clang::objc_root_class"},
  {AttributeCommonInfo::AS_C23, "clang::objc_root_class"},
};
struct ParsedAttrInfoObjCRootClass final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoObjCRootClass() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ObjCRootClass,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/ObjCRootClassSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ObjCInterfaceDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "Objective-C interfaces";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_interface, /*IsSupported=*/LangOpts.ObjC));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) ObjCRootClassAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoObjCRootClass Instance;
};
const ParsedAttrInfoObjCRootClass ParsedAttrInfoObjCRootClass::Instance;
static constexpr ParsedAttrInfo::Spelling ObjCRuntimeNameSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "objc_runtime_name"},
  {AttributeCommonInfo::AS_CXX11, "clang::objc_runtime_name"},
  {AttributeCommonInfo::AS_C23, "clang::objc_runtime_name"},
};
static constexpr const char *ObjCRuntimeNameArgNames[] = {
"MetadataName",};
struct ParsedAttrInfoObjCRuntimeName final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoObjCRuntimeName() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ObjCRuntimeName,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/ObjCRuntimeNameSpellings,
    /*ArgNames=*/ObjCRuntimeNameArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ObjCInterfaceDecl>(D) && !isa<ObjCProtocolDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "Objective-C interfaces and Objective-C protocols";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_interface, /*IsSupported=*/LangOpts.ObjC));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_protocol, /*IsSupported=*/LangOpts.ObjC));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoObjCRuntimeName Instance;
};
const ParsedAttrInfoObjCRuntimeName ParsedAttrInfoObjCRuntimeName::Instance;
static constexpr ParsedAttrInfo::Spelling ObjCRuntimeVisibleSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "objc_runtime_visible"},
  {AttributeCommonInfo::AS_CXX11, "clang::objc_runtime_visible"},
  {AttributeCommonInfo::AS_C23, "clang::objc_runtime_visible"},
};
struct ParsedAttrInfoObjCRuntimeVisible final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoObjCRuntimeVisible() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ObjCRuntimeVisible,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/ObjCRuntimeVisibleSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ObjCInterfaceDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "Objective-C interfaces";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_interface, /*IsSupported=*/LangOpts.ObjC));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) ObjCRuntimeVisibleAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoObjCRuntimeVisible Instance;
};
const ParsedAttrInfoObjCRuntimeVisible ParsedAttrInfoObjCRuntimeVisible::Instance;
static constexpr ParsedAttrInfo::Spelling ObjCSubclassingRestrictedSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "objc_subclassing_restricted"},
  {AttributeCommonInfo::AS_CXX11, "clang::objc_subclassing_restricted"},
  {AttributeCommonInfo::AS_C23, "clang::objc_subclassing_restricted"},
};
struct ParsedAttrInfoObjCSubclassingRestricted final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoObjCSubclassingRestricted() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ObjCSubclassingRestricted,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/ObjCSubclassingRestrictedSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ObjCInterfaceDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "Objective-C interfaces";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_interface, /*IsSupported=*/LangOpts.ObjC));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) ObjCSubclassingRestrictedAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoObjCSubclassingRestricted Instance;
};
const ParsedAttrInfoObjCSubclassingRestricted ParsedAttrInfoObjCSubclassingRestricted::Instance;
struct ParsedAttrInfoOpenACCRoutineAnnot final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoOpenACCRoutineAnnot() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_OpenACCRoutineAnnot,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/{},
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoOpenACCRoutineAnnot Instance;
};
const ParsedAttrInfoOpenACCRoutineAnnot ParsedAttrInfoOpenACCRoutineAnnot::Instance;
static constexpr ParsedAttrInfo::Spelling OpenCLAccessSpellings[] = {
  {AttributeCommonInfo::AS_Keyword, "__read_only"},
  {AttributeCommonInfo::AS_Keyword, "read_only"},
  {AttributeCommonInfo::AS_Keyword, "__write_only"},
  {AttributeCommonInfo::AS_Keyword, "write_only"},
  {AttributeCommonInfo::AS_Keyword, "__read_write"},
  {AttributeCommonInfo::AS_Keyword, "read_write"},
};
struct ParsedAttrInfoOpenCLAccess final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoOpenCLAccess() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_OpenCLAccess,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/OpenCLAccessSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ParmVarDecl>(D) && !isa<TypedefNameDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "parameters and typedefs";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

unsigned spellingIndexToSemanticSpelling(const ParsedAttr &Attr) const override {
  enum Spelling {
    Keyword_read_only = 0,
    Keyword_write_only = 2,
    Keyword_read_write = 4,
  SpellingNotCalculated = 15

  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return Keyword_read_only;
    case 1: return Keyword_read_only;
    case 2: return Keyword_write_only;
    case 3: return Keyword_write_only;
    case 4: return Keyword_read_write;
    case 5: return Keyword_read_write;
  }
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoOpenCLAccess Instance;
};
const ParsedAttrInfoOpenCLAccess ParsedAttrInfoOpenCLAccess::Instance;
static constexpr ParsedAttrInfo::Spelling OpenCLConstantAddressSpaceSpellings[] = {
  {AttributeCommonInfo::AS_Keyword, "__constant"},
  {AttributeCommonInfo::AS_Keyword, "constant"},
  {AttributeCommonInfo::AS_GNU, "opencl_constant"},
  {AttributeCommonInfo::AS_CXX11, "clang::opencl_constant"},
  {AttributeCommonInfo::AS_C23, "clang::opencl_constant"},
};
struct ParsedAttrInfoOpenCLConstantAddressSpace final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoOpenCLConstantAddressSpace() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_OpenCLConstantAddressSpace,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/OpenCLConstantAddressSpaceSpellings,
    /*ArgNames=*/{}) {}
unsigned spellingIndexToSemanticSpelling(const ParsedAttr &Attr) const override {
  enum Spelling {
    Keyword_constant = 0,
    GNU_opencl_constant = 2,
    CXX11_clang_opencl_constant = 3,
    C23_clang_opencl_constant = 4,
  SpellingNotCalculated = 15

  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return Keyword_constant;
    case 1: return Keyword_constant;
    case 2: return GNU_opencl_constant;
    case 3: return CXX11_clang_opencl_constant;
    case 4: return C23_clang_opencl_constant;
  }
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoOpenCLConstantAddressSpace Instance;
};
const ParsedAttrInfoOpenCLConstantAddressSpace ParsedAttrInfoOpenCLConstantAddressSpace::Instance;
static constexpr ParsedAttrInfo::Spelling OpenCLGenericAddressSpaceSpellings[] = {
  {AttributeCommonInfo::AS_Keyword, "__generic"},
  {AttributeCommonInfo::AS_Keyword, "generic"},
  {AttributeCommonInfo::AS_GNU, "opencl_generic"},
  {AttributeCommonInfo::AS_CXX11, "clang::opencl_generic"},
  {AttributeCommonInfo::AS_C23, "clang::opencl_generic"},
};
struct ParsedAttrInfoOpenCLGenericAddressSpace final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoOpenCLGenericAddressSpace() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_OpenCLGenericAddressSpace,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/OpenCLGenericAddressSpaceSpellings,
    /*ArgNames=*/{}) {}
unsigned spellingIndexToSemanticSpelling(const ParsedAttr &Attr) const override {
  enum Spelling {
    Keyword_generic = 0,
    GNU_opencl_generic = 2,
    CXX11_clang_opencl_generic = 3,
    C23_clang_opencl_generic = 4,
  SpellingNotCalculated = 15

  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return Keyword_generic;
    case 1: return Keyword_generic;
    case 2: return GNU_opencl_generic;
    case 3: return CXX11_clang_opencl_generic;
    case 4: return C23_clang_opencl_generic;
  }
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoOpenCLGenericAddressSpace Instance;
};
const ParsedAttrInfoOpenCLGenericAddressSpace ParsedAttrInfoOpenCLGenericAddressSpace::Instance;
static constexpr ParsedAttrInfo::Spelling OpenCLGlobalAddressSpaceSpellings[] = {
  {AttributeCommonInfo::AS_Keyword, "__global"},
  {AttributeCommonInfo::AS_Keyword, "global"},
  {AttributeCommonInfo::AS_GNU, "opencl_global"},
  {AttributeCommonInfo::AS_CXX11, "clang::opencl_global"},
  {AttributeCommonInfo::AS_C23, "clang::opencl_global"},
};
struct ParsedAttrInfoOpenCLGlobalAddressSpace final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoOpenCLGlobalAddressSpace() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_OpenCLGlobalAddressSpace,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/OpenCLGlobalAddressSpaceSpellings,
    /*ArgNames=*/{}) {}
unsigned spellingIndexToSemanticSpelling(const ParsedAttr &Attr) const override {
  enum Spelling {
    Keyword_global = 0,
    GNU_opencl_global = 2,
    CXX11_clang_opencl_global = 3,
    C23_clang_opencl_global = 4,
  SpellingNotCalculated = 15

  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return Keyword_global;
    case 1: return Keyword_global;
    case 2: return GNU_opencl_global;
    case 3: return CXX11_clang_opencl_global;
    case 4: return C23_clang_opencl_global;
  }
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoOpenCLGlobalAddressSpace Instance;
};
const ParsedAttrInfoOpenCLGlobalAddressSpace ParsedAttrInfoOpenCLGlobalAddressSpace::Instance;
static constexpr ParsedAttrInfo::Spelling OpenCLGlobalDeviceAddressSpaceSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "opencl_global_device"},
  {AttributeCommonInfo::AS_CXX11, "clang::opencl_global_device"},
  {AttributeCommonInfo::AS_C23, "clang::opencl_global_device"},
};
struct ParsedAttrInfoOpenCLGlobalDeviceAddressSpace final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoOpenCLGlobalDeviceAddressSpace() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_OpenCLGlobalDeviceAddressSpace,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/OpenCLGlobalDeviceAddressSpaceSpellings,
    /*ArgNames=*/{}) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoOpenCLGlobalDeviceAddressSpace Instance;
};
const ParsedAttrInfoOpenCLGlobalDeviceAddressSpace ParsedAttrInfoOpenCLGlobalDeviceAddressSpace::Instance;
static constexpr ParsedAttrInfo::Spelling OpenCLGlobalHostAddressSpaceSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "opencl_global_host"},
  {AttributeCommonInfo::AS_CXX11, "clang::opencl_global_host"},
  {AttributeCommonInfo::AS_C23, "clang::opencl_global_host"},
};
struct ParsedAttrInfoOpenCLGlobalHostAddressSpace final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoOpenCLGlobalHostAddressSpace() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_OpenCLGlobalHostAddressSpace,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/OpenCLGlobalHostAddressSpaceSpellings,
    /*ArgNames=*/{}) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoOpenCLGlobalHostAddressSpace Instance;
};
const ParsedAttrInfoOpenCLGlobalHostAddressSpace ParsedAttrInfoOpenCLGlobalHostAddressSpace::Instance;
static constexpr ParsedAttrInfo::Spelling OpenCLIntelReqdSubGroupSizeSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "intel_reqd_sub_group_size"},
};
static constexpr const char *OpenCLIntelReqdSubGroupSizeArgNames[] = {
"SubGroupSize",};
struct ParsedAttrInfoOpenCLIntelReqdSubGroupSize final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoOpenCLIntelReqdSubGroupSize() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_OpenCLIntelReqdSubGroupSize,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/OpenCLIntelReqdSubGroupSizeSpellings,
    /*ArgNames=*/OpenCLIntelReqdSubGroupSizeArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoOpenCLIntelReqdSubGroupSize Instance;
};
const ParsedAttrInfoOpenCLIntelReqdSubGroupSize ParsedAttrInfoOpenCLIntelReqdSubGroupSize::Instance;
static constexpr ParsedAttrInfo::Spelling OpenCLKernelSpellings[] = {
  {AttributeCommonInfo::AS_Keyword, "__kernel"},
  {AttributeCommonInfo::AS_Keyword, "kernel"},
};
struct ParsedAttrInfoOpenCLKernel final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoOpenCLKernel() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_OpenCLKernel,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/OpenCLKernelSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) OpenCLKernelAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoOpenCLKernel Instance;
};
const ParsedAttrInfoOpenCLKernel ParsedAttrInfoOpenCLKernel::Instance;
static constexpr ParsedAttrInfo::Spelling OpenCLLocalAddressSpaceSpellings[] = {
  {AttributeCommonInfo::AS_Keyword, "__local"},
  {AttributeCommonInfo::AS_Keyword, "local"},
  {AttributeCommonInfo::AS_GNU, "opencl_local"},
  {AttributeCommonInfo::AS_CXX11, "clang::opencl_local"},
  {AttributeCommonInfo::AS_C23, "clang::opencl_local"},
};
struct ParsedAttrInfoOpenCLLocalAddressSpace final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoOpenCLLocalAddressSpace() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_OpenCLLocalAddressSpace,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/OpenCLLocalAddressSpaceSpellings,
    /*ArgNames=*/{}) {}
unsigned spellingIndexToSemanticSpelling(const ParsedAttr &Attr) const override {
  enum Spelling {
    Keyword_local = 0,
    GNU_opencl_local = 2,
    CXX11_clang_opencl_local = 3,
    C23_clang_opencl_local = 4,
  SpellingNotCalculated = 15

  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return Keyword_local;
    case 1: return Keyword_local;
    case 2: return GNU_opencl_local;
    case 3: return CXX11_clang_opencl_local;
    case 4: return C23_clang_opencl_local;
  }
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoOpenCLLocalAddressSpace Instance;
};
const ParsedAttrInfoOpenCLLocalAddressSpace ParsedAttrInfoOpenCLLocalAddressSpace::Instance;
static constexpr ParsedAttrInfo::Spelling OpenCLNoSVMSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "nosvm"},
};
struct ParsedAttrInfoOpenCLNoSVM final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoOpenCLNoSVM() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_OpenCLNoSVM,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/OpenCLNoSVMSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<VarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "variables";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.OpenCL;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoOpenCLNoSVM Instance;
};
const ParsedAttrInfoOpenCLNoSVM ParsedAttrInfoOpenCLNoSVM::Instance;
static constexpr ParsedAttrInfo::Spelling OpenCLPrivateAddressSpaceSpellings[] = {
  {AttributeCommonInfo::AS_Keyword, "__private"},
  {AttributeCommonInfo::AS_Keyword, "private"},
  {AttributeCommonInfo::AS_GNU, "opencl_private"},
  {AttributeCommonInfo::AS_CXX11, "clang::opencl_private"},
  {AttributeCommonInfo::AS_C23, "clang::opencl_private"},
};
struct ParsedAttrInfoOpenCLPrivateAddressSpace final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoOpenCLPrivateAddressSpace() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_OpenCLPrivateAddressSpace,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/OpenCLPrivateAddressSpaceSpellings,
    /*ArgNames=*/{}) {}
unsigned spellingIndexToSemanticSpelling(const ParsedAttr &Attr) const override {
  enum Spelling {
    Keyword_private = 0,
    GNU_opencl_private = 2,
    CXX11_clang_opencl_private = 3,
    C23_clang_opencl_private = 4,
  SpellingNotCalculated = 15

  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return Keyword_private;
    case 1: return Keyword_private;
    case 2: return GNU_opencl_private;
    case 3: return CXX11_clang_opencl_private;
    case 4: return C23_clang_opencl_private;
  }
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoOpenCLPrivateAddressSpace Instance;
};
const ParsedAttrInfoOpenCLPrivateAddressSpace ParsedAttrInfoOpenCLPrivateAddressSpace::Instance;
static constexpr ParsedAttrInfo::Spelling OpenCLUnrollHintSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "opencl_unroll_hint"},
};
static constexpr const char *OpenCLUnrollHintArgNames[] = {
"UnrollHint",};
struct ParsedAttrInfoOpenCLUnrollHint final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoOpenCLUnrollHint() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_OpenCLUnrollHint,
    /*NumArgs=*/0,
    /*OptArgs=*/1,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/1,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/OpenCLUnrollHintSpellings,
    /*ArgNames=*/OpenCLUnrollHintArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &AL, const Decl *D) const override {
  S.Diag(AL.getLoc(), diag::err_attribute_invalid_on_decl)
    << AL << AL.isRegularKeywordAttribute() << D->getLocation();
  return false;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &Attr, const Stmt *St) const override {
  if (!isa<ForStmt>(St) && !isa<CXXForRangeStmt>(St) && !isa<WhileStmt>(St) && !isa<DoStmt>(St)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "'for', 'while', and 'do' statements";
    return false;
  }
  return true;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoOpenCLUnrollHint Instance;
};
const ParsedAttrInfoOpenCLUnrollHint ParsedAttrInfoOpenCLUnrollHint::Instance;
static constexpr ParsedAttrInfo::Spelling OptimizeNoneSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "optnone"},
  {AttributeCommonInfo::AS_CXX11, "clang::optnone"},
  {AttributeCommonInfo::AS_C23, "clang::optnone"},
};
struct ParsedAttrInfoOptimizeNone final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoOptimizeNone() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_OptimizeNone,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/OptimizeNoneSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D) && !isa<ObjCMethodDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions and Objective-C methods";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoOptimizeNone Instance;
};
const ParsedAttrInfoOptimizeNone ParsedAttrInfoOptimizeNone::Instance;
static constexpr ParsedAttrInfo::Spelling OverloadableSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "overloadable"},
  {AttributeCommonInfo::AS_CXX11, "clang::overloadable"},
  {AttributeCommonInfo::AS_C23, "clang::overloadable"},
};
struct ParsedAttrInfoOverloadable final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoOverloadable() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Overloadable,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/OverloadableSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) OverloadableAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoOverloadable Instance;
};
const ParsedAttrInfoOverloadable ParsedAttrInfoOverloadable::Instance;
static constexpr ParsedAttrInfo::Spelling OwnerSpellings[] = {
  {AttributeCommonInfo::AS_CXX11, "gsl::Owner"},
};
static constexpr const char *OwnerArgNames[] = {
"DerefType",};
struct ParsedAttrInfoOwner final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoOwner() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Owner,
    /*NumArgs=*/0,
    /*OptArgs=*/1,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/OwnerSpellings,
    /*ArgNames=*/OwnerArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isStruct(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "structs";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

  using ParsedAttrInfo::diagMutualExclusion;

  bool diagMutualExclusion(Sema &S, const ParsedAttr &AL, const Decl *D) const override {
    if (const auto *A = D->getAttr<PointerAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    return true;
  }

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record_not_is_union, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoOwner Instance;
};
const ParsedAttrInfoOwner ParsedAttrInfoOwner::Instance;
static constexpr ParsedAttrInfo::Spelling OwnershipSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "ownership_holds"},
  {AttributeCommonInfo::AS_CXX11, "clang::ownership_holds"},
  {AttributeCommonInfo::AS_C23, "clang::ownership_holds"},
  {AttributeCommonInfo::AS_GNU, "ownership_returns"},
  {AttributeCommonInfo::AS_CXX11, "clang::ownership_returns"},
  {AttributeCommonInfo::AS_C23, "clang::ownership_returns"},
  {AttributeCommonInfo::AS_GNU, "ownership_takes"},
  {AttributeCommonInfo::AS_CXX11, "clang::ownership_takes"},
  {AttributeCommonInfo::AS_C23, "clang::ownership_takes"},
};
static constexpr const char *OwnershipArgNames[] = {
"Module","Args...",};
struct ParsedAttrInfoOwnership final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoOwnership() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Ownership,
    /*NumArgs=*/1,
    /*OptArgs=*/15,
    /*NumArgMembers=*/2,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/OwnershipSpellings,
    /*ArgNames=*/OwnershipArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isHasFunctionProto(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "non-K&R-style functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

unsigned spellingIndexToSemanticSpelling(const ParsedAttr &Attr) const override {
  enum Spelling {
    GNU_ownership_holds = 0,
    CXX11_clang_ownership_holds = 1,
    C23_clang_ownership_holds = 2,
    GNU_ownership_returns = 3,
    CXX11_clang_ownership_returns = 4,
    C23_clang_ownership_returns = 5,
    GNU_ownership_takes = 6,
    CXX11_clang_ownership_takes = 7,
    C23_clang_ownership_takes = 8,
  SpellingNotCalculated = 15

  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return GNU_ownership_holds;
    case 1: return CXX11_clang_ownership_holds;
    case 2: return C23_clang_ownership_holds;
    case 3: return GNU_ownership_returns;
    case 4: return CXX11_clang_ownership_returns;
    case 5: return C23_clang_ownership_returns;
    case 6: return GNU_ownership_takes;
    case 7: return CXX11_clang_ownership_takes;
    case 8: return C23_clang_ownership_takes;
  }
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoOwnership Instance;
};
const ParsedAttrInfoOwnership ParsedAttrInfoOwnership::Instance;
static constexpr ParsedAttrInfo::Spelling PackedSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "packed"},
  {AttributeCommonInfo::AS_CXX11, "gnu::packed"},
  {AttributeCommonInfo::AS_C23, "gnu::packed"},
};
struct ParsedAttrInfoPacked final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoPacked() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Packed,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/PackedSpellings,
    /*ArgNames=*/{}) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoPacked Instance;
};
const ParsedAttrInfoPacked ParsedAttrInfoPacked::Instance;
static constexpr ParsedAttrInfo::Spelling ParamTypestateSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "param_typestate"},
  {AttributeCommonInfo::AS_CXX11, "clang::param_typestate"},
};
static constexpr const char *ParamTypestateArgNames[] = {
"ParamState",};
struct ParsedAttrInfoParamTypestate final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoParamTypestate() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ParamTypestate,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/ParamTypestateSpellings,
    /*ArgNames=*/ParamTypestateArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ParmVarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "parameters";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable_is_parameter, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoParamTypestate Instance;
};
const ParsedAttrInfoParamTypestate ParsedAttrInfoParamTypestate::Instance;
static constexpr ParsedAttrInfo::Spelling PascalSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "pascal"},
  {AttributeCommonInfo::AS_CXX11, "clang::pascal"},
  {AttributeCommonInfo::AS_C23, "clang::pascal"},
  {AttributeCommonInfo::AS_Keyword, "__pascal"},
  {AttributeCommonInfo::AS_Keyword, "_pascal"},
};
struct ParsedAttrInfoPascal final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoPascal() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Pascal,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/PascalSpellings,
    /*ArgNames=*/{}) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoPascal Instance;
};
const ParsedAttrInfoPascal ParsedAttrInfoPascal::Instance;
static constexpr ParsedAttrInfo::Spelling PassObjectSizeSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "pass_object_size"},
  {AttributeCommonInfo::AS_CXX11, "clang::pass_object_size"},
  {AttributeCommonInfo::AS_C23, "clang::pass_object_size"},
  {AttributeCommonInfo::AS_GNU, "pass_dynamic_object_size"},
  {AttributeCommonInfo::AS_CXX11, "clang::pass_dynamic_object_size"},
  {AttributeCommonInfo::AS_C23, "clang::pass_dynamic_object_size"},
};
static constexpr const char *PassObjectSizeArgNames[] = {
"Type",};
struct ParsedAttrInfoPassObjectSize final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoPassObjectSize() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_PassObjectSize,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/PassObjectSizeSpellings,
    /*ArgNames=*/PassObjectSizeArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ParmVarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "parameters";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

unsigned spellingIndexToSemanticSpelling(const ParsedAttr &Attr) const override {
  enum Spelling {
    GNU_pass_object_size = 0,
    CXX11_clang_pass_object_size = 1,
    C23_clang_pass_object_size = 2,
    GNU_pass_dynamic_object_size = 3,
    CXX11_clang_pass_dynamic_object_size = 4,
    C23_clang_pass_dynamic_object_size = 5,
  SpellingNotCalculated = 15

  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return GNU_pass_object_size;
    case 1: return CXX11_clang_pass_object_size;
    case 2: return C23_clang_pass_object_size;
    case 3: return GNU_pass_dynamic_object_size;
    case 4: return CXX11_clang_pass_dynamic_object_size;
    case 5: return C23_clang_pass_dynamic_object_size;
  }
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable_is_parameter, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoPassObjectSize Instance;
};
const ParsedAttrInfoPassObjectSize ParsedAttrInfoPassObjectSize::Instance;
static constexpr ParsedAttrInfo::Spelling PatchableFunctionEntrySpellings[] = {
  {AttributeCommonInfo::AS_GNU, "patchable_function_entry"},
  {AttributeCommonInfo::AS_CXX11, "gnu::patchable_function_entry"},
  {AttributeCommonInfo::AS_C23, "gnu::patchable_function_entry"},
};
static constexpr const char *PatchableFunctionEntryArgNames[] = {
"Count","Offset","Section",};
struct ParsedAttrInfoPatchableFunctionEntry final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoPatchableFunctionEntry() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_PatchableFunctionEntry,
    /*NumArgs=*/1,
    /*OptArgs=*/2,
    /*NumArgMembers=*/3,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/PatchableFunctionEntrySpellings,
    /*ArgNames=*/PatchableFunctionEntryArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D) && !isa<ObjCMethodDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions and Objective-C methods";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && (T.getArch() == llvm::Triple::aarch64 || T.getArch() == llvm::Triple::aarch64_be || T.getArch() == llvm::Triple::loongarch32 || T.getArch() == llvm::Triple::loongarch64 || T.getArch() == llvm::Triple::riscv32 || T.getArch() == llvm::Triple::riscv64 || T.getArch() == llvm::Triple::x86 || T.getArch() == llvm::Triple::x86_64 || T.getArch() == llvm::Triple::ppc || T.getArch() == llvm::Triple::ppc64);
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoPatchableFunctionEntry Instance;
};
const ParsedAttrInfoPatchableFunctionEntry ParsedAttrInfoPatchableFunctionEntry::Instance;
static constexpr ParsedAttrInfo::Spelling PcsSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "pcs"},
  {AttributeCommonInfo::AS_CXX11, "gnu::pcs"},
  {AttributeCommonInfo::AS_C23, "gnu::pcs"},
};
static constexpr const char *PcsArgNames[] = {
"PCS",};
struct ParsedAttrInfoPcs final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoPcs() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Pcs,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/PcsSpellings,
    /*ArgNames=*/PcsArgNames) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoPcs Instance;
};
const ParsedAttrInfoPcs ParsedAttrInfoPcs::Instance;
static constexpr ParsedAttrInfo::Spelling PointerSpellings[] = {
  {AttributeCommonInfo::AS_CXX11, "gsl::Pointer"},
};
static constexpr const char *PointerArgNames[] = {
"DerefType",};
struct ParsedAttrInfoPointer final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoPointer() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Pointer,
    /*NumArgs=*/0,
    /*OptArgs=*/1,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/PointerSpellings,
    /*ArgNames=*/PointerArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isStruct(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "structs";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

  using ParsedAttrInfo::diagMutualExclusion;

  bool diagMutualExclusion(Sema &S, const ParsedAttr &AL, const Decl *D) const override {
    if (const auto *A = D->getAttr<OwnerAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    return true;
  }

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record_not_is_union, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoPointer Instance;
};
const ParsedAttrInfoPointer ParsedAttrInfoPointer::Instance;
static constexpr ParsedAttrInfo::Spelling PointerAuthSpellings[] = {
  {AttributeCommonInfo::AS_Keyword, "__ptrauth"},
};
static constexpr const char *PointerAuthArgNames[] = {
"Key","AddressDiscriminated","ExtraDiscriminator",};
struct ParsedAttrInfoPointerAuth final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoPointerAuth() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_PointerAuth,
    /*NumArgs=*/1,
    /*OptArgs=*/2,
    /*NumArgMembers=*/3,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/PointerAuthSpellings,
    /*ArgNames=*/PointerAuthArgNames) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoPointerAuth Instance;
};
const ParsedAttrInfoPointerAuth ParsedAttrInfoPointerAuth::Instance;
static constexpr const char *PragmaClangBSSSectionArgNames[] = {
"Name",};
struct ParsedAttrInfoPragmaClangBSSSection final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoPragmaClangBSSSection() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_PragmaClangBSSSection,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/{},
    /*ArgNames=*/PragmaClangBSSSectionArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isGlobalVar(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "global variables";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoPragmaClangBSSSection Instance;
};
const ParsedAttrInfoPragmaClangBSSSection ParsedAttrInfoPragmaClangBSSSection::Instance;
static constexpr const char *PragmaClangDataSectionArgNames[] = {
"Name",};
struct ParsedAttrInfoPragmaClangDataSection final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoPragmaClangDataSection() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_PragmaClangDataSection,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/{},
    /*ArgNames=*/PragmaClangDataSectionArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isGlobalVar(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "global variables";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoPragmaClangDataSection Instance;
};
const ParsedAttrInfoPragmaClangDataSection ParsedAttrInfoPragmaClangDataSection::Instance;
static constexpr const char *PragmaClangRelroSectionArgNames[] = {
"Name",};
struct ParsedAttrInfoPragmaClangRelroSection final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoPragmaClangRelroSection() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_PragmaClangRelroSection,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/{},
    /*ArgNames=*/PragmaClangRelroSectionArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isGlobalVar(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "global variables";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoPragmaClangRelroSection Instance;
};
const ParsedAttrInfoPragmaClangRelroSection ParsedAttrInfoPragmaClangRelroSection::Instance;
static constexpr const char *PragmaClangRodataSectionArgNames[] = {
"Name",};
struct ParsedAttrInfoPragmaClangRodataSection final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoPragmaClangRodataSection() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_PragmaClangRodataSection,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/{},
    /*ArgNames=*/PragmaClangRodataSectionArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isGlobalVar(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "global variables";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoPragmaClangRodataSection Instance;
};
const ParsedAttrInfoPragmaClangRodataSection ParsedAttrInfoPragmaClangRodataSection::Instance;
static constexpr const char *PragmaClangTextSectionArgNames[] = {
"Name",};
struct ParsedAttrInfoPragmaClangTextSection final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoPragmaClangTextSection() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_PragmaClangTextSection,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/{},
    /*ArgNames=*/PragmaClangTextSectionArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoPragmaClangTextSection Instance;
};
const ParsedAttrInfoPragmaClangTextSection ParsedAttrInfoPragmaClangTextSection::Instance;
static constexpr ParsedAttrInfo::Spelling PreferredNameSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "preferred_name"},
  {AttributeCommonInfo::AS_CXX11, "clang::preferred_name"},
};
static constexpr const char *PreferredNameArgNames[] = {
"TypedefType",};
struct ParsedAttrInfoPreferredName final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoPreferredName() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_PreferredName,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/PreferredNameSpellings,
    /*ArgNames=*/PreferredNameArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isClassTmpl(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "class templates";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoPreferredName Instance;
};
const ParsedAttrInfoPreferredName ParsedAttrInfoPreferredName::Instance;
static constexpr ParsedAttrInfo::Spelling PreferredTypeSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "preferred_type"},
  {AttributeCommonInfo::AS_CXX11, "clang::preferred_type"},
  {AttributeCommonInfo::AS_C23, "clang::preferred_type"},
};
static constexpr const char *PreferredTypeArgNames[] = {
"Type",};
struct ParsedAttrInfoPreferredType final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoPreferredType() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_PreferredType,
    /*NumArgs=*/0,
    /*OptArgs=*/1,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/PreferredTypeSpellings,
    /*ArgNames=*/PreferredTypeArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isBitField(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "bit-field data members";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoPreferredType Instance;
};
const ParsedAttrInfoPreferredType ParsedAttrInfoPreferredType::Instance;
static constexpr ParsedAttrInfo::Spelling PreserveAllSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "preserve_all"},
  {AttributeCommonInfo::AS_CXX11, "clang::preserve_all"},
  {AttributeCommonInfo::AS_C23, "clang::preserve_all"},
};
struct ParsedAttrInfoPreserveAll final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoPreserveAll() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_PreserveAll,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/PreserveAllSpellings,
    /*ArgNames=*/{}) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoPreserveAll Instance;
};
const ParsedAttrInfoPreserveAll ParsedAttrInfoPreserveAll::Instance;
static constexpr ParsedAttrInfo::Spelling PreserveMostSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "preserve_most"},
  {AttributeCommonInfo::AS_CXX11, "clang::preserve_most"},
  {AttributeCommonInfo::AS_C23, "clang::preserve_most"},
};
struct ParsedAttrInfoPreserveMost final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoPreserveMost() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_PreserveMost,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/PreserveMostSpellings,
    /*ArgNames=*/{}) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoPreserveMost Instance;
};
const ParsedAttrInfoPreserveMost ParsedAttrInfoPreserveMost::Instance;
static constexpr ParsedAttrInfo::Spelling PreserveNoneSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "preserve_none"},
  {AttributeCommonInfo::AS_CXX11, "clang::preserve_none"},
  {AttributeCommonInfo::AS_C23, "clang::preserve_none"},
};
struct ParsedAttrInfoPreserveNone final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoPreserveNone() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_PreserveNone,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/PreserveNoneSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isFunctionLike(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions and function pointers";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && (T.getArch() == llvm::Triple::aarch64 || T.getArch() == llvm::Triple::aarch64_be || T.getArch() == llvm::Triple::aarch64_32 || T.getArch() == llvm::Triple::x86 || T.getArch() == llvm::Triple::x86_64);
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_hasType_functionType, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoPreserveNone Instance;
};
const ParsedAttrInfoPreserveNone ParsedAttrInfoPreserveNone::Instance;
static constexpr ParsedAttrInfo::Spelling PtGuardedBySpellings[] = {
  {AttributeCommonInfo::AS_GNU, "pt_guarded_by"},
};
static constexpr const char *PtGuardedByArgNames[] = {
"Arg",};
struct ParsedAttrInfoPtGuardedBy final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoPtGuardedBy() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_PtGuardedBy,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/PtGuardedBySpellings,
    /*ArgNames=*/PtGuardedByArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FieldDecl>(D) && !isSharedVar(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "non-static data members and global variables";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool isParamExpr(size_t N) const override {
  return (N == 0) || false;
}

static const ParsedAttrInfoPtGuardedBy Instance;
};
const ParsedAttrInfoPtGuardedBy ParsedAttrInfoPtGuardedBy::Instance;
static constexpr ParsedAttrInfo::Spelling PtGuardedVarSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "pt_guarded_var"},
  {AttributeCommonInfo::AS_CXX11, "clang::pt_guarded_var"},
};
struct ParsedAttrInfoPtGuardedVar final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoPtGuardedVar() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_PtGuardedVar,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/PtGuardedVarSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FieldDecl>(D) && !isSharedVar(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "non-static data members and global variables";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoPtGuardedVar Instance;
};
const ParsedAttrInfoPtGuardedVar ParsedAttrInfoPtGuardedVar::Instance;
static constexpr ParsedAttrInfo::Spelling Ptr32Spellings[] = {
  {AttributeCommonInfo::AS_Keyword, "__ptr32"},
};
struct ParsedAttrInfoPtr32 final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoPtr32() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Ptr32,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/Ptr32Spellings,
    /*ArgNames=*/{}) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoPtr32 Instance;
};
const ParsedAttrInfoPtr32 ParsedAttrInfoPtr32::Instance;
static constexpr ParsedAttrInfo::Spelling Ptr64Spellings[] = {
  {AttributeCommonInfo::AS_Keyword, "__ptr64"},
};
struct ParsedAttrInfoPtr64 final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoPtr64() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Ptr64,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/Ptr64Spellings,
    /*ArgNames=*/{}) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoPtr64 Instance;
};
const ParsedAttrInfoPtr64 ParsedAttrInfoPtr64::Instance;
static constexpr ParsedAttrInfo::Spelling PureSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "pure"},
  {AttributeCommonInfo::AS_CXX11, "gnu::pure"},
  {AttributeCommonInfo::AS_C23, "gnu::pure"},
};
struct ParsedAttrInfoPure final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoPure() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Pure,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/PureSpellings,
    /*ArgNames=*/{}) {}
AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) PureAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoPure Instance;
};
const ParsedAttrInfoPure ParsedAttrInfoPure::Instance;
static constexpr ParsedAttrInfo::Spelling RISCVRVVVectorBitsSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "riscv_rvv_vector_bits"},
};
static constexpr const char *RISCVRVVVectorBitsArgNames[] = {
"NumBits",};
struct ParsedAttrInfoRISCVRVVVectorBits final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoRISCVRVVVectorBits() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_RISCVRVVVectorBits,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/RISCVRVVVectorBitsSpellings,
    /*ArgNames=*/RISCVRVVVectorBitsArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<TypedefNameDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "typedefs";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoRISCVRVVVectorBits Instance;
};
const ParsedAttrInfoRISCVRVVVectorBits ParsedAttrInfoRISCVRVVVectorBits::Instance;
static constexpr ParsedAttrInfo::Spelling RISCVVLSCCSpellings[] = {
  {AttributeCommonInfo::AS_CXX11, "riscv::vls_cc"},
  {AttributeCommonInfo::AS_C23, "riscv::vls_cc"},
  {AttributeCommonInfo::AS_GNU, "riscv_vls_cc"},
  {AttributeCommonInfo::AS_CXX11, "clang::riscv_vls_cc"},
  {AttributeCommonInfo::AS_C23, "clang::riscv_vls_cc"},
};
static constexpr const char *RISCVVLSCCArgNames[] = {
"VectorWidth",};
struct ParsedAttrInfoRISCVVLSCC final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoRISCVVLSCC() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_RISCVVLSCC,
    /*NumArgs=*/0,
    /*OptArgs=*/1,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/RISCVVLSCCSpellings,
    /*ArgNames=*/RISCVVLSCCArgNames) {}
bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && (T.getArch() == llvm::Triple::riscv32 || T.getArch() == llvm::Triple::riscv64);
}

unsigned spellingIndexToSemanticSpelling(const ParsedAttr &Attr) const override {
  enum Spelling {
    CXX11_riscv_vls_cc = 0,
    C23_riscv_vls_cc = 1,
    GNU_riscv_vls_cc = 2,
    CXX11_clang_riscv_vls_cc = 3,
    C23_clang_riscv_vls_cc = 4,
  SpellingNotCalculated = 15

  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return CXX11_riscv_vls_cc;
    case 1: return C23_riscv_vls_cc;
    case 2: return GNU_riscv_vls_cc;
    case 3: return CXX11_clang_riscv_vls_cc;
    case 4: return C23_clang_riscv_vls_cc;
  }
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoRISCVVLSCC Instance;
};
const ParsedAttrInfoRISCVVLSCC ParsedAttrInfoRISCVVLSCC::Instance;
static constexpr ParsedAttrInfo::Spelling RISCVVectorCCSpellings[] = {
  {AttributeCommonInfo::AS_CXX11, "riscv::vector_cc"},
  {AttributeCommonInfo::AS_C23, "riscv::vector_cc"},
  {AttributeCommonInfo::AS_GNU, "riscv_vector_cc"},
  {AttributeCommonInfo::AS_CXX11, "clang::riscv_vector_cc"},
  {AttributeCommonInfo::AS_C23, "clang::riscv_vector_cc"},
};
struct ParsedAttrInfoRISCVVectorCC final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoRISCVVectorCC() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_RISCVVectorCC,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/RISCVVectorCCSpellings,
    /*ArgNames=*/{}) {}
bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && (T.getArch() == llvm::Triple::riscv32 || T.getArch() == llvm::Triple::riscv64);
}

unsigned spellingIndexToSemanticSpelling(const ParsedAttr &Attr) const override {
  enum Spelling {
    CXX11_riscv_vector_cc = 0,
    C23_riscv_vector_cc = 1,
    GNU_riscv_vector_cc = 2,
    CXX11_clang_riscv_vector_cc = 3,
    C23_clang_riscv_vector_cc = 4,
  SpellingNotCalculated = 15

  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return CXX11_riscv_vector_cc;
    case 1: return C23_riscv_vector_cc;
    case 2: return GNU_riscv_vector_cc;
    case 3: return CXX11_clang_riscv_vector_cc;
    case 4: return C23_clang_riscv_vector_cc;
  }
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoRISCVVectorCC Instance;
};
const ParsedAttrInfoRISCVVectorCC ParsedAttrInfoRISCVVectorCC::Instance;
static constexpr ParsedAttrInfo::Spelling RandomizeLayoutSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "randomize_layout"},
  {AttributeCommonInfo::AS_CXX11, "gnu::randomize_layout"},
  {AttributeCommonInfo::AS_C23, "gnu::randomize_layout"},
};
struct ParsedAttrInfoRandomizeLayout final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoRandomizeLayout() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_RandomizeLayout,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/RandomizeLayoutSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<RecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "structs, unions, and classes";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

  using ParsedAttrInfo::diagMutualExclusion;

  bool diagMutualExclusion(Sema &S, const ParsedAttr &AL, const Decl *D) const override {
    if (const auto *A = D->getAttr<NoRandomizeLayoutAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    return true;
  }

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return (!LangOpts.CPlusPlus);
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoRandomizeLayout Instance;
};
const ParsedAttrInfoRandomizeLayout ParsedAttrInfoRandomizeLayout::Instance;
static constexpr ParsedAttrInfo::Spelling ReadOnlyPlacementSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "enforce_read_only_placement"},
  {AttributeCommonInfo::AS_CXX11, "clang::enforce_read_only_placement"},
  {AttributeCommonInfo::AS_C23, "clang::enforce_read_only_placement"},
};
struct ParsedAttrInfoReadOnlyPlacement final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoReadOnlyPlacement() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ReadOnlyPlacement,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/ReadOnlyPlacementSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<RecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "structs, unions, and classes";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoReadOnlyPlacement Instance;
};
const ParsedAttrInfoReadOnlyPlacement ParsedAttrInfoReadOnlyPlacement::Instance;
static constexpr ParsedAttrInfo::Spelling RegCallSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "regcall"},
  {AttributeCommonInfo::AS_CXX11, "gnu::regcall"},
  {AttributeCommonInfo::AS_C23, "gnu::regcall"},
  {AttributeCommonInfo::AS_Keyword, "__regcall"},
};
struct ParsedAttrInfoRegCall final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoRegCall() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_RegCall,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/RegCallSpellings,
    /*ArgNames=*/{}) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoRegCall Instance;
};
const ParsedAttrInfoRegCall ParsedAttrInfoRegCall::Instance;
static constexpr ParsedAttrInfo::Spelling RegparmSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "regparm"},
  {AttributeCommonInfo::AS_CXX11, "gnu::regparm"},
  {AttributeCommonInfo::AS_C23, "gnu::regparm"},
};
static constexpr const char *RegparmArgNames[] = {
"NumParams",};
struct ParsedAttrInfoRegparm final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoRegparm() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Regparm,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/RegparmSpellings,
    /*ArgNames=*/RegparmArgNames) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoRegparm Instance;
};
const ParsedAttrInfoRegparm ParsedAttrInfoRegparm::Instance;
static constexpr ParsedAttrInfo::Spelling ReinitializesSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "reinitializes"},
  {AttributeCommonInfo::AS_CXX11, "clang::reinitializes"},
};
struct ParsedAttrInfoReinitializes final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoReinitializes() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Reinitializes,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/ReinitializesSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isNonStaticNonConstCXXMethod(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "non-static non-const member functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) ReinitializesAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoReinitializes Instance;
};
const ParsedAttrInfoReinitializes ParsedAttrInfoReinitializes::Instance;
static constexpr ParsedAttrInfo::Spelling ReleaseCapabilitySpellings[] = {
  {AttributeCommonInfo::AS_GNU, "release_capability"},
  {AttributeCommonInfo::AS_CXX11, "clang::release_capability"},
  {AttributeCommonInfo::AS_GNU, "release_shared_capability"},
  {AttributeCommonInfo::AS_CXX11, "clang::release_shared_capability"},
  {AttributeCommonInfo::AS_GNU, "release_generic_capability"},
  {AttributeCommonInfo::AS_CXX11, "clang::release_generic_capability"},
  {AttributeCommonInfo::AS_GNU, "unlock_function"},
  {AttributeCommonInfo::AS_CXX11, "clang::unlock_function"},
};
static constexpr const char *ReleaseCapabilityArgNames[] = {
"Args...",};
struct ParsedAttrInfoReleaseCapability final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoReleaseCapability() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ReleaseCapability,
    /*NumArgs=*/0,
    /*OptArgs=*/15,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/ReleaseCapabilitySpellings,
    /*ArgNames=*/ReleaseCapabilityArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D) && !isa<ParmVarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions and parameters";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

unsigned spellingIndexToSemanticSpelling(const ParsedAttr &Attr) const override {
  enum Spelling {
    GNU_release_capability = 0,
    CXX11_clang_release_capability = 1,
    GNU_release_shared_capability = 2,
    CXX11_clang_release_shared_capability = 3,
    GNU_release_generic_capability = 4,
    CXX11_clang_release_generic_capability = 5,
    GNU_unlock_function = 6,
    CXX11_clang_unlock_function = 7,
  SpellingNotCalculated = 15

  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return GNU_release_capability;
    case 1: return CXX11_clang_release_capability;
    case 2: return GNU_release_shared_capability;
    case 3: return CXX11_clang_release_shared_capability;
    case 4: return GNU_release_generic_capability;
    case 5: return CXX11_clang_release_generic_capability;
    case 6: return GNU_unlock_function;
    case 7: return CXX11_clang_unlock_function;
  }
}

bool isParamExpr(size_t N) const override {
  return (N == 0) || false;
}

static const ParsedAttrInfoReleaseCapability Instance;
};
const ParsedAttrInfoReleaseCapability ParsedAttrInfoReleaseCapability::Instance;
static constexpr ParsedAttrInfo::Spelling ReleaseHandleSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "release_handle"},
  {AttributeCommonInfo::AS_CXX11, "clang::release_handle"},
  {AttributeCommonInfo::AS_C23, "clang::release_handle"},
};
static constexpr const char *ReleaseHandleArgNames[] = {
"HandleType",};
struct ParsedAttrInfoReleaseHandle final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoReleaseHandle() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ReleaseHandle,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/ReleaseHandleSpellings,
    /*ArgNames=*/ReleaseHandleArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ParmVarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "parameters";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable_is_parameter, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoReleaseHandle Instance;
};
const ParsedAttrInfoReleaseHandle ParsedAttrInfoReleaseHandle::Instance;
static constexpr ParsedAttrInfo::Spelling ReqdWorkGroupSizeSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "reqd_work_group_size"},
};
static constexpr const char *ReqdWorkGroupSizeArgNames[] = {
"XDim","YDim","ZDim",};
struct ParsedAttrInfoReqdWorkGroupSize final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoReqdWorkGroupSize() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ReqdWorkGroupSize,
    /*NumArgs=*/3,
    /*OptArgs=*/0,
    /*NumArgMembers=*/3,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/ReqdWorkGroupSizeSpellings,
    /*ArgNames=*/ReqdWorkGroupSizeArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return (N == 0) || (N == 1) || (N == 2) || false;
}

static const ParsedAttrInfoReqdWorkGroupSize Instance;
};
const ParsedAttrInfoReqdWorkGroupSize ParsedAttrInfoReqdWorkGroupSize::Instance;
static constexpr ParsedAttrInfo::Spelling RequiresCapabilitySpellings[] = {
  {AttributeCommonInfo::AS_GNU, "requires_capability"},
  {AttributeCommonInfo::AS_CXX11, "clang::requires_capability"},
  {AttributeCommonInfo::AS_GNU, "exclusive_locks_required"},
  {AttributeCommonInfo::AS_CXX11, "clang::exclusive_locks_required"},
  {AttributeCommonInfo::AS_GNU, "requires_shared_capability"},
  {AttributeCommonInfo::AS_CXX11, "clang::requires_shared_capability"},
  {AttributeCommonInfo::AS_GNU, "shared_locks_required"},
  {AttributeCommonInfo::AS_CXX11, "clang::shared_locks_required"},
};
static constexpr const char *RequiresCapabilityArgNames[] = {
"Args...",};
struct ParsedAttrInfoRequiresCapability final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoRequiresCapability() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_RequiresCapability,
    /*NumArgs=*/0,
    /*OptArgs=*/15,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/RequiresCapabilitySpellings,
    /*ArgNames=*/RequiresCapabilityArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D) && !isa<ParmVarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions and parameters";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

unsigned spellingIndexToSemanticSpelling(const ParsedAttr &Attr) const override {
  enum Spelling {
    GNU_requires_capability = 0,
    CXX11_clang_requires_capability = 1,
    GNU_exclusive_locks_required = 2,
    CXX11_clang_exclusive_locks_required = 3,
    GNU_requires_shared_capability = 4,
    CXX11_clang_requires_shared_capability = 5,
    GNU_shared_locks_required = 6,
    CXX11_clang_shared_locks_required = 7,
  SpellingNotCalculated = 15

  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return GNU_requires_capability;
    case 1: return CXX11_clang_requires_capability;
    case 2: return GNU_exclusive_locks_required;
    case 3: return CXX11_clang_exclusive_locks_required;
    case 4: return GNU_requires_shared_capability;
    case 5: return CXX11_clang_requires_shared_capability;
    case 6: return GNU_shared_locks_required;
    case 7: return CXX11_clang_shared_locks_required;
  }
}

bool isParamExpr(size_t N) const override {
  return (N == 0) || false;
}

static const ParsedAttrInfoRequiresCapability Instance;
};
const ParsedAttrInfoRequiresCapability ParsedAttrInfoRequiresCapability::Instance;
static constexpr ParsedAttrInfo::Spelling RestrictSpellings[] = {
  {AttributeCommonInfo::AS_Declspec, "restrict"},
  {AttributeCommonInfo::AS_GNU, "malloc"},
  {AttributeCommonInfo::AS_CXX11, "gnu::malloc"},
  {AttributeCommonInfo::AS_C23, "gnu::malloc"},
};
static constexpr const char *RestrictArgNames[] = {
"Deallocator","DeallocatorPtrArgIndex",};
struct ParsedAttrInfoRestrict final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoRestrict() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Restrict,
    /*NumArgs=*/0,
    /*OptArgs=*/2,
    /*NumArgMembers=*/2,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/RestrictSpellings,
    /*ArgNames=*/RestrictArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

unsigned spellingIndexToSemanticSpelling(const ParsedAttr &Attr) const override {
  enum Spelling {
    Declspec_restrict = 0,
    GNU_malloc = 1,
    CXX11_gnu_malloc = 2,
    C23_gnu_malloc = 3,
  SpellingNotCalculated = 15

  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return Declspec_restrict;
    case 1: return GNU_malloc;
    case 2: return CXX11_gnu_malloc;
    case 3: return C23_gnu_malloc;
  }
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return (N == 0) || false;
}

static const ParsedAttrInfoRestrict Instance;
};
const ParsedAttrInfoRestrict ParsedAttrInfoRestrict::Instance;
static constexpr ParsedAttrInfo::Spelling RetainSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "retain"},
  {AttributeCommonInfo::AS_CXX11, "gnu::retain"},
  {AttributeCommonInfo::AS_C23, "gnu::retain"},
};
struct ParsedAttrInfoRetain final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoRetain() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Retain,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/RetainSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isNonLocalVar(D) && !isa<FunctionDecl>(D) && !isa<ObjCMethodDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "variables with non-local storage, functions, and Objective-C methods";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) RetainAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoRetain Instance;
};
const ParsedAttrInfoRetain ParsedAttrInfoRetain::Instance;
static constexpr ParsedAttrInfo::Spelling ReturnTypestateSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "return_typestate"},
  {AttributeCommonInfo::AS_CXX11, "clang::return_typestate"},
};
static constexpr const char *ReturnTypestateArgNames[] = {
"State",};
struct ParsedAttrInfoReturnTypestate final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoReturnTypestate() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ReturnTypestate,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/ReturnTypestateSpellings,
    /*ArgNames=*/ReturnTypestateArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D) && !isa<ParmVarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions and parameters";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable_is_parameter, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoReturnTypestate Instance;
};
const ParsedAttrInfoReturnTypestate ParsedAttrInfoReturnTypestate::Instance;
static constexpr ParsedAttrInfo::Spelling ReturnsNonNullSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "returns_nonnull"},
  {AttributeCommonInfo::AS_CXX11, "gnu::returns_nonnull"},
  {AttributeCommonInfo::AS_C23, "gnu::returns_nonnull"},
};
struct ParsedAttrInfoReturnsNonNull final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoReturnsNonNull() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ReturnsNonNull,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/ReturnsNonNullSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ObjCMethodDecl>(D) && !isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "Objective-C methods and functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoReturnsNonNull Instance;
};
const ParsedAttrInfoReturnsNonNull ParsedAttrInfoReturnsNonNull::Instance;
static constexpr ParsedAttrInfo::Spelling ReturnsTwiceSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "returns_twice"},
  {AttributeCommonInfo::AS_CXX11, "gnu::returns_twice"},
  {AttributeCommonInfo::AS_C23, "gnu::returns_twice"},
};
struct ParsedAttrInfoReturnsTwice final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoReturnsTwice() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ReturnsTwice,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/ReturnsTwiceSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) ReturnsTwiceAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoReturnsTwice Instance;
};
const ParsedAttrInfoReturnsTwice ParsedAttrInfoReturnsTwice::Instance;
static constexpr ParsedAttrInfo::Spelling SPtrSpellings[] = {
  {AttributeCommonInfo::AS_Keyword, "__sptr"},
};
struct ParsedAttrInfoSPtr final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoSPtr() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_SPtr,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/SPtrSpellings,
    /*ArgNames=*/{}) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoSPtr Instance;
};
const ParsedAttrInfoSPtr ParsedAttrInfoSPtr::Instance;
static constexpr ParsedAttrInfo::Spelling SYCLKernelSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "sycl_kernel"},
  {AttributeCommonInfo::AS_CXX11, "clang::sycl_kernel"},
  {AttributeCommonInfo::AS_C23, "clang::sycl_kernel"},
};
struct ParsedAttrInfoSYCLKernel final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoSYCLKernel() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_SYCLKernel,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/SYCLKernelSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isFunctionTmpl(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "function templates";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.SYCLIsDevice;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoSYCLKernel Instance;
};
const ParsedAttrInfoSYCLKernel ParsedAttrInfoSYCLKernel::Instance;
static constexpr ParsedAttrInfo::Spelling SYCLKernelEntryPointSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "sycl_kernel_entry_point"},
  {AttributeCommonInfo::AS_CXX11, "clang::sycl_kernel_entry_point"},
  {AttributeCommonInfo::AS_C23, "clang::sycl_kernel_entry_point"},
};
static constexpr const char *SYCLKernelEntryPointArgNames[] = {
"KernelName",};
struct ParsedAttrInfoSYCLKernelEntryPoint final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoSYCLKernelEntryPoint() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_SYCLKernelEntryPoint,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/SYCLKernelEntryPointSpellings,
    /*ArgNames=*/SYCLKernelEntryPointArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.SYCLIsHost || LangOpts.SYCLIsDevice;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoSYCLKernelEntryPoint Instance;
};
const ParsedAttrInfoSYCLKernelEntryPoint ParsedAttrInfoSYCLKernelEntryPoint::Instance;
static constexpr ParsedAttrInfo::Spelling SYCLSpecialClassSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "sycl_special_class"},
  {AttributeCommonInfo::AS_CXX11, "clang::sycl_special_class"},
  {AttributeCommonInfo::AS_C23, "clang::sycl_special_class"},
};
struct ParsedAttrInfoSYCLSpecialClass final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoSYCLSpecialClass() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_SYCLSpecialClass,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/SYCLSpecialClassSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<CXXRecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "classes";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.SYCLIsDevice;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoSYCLSpecialClass Instance;
};
const ParsedAttrInfoSYCLSpecialClass ParsedAttrInfoSYCLSpecialClass::Instance;
static constexpr ParsedAttrInfo::Spelling ScopedLockableSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "scoped_lockable"},
  {AttributeCommonInfo::AS_CXX11, "clang::scoped_lockable"},
};
struct ParsedAttrInfoScopedLockable final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoScopedLockable() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ScopedLockable,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/ScopedLockableSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<RecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "structs, unions, and classes";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) ScopedLockableAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoScopedLockable Instance;
};
const ParsedAttrInfoScopedLockable ParsedAttrInfoScopedLockable::Instance;
static constexpr ParsedAttrInfo::Spelling SectionSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "section"},
  {AttributeCommonInfo::AS_CXX11, "gnu::section"},
  {AttributeCommonInfo::AS_C23, "gnu::section"},
  {AttributeCommonInfo::AS_Declspec, "allocate"},
};
static constexpr const char *SectionArgNames[] = {
"Name",};
struct ParsedAttrInfoSection final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoSection() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Section,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/SectionSpellings,
    /*ArgNames=*/SectionArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D) && !isGlobalVar(D) && !isa<ObjCMethodDecl>(D) && !isa<ObjCPropertyDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions, global variables, Objective-C methods, and Objective-C properties";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

unsigned spellingIndexToSemanticSpelling(const ParsedAttr &Attr) const override {
  enum Spelling {
    GNU_section = 0,
    CXX11_gnu_section = 1,
    C23_gnu_section = 2,
    Declspec_allocate = 3,
  SpellingNotCalculated = 15

  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return GNU_section;
    case 1: return CXX11_gnu_section;
    case 2: return C23_gnu_section;
    case 3: return Declspec_allocate;
  }
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable_is_global, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_property, /*IsSupported=*/LangOpts.ObjC));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoSection Instance;
};
const ParsedAttrInfoSection ParsedAttrInfoSection::Instance;
static constexpr ParsedAttrInfo::Spelling SelectAnySpellings[] = {
  {AttributeCommonInfo::AS_Declspec, "selectany"},
  {AttributeCommonInfo::AS_GNU, "selectany"},
  {AttributeCommonInfo::AS_CXX11, "gnu::selectany"},
  {AttributeCommonInfo::AS_C23, "gnu::selectany"},
};
struct ParsedAttrInfoSelectAny final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoSelectAny() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_SelectAny,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/SelectAnySpellings,
    /*ArgNames=*/{}) {}
AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) SelectAnyAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoSelectAny Instance;
};
const ParsedAttrInfoSelectAny ParsedAttrInfoSelectAny::Instance;
static constexpr ParsedAttrInfo::Spelling SentinelSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "sentinel"},
  {AttributeCommonInfo::AS_CXX11, "gnu::sentinel"},
  {AttributeCommonInfo::AS_C23, "gnu::sentinel"},
};
static constexpr const char *SentinelArgNames[] = {
"Sentinel","NullPos",};
struct ParsedAttrInfoSentinel final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoSentinel() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Sentinel,
    /*NumArgs=*/0,
    /*OptArgs=*/2,
    /*NumArgMembers=*/2,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/SentinelSpellings,
    /*ArgNames=*/SentinelArgNames) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoSentinel Instance;
};
const ParsedAttrInfoSentinel ParsedAttrInfoSentinel::Instance;
static constexpr ParsedAttrInfo::Spelling SetTypestateSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "set_typestate"},
  {AttributeCommonInfo::AS_CXX11, "clang::set_typestate"},
};
static constexpr const char *SetTypestateArgNames[] = {
"NewState",};
struct ParsedAttrInfoSetTypestate final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoSetTypestate() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_SetTypestate,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/SetTypestateSpellings,
    /*ArgNames=*/SetTypestateArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<CXXMethodDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function_is_member, /*IsSupported=*/LangOpts.CPlusPlus));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoSetTypestate Instance;
};
const ParsedAttrInfoSetTypestate ParsedAttrInfoSetTypestate::Instance;
static constexpr ParsedAttrInfo::Spelling SizedBySpellings[] = {
  {AttributeCommonInfo::AS_GNU, "sized_by"},
  {AttributeCommonInfo::AS_CXX11, "clang::sized_by"},
  {AttributeCommonInfo::AS_C23, "clang::sized_by"},
};
static constexpr const char *SizedByArgNames[] = {
"Size","NestedLevel",};
struct ParsedAttrInfoSizedBy final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoSizedBy() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_SizedBy,
    /*NumArgs=*/1,
    /*OptArgs=*/1,
    /*NumArgMembers=*/2,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/SizedBySpellings,
    /*ArgNames=*/SizedByArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FieldDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "non-static data members";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return (!LangOpts.CPlusPlus);
}

bool isParamExpr(size_t N) const override {
  return (N == 0) || false;
}

static const ParsedAttrInfoSizedBy Instance;
};
const ParsedAttrInfoSizedBy ParsedAttrInfoSizedBy::Instance;
static constexpr ParsedAttrInfo::Spelling SizedByOrNullSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "sized_by_or_null"},
  {AttributeCommonInfo::AS_CXX11, "clang::sized_by_or_null"},
  {AttributeCommonInfo::AS_C23, "clang::sized_by_or_null"},
};
static constexpr const char *SizedByOrNullArgNames[] = {
"Size","NestedLevel",};
struct ParsedAttrInfoSizedByOrNull final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoSizedByOrNull() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_SizedByOrNull,
    /*NumArgs=*/1,
    /*OptArgs=*/1,
    /*NumArgMembers=*/2,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/SizedByOrNullSpellings,
    /*ArgNames=*/SizedByOrNullArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FieldDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "non-static data members";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return (!LangOpts.CPlusPlus);
}

bool isParamExpr(size_t N) const override {
  return (N == 0) || false;
}

static const ParsedAttrInfoSizedByOrNull Instance;
};
const ParsedAttrInfoSizedByOrNull ParsedAttrInfoSizedByOrNull::Instance;
static constexpr ParsedAttrInfo::Spelling SpeculativeLoadHardeningSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "speculative_load_hardening"},
  {AttributeCommonInfo::AS_CXX11, "clang::speculative_load_hardening"},
  {AttributeCommonInfo::AS_C23, "clang::speculative_load_hardening"},
};
struct ParsedAttrInfoSpeculativeLoadHardening final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoSpeculativeLoadHardening() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_SpeculativeLoadHardening,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/SpeculativeLoadHardeningSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D) && !isa<ObjCMethodDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions and Objective-C methods";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

  using ParsedAttrInfo::diagMutualExclusion;

  bool diagMutualExclusion(Sema &S, const ParsedAttr &AL, const Decl *D) const override {
    if (const auto *A = D->getAttr<NoSpeculativeLoadHardeningAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    return true;
  }

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) SpeculativeLoadHardeningAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoSpeculativeLoadHardening Instance;
};
const ParsedAttrInfoSpeculativeLoadHardening ParsedAttrInfoSpeculativeLoadHardening::Instance;
static constexpr ParsedAttrInfo::Spelling StandaloneDebugSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "standalone_debug"},
  {AttributeCommonInfo::AS_CXX11, "clang::standalone_debug"},
};
struct ParsedAttrInfoStandaloneDebug final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoStandaloneDebug() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_StandaloneDebug,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/StandaloneDebugSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<CXXRecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "classes";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.CPlusPlus;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) StandaloneDebugAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoStandaloneDebug Instance;
};
const ParsedAttrInfoStandaloneDebug ParsedAttrInfoStandaloneDebug::Instance;
static constexpr ParsedAttrInfo::Spelling StdCallSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "stdcall"},
  {AttributeCommonInfo::AS_CXX11, "gnu::stdcall"},
  {AttributeCommonInfo::AS_C23, "gnu::stdcall"},
  {AttributeCommonInfo::AS_Keyword, "__stdcall"},
  {AttributeCommonInfo::AS_Keyword, "_stdcall"},
};
struct ParsedAttrInfoStdCall final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoStdCall() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_StdCall,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/StdCallSpellings,
    /*ArgNames=*/{}) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoStdCall Instance;
};
const ParsedAttrInfoStdCall ParsedAttrInfoStdCall::Instance;
struct ParsedAttrInfoStrictFP final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoStrictFP() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_StrictFP,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/{},
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoStrictFP Instance;
};
const ParsedAttrInfoStrictFP ParsedAttrInfoStrictFP::Instance;
static constexpr ParsedAttrInfo::Spelling StrictGuardStackCheckSpellings[] = {
  {AttributeCommonInfo::AS_Declspec, "strict_gs_check"},
};
struct ParsedAttrInfoStrictGuardStackCheck final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoStrictGuardStackCheck() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_StrictGuardStackCheck,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/StrictGuardStackCheckSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) StrictGuardStackCheckAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoStrictGuardStackCheck Instance;
};
const ParsedAttrInfoStrictGuardStackCheck ParsedAttrInfoStrictGuardStackCheck::Instance;
static constexpr ParsedAttrInfo::Spelling SuppressSpellings[] = {
  {AttributeCommonInfo::AS_CXX11, "gsl::suppress"},
  {AttributeCommonInfo::AS_GNU, "suppress"},
  {AttributeCommonInfo::AS_CXX11, "clang::suppress"},
  {AttributeCommonInfo::AS_C23, "clang::suppress"},
};
static constexpr const char *SuppressArgNames[] = {
"DiagnosticIdentifiers...",};
struct ParsedAttrInfoSuppress final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoSuppress() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Suppress,
    /*NumArgs=*/0,
    /*OptArgs=*/15,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/1,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/SuppressSpellings,
    /*ArgNames=*/SuppressArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<VarDecl>(D) && !isa<FieldDecl>(D) && !isa<ObjCPropertyDecl>(D) && !isa<FunctionDecl>(D) && !isa<ObjCMethodDecl>(D) && !isa<RecordDecl>(D) && !isa<ObjCInterfaceDecl>(D) && !isa<ObjCImplementationDecl>(D) && !isa<NamespaceDecl>(D) && !isa<EmptyDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "variables, functions, structs, interfaces, and namespaces";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &Attr, const Stmt *St) const override {
  if (!isa<Stmt>(St)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "variables, functions, structs, interfaces, and namespaces";
    return false;
  }
  return true;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoSuppress Instance;
};
const ParsedAttrInfoSuppress ParsedAttrInfoSuppress::Instance;
static constexpr ParsedAttrInfo::Spelling SwiftAsyncSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "swift_async"},
  {AttributeCommonInfo::AS_CXX11, "clang::swift_async"},
  {AttributeCommonInfo::AS_C23, "clang::swift_async"},
};
static constexpr const char *SwiftAsyncArgNames[] = {
"Kind","CompletionHandlerIndex",};
struct ParsedAttrInfoSwiftAsync final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoSwiftAsync() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_SwiftAsync,
    /*NumArgs=*/1,
    /*OptArgs=*/1,
    /*NumArgMembers=*/2,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/SwiftAsyncSpellings,
    /*ArgNames=*/SwiftAsyncArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D) && !isa<ObjCMethodDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions and Objective-C methods";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoSwiftAsync Instance;
};
const ParsedAttrInfoSwiftAsync ParsedAttrInfoSwiftAsync::Instance;
static constexpr ParsedAttrInfo::Spelling SwiftAsyncCallSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "swiftasynccall"},
  {AttributeCommonInfo::AS_CXX11, "clang::swiftasynccall"},
  {AttributeCommonInfo::AS_C23, "clang::swiftasynccall"},
};
struct ParsedAttrInfoSwiftAsyncCall final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoSwiftAsyncCall() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_SwiftAsyncCall,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/SwiftAsyncCallSpellings,
    /*ArgNames=*/{}) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoSwiftAsyncCall Instance;
};
const ParsedAttrInfoSwiftAsyncCall ParsedAttrInfoSwiftAsyncCall::Instance;
static constexpr ParsedAttrInfo::Spelling SwiftAsyncContextSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "swift_async_context"},
  {AttributeCommonInfo::AS_CXX11, "clang::swift_async_context"},
  {AttributeCommonInfo::AS_C23, "clang::swift_async_context"},
};
struct ParsedAttrInfoSwiftAsyncContext final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoSwiftAsyncContext() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_SwiftAsyncContext,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/SwiftAsyncContextSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ParmVarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "parameters";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable_is_parameter, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoSwiftAsyncContext Instance;
};
const ParsedAttrInfoSwiftAsyncContext ParsedAttrInfoSwiftAsyncContext::Instance;
static constexpr ParsedAttrInfo::Spelling SwiftAsyncErrorSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "swift_async_error"},
  {AttributeCommonInfo::AS_CXX11, "clang::swift_async_error"},
  {AttributeCommonInfo::AS_C23, "clang::swift_async_error"},
};
static constexpr const char *SwiftAsyncErrorArgNames[] = {
"Convention","HandlerParamIdx",};
struct ParsedAttrInfoSwiftAsyncError final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoSwiftAsyncError() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_SwiftAsyncError,
    /*NumArgs=*/1,
    /*OptArgs=*/1,
    /*NumArgMembers=*/2,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/SwiftAsyncErrorSpellings,
    /*ArgNames=*/SwiftAsyncErrorArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D) && !isa<ObjCMethodDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions and Objective-C methods";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoSwiftAsyncError Instance;
};
const ParsedAttrInfoSwiftAsyncError ParsedAttrInfoSwiftAsyncError::Instance;
static constexpr ParsedAttrInfo::Spelling SwiftAsyncNameSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "swift_async_name"},
};
static constexpr const char *SwiftAsyncNameArgNames[] = {
"Name",};
struct ParsedAttrInfoSwiftAsyncName final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoSwiftAsyncName() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_SwiftAsyncName,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/SwiftAsyncNameSpellings,
    /*ArgNames=*/SwiftAsyncNameArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ObjCMethodDecl>(D) && !isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "Objective-C methods and functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoSwiftAsyncName Instance;
};
const ParsedAttrInfoSwiftAsyncName ParsedAttrInfoSwiftAsyncName::Instance;
static constexpr ParsedAttrInfo::Spelling SwiftAttrSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "swift_attr"},
};
static constexpr const char *SwiftAttrArgNames[] = {
"Attribute",};
struct ParsedAttrInfoSwiftAttr final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoSwiftAttr() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_SwiftAttr,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/SwiftAttrSpellings,
    /*ArgNames=*/SwiftAttrArgNames) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoSwiftAttr Instance;
};
const ParsedAttrInfoSwiftAttr ParsedAttrInfoSwiftAttr::Instance;
static constexpr ParsedAttrInfo::Spelling SwiftBridgeSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "swift_bridge"},
};
static constexpr const char *SwiftBridgeArgNames[] = {
"SwiftType",};
struct ParsedAttrInfoSwiftBridge final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoSwiftBridge() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_SwiftBridge,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/SwiftBridgeSpellings,
    /*ArgNames=*/SwiftBridgeArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<TagDecl>(D) && !isa<TypedefNameDecl>(D) && !isa<ObjCInterfaceDecl>(D) && !isa<ObjCProtocolDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "tag types, typedefs, Objective-C interfaces, and Objective-C protocols";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoSwiftBridge Instance;
};
const ParsedAttrInfoSwiftBridge ParsedAttrInfoSwiftBridge::Instance;
static constexpr ParsedAttrInfo::Spelling SwiftBridgedTypedefSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "swift_bridged_typedef"},
};
struct ParsedAttrInfoSwiftBridgedTypedef final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoSwiftBridgedTypedef() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_SwiftBridgedTypedef,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/SwiftBridgedTypedefSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<TypedefNameDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "typedefs";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_type_alias, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) SwiftBridgedTypedefAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoSwiftBridgedTypedef Instance;
};
const ParsedAttrInfoSwiftBridgedTypedef ParsedAttrInfoSwiftBridgedTypedef::Instance;
static constexpr ParsedAttrInfo::Spelling SwiftCallSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "swiftcall"},
  {AttributeCommonInfo::AS_CXX11, "clang::swiftcall"},
  {AttributeCommonInfo::AS_C23, "clang::swiftcall"},
};
struct ParsedAttrInfoSwiftCall final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoSwiftCall() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_SwiftCall,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/SwiftCallSpellings,
    /*ArgNames=*/{}) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoSwiftCall Instance;
};
const ParsedAttrInfoSwiftCall ParsedAttrInfoSwiftCall::Instance;
static constexpr ParsedAttrInfo::Spelling SwiftContextSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "swift_context"},
  {AttributeCommonInfo::AS_CXX11, "clang::swift_context"},
  {AttributeCommonInfo::AS_C23, "clang::swift_context"},
};
struct ParsedAttrInfoSwiftContext final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoSwiftContext() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_SwiftContext,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/SwiftContextSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ParmVarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "parameters";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable_is_parameter, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoSwiftContext Instance;
};
const ParsedAttrInfoSwiftContext ParsedAttrInfoSwiftContext::Instance;
static constexpr ParsedAttrInfo::Spelling SwiftErrorSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "swift_error"},
};
static constexpr const char *SwiftErrorArgNames[] = {
"Convention",};
struct ParsedAttrInfoSwiftError final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoSwiftError() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_SwiftError,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/SwiftErrorSpellings,
    /*ArgNames=*/SwiftErrorArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D) && !isa<ObjCMethodDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions and Objective-C methods";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoSwiftError Instance;
};
const ParsedAttrInfoSwiftError ParsedAttrInfoSwiftError::Instance;
static constexpr ParsedAttrInfo::Spelling SwiftErrorResultSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "swift_error_result"},
  {AttributeCommonInfo::AS_CXX11, "clang::swift_error_result"},
  {AttributeCommonInfo::AS_C23, "clang::swift_error_result"},
};
struct ParsedAttrInfoSwiftErrorResult final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoSwiftErrorResult() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_SwiftErrorResult,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/SwiftErrorResultSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ParmVarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "parameters";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable_is_parameter, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoSwiftErrorResult Instance;
};
const ParsedAttrInfoSwiftErrorResult ParsedAttrInfoSwiftErrorResult::Instance;
static constexpr ParsedAttrInfo::Spelling SwiftIndirectResultSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "swift_indirect_result"},
  {AttributeCommonInfo::AS_CXX11, "clang::swift_indirect_result"},
  {AttributeCommonInfo::AS_C23, "clang::swift_indirect_result"},
};
struct ParsedAttrInfoSwiftIndirectResult final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoSwiftIndirectResult() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_SwiftIndirectResult,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/SwiftIndirectResultSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ParmVarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "parameters";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable_is_parameter, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoSwiftIndirectResult Instance;
};
const ParsedAttrInfoSwiftIndirectResult ParsedAttrInfoSwiftIndirectResult::Instance;
static constexpr ParsedAttrInfo::Spelling SwiftNameSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "swift_name"},
};
static constexpr const char *SwiftNameArgNames[] = {
"Name",};
struct ParsedAttrInfoSwiftName final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoSwiftName() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_SwiftName,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/SwiftNameSpellings,
    /*ArgNames=*/SwiftNameArgNames) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoSwiftName Instance;
};
const ParsedAttrInfoSwiftName ParsedAttrInfoSwiftName::Instance;
static constexpr ParsedAttrInfo::Spelling SwiftNewTypeSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "swift_newtype"},
  {AttributeCommonInfo::AS_GNU, "swift_wrapper"},
};
static constexpr const char *SwiftNewTypeArgNames[] = {
"NewtypeKind",};
struct ParsedAttrInfoSwiftNewType final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoSwiftNewType() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_SwiftNewType,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/1,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/SwiftNewTypeSpellings,
    /*ArgNames=*/SwiftNewTypeArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<TypedefNameDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "typedefs";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

unsigned spellingIndexToSemanticSpelling(const ParsedAttr &Attr) const override {
  enum Spelling {
    GNU_swift_newtype = 0,
    GNU_swift_wrapper = 1,
  SpellingNotCalculated = 15

  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return GNU_swift_newtype;
    case 1: return GNU_swift_wrapper;
  }
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_type_alias, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoSwiftNewType Instance;
};
const ParsedAttrInfoSwiftNewType ParsedAttrInfoSwiftNewType::Instance;
static constexpr ParsedAttrInfo::Spelling SwiftObjCMembersSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "swift_objc_members"},
};
struct ParsedAttrInfoSwiftObjCMembers final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoSwiftObjCMembers() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_SwiftObjCMembers,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/SwiftObjCMembersSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ObjCInterfaceDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "Objective-C interfaces";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_interface, /*IsSupported=*/LangOpts.ObjC));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) SwiftObjCMembersAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoSwiftObjCMembers Instance;
};
const ParsedAttrInfoSwiftObjCMembers ParsedAttrInfoSwiftObjCMembers::Instance;
static constexpr ParsedAttrInfo::Spelling SwiftPrivateSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "swift_private"},
};
struct ParsedAttrInfoSwiftPrivate final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoSwiftPrivate() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_SwiftPrivate,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/SwiftPrivateSpellings,
    /*ArgNames=*/{}) {}
AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) SwiftPrivateAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoSwiftPrivate Instance;
};
const ParsedAttrInfoSwiftPrivate ParsedAttrInfoSwiftPrivate::Instance;
static constexpr ParsedAttrInfo::Spelling SysVABISpellings[] = {
  {AttributeCommonInfo::AS_GNU, "sysv_abi"},
  {AttributeCommonInfo::AS_CXX11, "gnu::sysv_abi"},
  {AttributeCommonInfo::AS_C23, "gnu::sysv_abi"},
};
struct ParsedAttrInfoSysVABI final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoSysVABI() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_SysVABI,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/SysVABISpellings,
    /*ArgNames=*/{}) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoSysVABI Instance;
};
const ParsedAttrInfoSysVABI ParsedAttrInfoSysVABI::Instance;
static constexpr ParsedAttrInfo::Spelling TLSModelSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "tls_model"},
  {AttributeCommonInfo::AS_CXX11, "gnu::tls_model"},
  {AttributeCommonInfo::AS_C23, "gnu::tls_model"},
};
static constexpr const char *TLSModelArgNames[] = {
"Model",};
struct ParsedAttrInfoTLSModel final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoTLSModel() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_TLSModel,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/TLSModelSpellings,
    /*ArgNames=*/TLSModelArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isTLSVar(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "thread-local variables";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable_is_thread_local, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoTLSModel Instance;
};
const ParsedAttrInfoTLSModel ParsedAttrInfoTLSModel::Instance;
static constexpr ParsedAttrInfo::Spelling TargetSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "target"},
  {AttributeCommonInfo::AS_CXX11, "gnu::target"},
  {AttributeCommonInfo::AS_C23, "gnu::target"},
};
static constexpr const char *TargetArgNames[] = {
"featuresStr",};
struct ParsedAttrInfoTarget final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoTarget() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Target,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/TargetSpellings,
    /*ArgNames=*/TargetArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

  using ParsedAttrInfo::diagMutualExclusion;

  bool diagMutualExclusion(Sema &S, const ParsedAttr &AL, const Decl *D) const override {
    if (const auto *A = D->getAttr<TargetClonesAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    if (const auto *A = D->getAttr<TargetVersionAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    if (const auto *A = D->getAttr<CPUDispatchAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    if (const auto *A = D->getAttr<CPUSpecificAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    return true;
  }

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoTarget Instance;
};
const ParsedAttrInfoTarget ParsedAttrInfoTarget::Instance;
static constexpr ParsedAttrInfo::Spelling TargetClonesSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "target_clones"},
  {AttributeCommonInfo::AS_CXX11, "gnu::target_clones"},
  {AttributeCommonInfo::AS_C23, "gnu::target_clones"},
};
static constexpr const char *TargetClonesArgNames[] = {
"featuresStrs...",};
struct ParsedAttrInfoTargetClones final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoTargetClones() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_TargetClones,
    /*NumArgs=*/0,
    /*OptArgs=*/15,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/TargetClonesSpellings,
    /*ArgNames=*/TargetClonesArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

  using ParsedAttrInfo::diagMutualExclusion;

  bool diagMutualExclusion(Sema &S, const ParsedAttr &AL, const Decl *D) const override {
    if (const auto *A = D->getAttr<TargetVersionAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    if (const auto *A = D->getAttr<TargetAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    if (const auto *A = D->getAttr<CPUDispatchAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    if (const auto *A = D->getAttr<CPUSpecificAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    return true;
  }

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoTargetClones Instance;
};
const ParsedAttrInfoTargetClones ParsedAttrInfoTargetClones::Instance;
static constexpr ParsedAttrInfo::Spelling TargetVersionSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "target_version"},
  {AttributeCommonInfo::AS_CXX11, "gnu::target_version"},
  {AttributeCommonInfo::AS_C23, "gnu::target_version"},
};
static constexpr const char *TargetVersionArgNames[] = {
"NamesStr",};
struct ParsedAttrInfoTargetVersion final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoTargetVersion() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_TargetVersion,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/TargetVersionSpellings,
    /*ArgNames=*/TargetVersionArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

  using ParsedAttrInfo::diagMutualExclusion;

  bool diagMutualExclusion(Sema &S, const ParsedAttr &AL, const Decl *D) const override {
    if (const auto *A = D->getAttr<TargetClonesAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    if (const auto *A = D->getAttr<TargetAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    if (const auto *A = D->getAttr<CPUDispatchAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    if (const auto *A = D->getAttr<CPUSpecificAttr>()) {
      S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible) << AL << A << (AL.isRegularKeywordAttribute() || A->isRegularKeywordAttribute());
      S.Diag(A->getLocation(), diag::note_conflicting_attribute);      
return false;
    }
    return true;
  }

bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && (T.getArch() == llvm::Triple::aarch64 || T.getArch() == llvm::Triple::aarch64_be || T.getArch() == llvm::Triple::aarch64_32 || T.getArch() == llvm::Triple::riscv32 || T.getArch() == llvm::Triple::riscv64);
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoTargetVersion Instance;
};
const ParsedAttrInfoTargetVersion ParsedAttrInfoTargetVersion::Instance;
static constexpr ParsedAttrInfo::Spelling TestTypestateSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "test_typestate"},
  {AttributeCommonInfo::AS_CXX11, "clang::test_typestate"},
};
static constexpr const char *TestTypestateArgNames[] = {
"TestState",};
struct ParsedAttrInfoTestTypestate final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoTestTypestate() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_TestTypestate,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/TestTypestateSpellings,
    /*ArgNames=*/TestTypestateArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<CXXMethodDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function_is_member, /*IsSupported=*/LangOpts.CPlusPlus));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoTestTypestate Instance;
};
const ParsedAttrInfoTestTypestate ParsedAttrInfoTestTypestate::Instance;
static constexpr ParsedAttrInfo::Spelling ThisCallSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "thiscall"},
  {AttributeCommonInfo::AS_CXX11, "gnu::thiscall"},
  {AttributeCommonInfo::AS_C23, "gnu::thiscall"},
  {AttributeCommonInfo::AS_Keyword, "__thiscall"},
  {AttributeCommonInfo::AS_Keyword, "_thiscall"},
};
struct ParsedAttrInfoThisCall final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoThisCall() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ThisCall,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/ThisCallSpellings,
    /*ArgNames=*/{}) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoThisCall Instance;
};
const ParsedAttrInfoThisCall ParsedAttrInfoThisCall::Instance;
static constexpr ParsedAttrInfo::Spelling ThreadSpellings[] = {
  {AttributeCommonInfo::AS_Declspec, "thread"},
};
struct ParsedAttrInfoThread final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoThread() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Thread,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/ThreadSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<VarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "variables";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.MicrosoftExt;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoThread Instance;
};
const ParsedAttrInfoThread ParsedAttrInfoThread::Instance;
static constexpr ParsedAttrInfo::Spelling TransparentUnionSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "transparent_union"},
  {AttributeCommonInfo::AS_CXX11, "gnu::transparent_union"},
  {AttributeCommonInfo::AS_C23, "gnu::transparent_union"},
};
struct ParsedAttrInfoTransparentUnion final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoTransparentUnion() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_TransparentUnion,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/TransparentUnionSpellings,
    /*ArgNames=*/{}) {}
bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return (!LangOpts.CPlusPlus);
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoTransparentUnion Instance;
};
const ParsedAttrInfoTransparentUnion ParsedAttrInfoTransparentUnion::Instance;
static constexpr ParsedAttrInfo::Spelling TrivialABISpellings[] = {
  {AttributeCommonInfo::AS_GNU, "trivial_abi"},
  {AttributeCommonInfo::AS_CXX11, "clang::trivial_abi"},
};
struct ParsedAttrInfoTrivialABI final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoTrivialABI() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_TrivialABI,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/TrivialABISpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<CXXRecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "classes";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.CPlusPlus;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) TrivialABIAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoTrivialABI Instance;
};
const ParsedAttrInfoTrivialABI ParsedAttrInfoTrivialABI::Instance;
static constexpr ParsedAttrInfo::Spelling TryAcquireCapabilitySpellings[] = {
  {AttributeCommonInfo::AS_GNU, "try_acquire_capability"},
  {AttributeCommonInfo::AS_CXX11, "clang::try_acquire_capability"},
  {AttributeCommonInfo::AS_GNU, "try_acquire_shared_capability"},
  {AttributeCommonInfo::AS_CXX11, "clang::try_acquire_shared_capability"},
  {AttributeCommonInfo::AS_GNU, "exclusive_trylock_function"},
  {AttributeCommonInfo::AS_GNU, "shared_trylock_function"},
};
static constexpr const char *TryAcquireCapabilityArgNames[] = {
"SuccessValue","Args...",};
struct ParsedAttrInfoTryAcquireCapability final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoTryAcquireCapability() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_TryAcquireCapability,
    /*NumArgs=*/1,
    /*OptArgs=*/15,
    /*NumArgMembers=*/2,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/TryAcquireCapabilitySpellings,
    /*ArgNames=*/TryAcquireCapabilityArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

unsigned spellingIndexToSemanticSpelling(const ParsedAttr &Attr) const override {
  enum Spelling {
    GNU_try_acquire_capability = 0,
    CXX11_clang_try_acquire_capability = 1,
    GNU_try_acquire_shared_capability = 2,
    CXX11_clang_try_acquire_shared_capability = 3,
    GNU_exclusive_trylock_function = 4,
    GNU_shared_trylock_function = 5,
  SpellingNotCalculated = 15

  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return GNU_try_acquire_capability;
    case 1: return CXX11_clang_try_acquire_capability;
    case 2: return GNU_try_acquire_shared_capability;
    case 3: return CXX11_clang_try_acquire_shared_capability;
    case 4: return GNU_exclusive_trylock_function;
    case 5: return GNU_shared_trylock_function;
  }
}

bool isParamExpr(size_t N) const override {
  return (N == 0) || (N == 1) || false;
}

static const ParsedAttrInfoTryAcquireCapability Instance;
};
const ParsedAttrInfoTryAcquireCapability ParsedAttrInfoTryAcquireCapability::Instance;
static constexpr ParsedAttrInfo::Spelling TypeNonNullSpellings[] = {
  {AttributeCommonInfo::AS_Keyword, "_Nonnull"},
};
struct ParsedAttrInfoTypeNonNull final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoTypeNonNull() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_TypeNonNull,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/TypeNonNullSpellings,
    /*ArgNames=*/{}) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoTypeNonNull Instance;
};
const ParsedAttrInfoTypeNonNull ParsedAttrInfoTypeNonNull::Instance;
static constexpr ParsedAttrInfo::Spelling TypeNullUnspecifiedSpellings[] = {
  {AttributeCommonInfo::AS_Keyword, "_Null_unspecified"},
};
struct ParsedAttrInfoTypeNullUnspecified final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoTypeNullUnspecified() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_TypeNullUnspecified,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/TypeNullUnspecifiedSpellings,
    /*ArgNames=*/{}) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoTypeNullUnspecified Instance;
};
const ParsedAttrInfoTypeNullUnspecified ParsedAttrInfoTypeNullUnspecified::Instance;
static constexpr ParsedAttrInfo::Spelling TypeNullableSpellings[] = {
  {AttributeCommonInfo::AS_Keyword, "_Nullable"},
};
struct ParsedAttrInfoTypeNullable final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoTypeNullable() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_TypeNullable,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/TypeNullableSpellings,
    /*ArgNames=*/{}) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoTypeNullable Instance;
};
const ParsedAttrInfoTypeNullable ParsedAttrInfoTypeNullable::Instance;
static constexpr ParsedAttrInfo::Spelling TypeNullableResultSpellings[] = {
  {AttributeCommonInfo::AS_Keyword, "_Nullable_result"},
};
struct ParsedAttrInfoTypeNullableResult final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoTypeNullableResult() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_TypeNullableResult,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/TypeNullableResultSpellings,
    /*ArgNames=*/{}) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoTypeNullableResult Instance;
};
const ParsedAttrInfoTypeNullableResult ParsedAttrInfoTypeNullableResult::Instance;
static constexpr ParsedAttrInfo::Spelling TypeTagForDatatypeSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "type_tag_for_datatype"},
  {AttributeCommonInfo::AS_CXX11, "clang::type_tag_for_datatype"},
  {AttributeCommonInfo::AS_C23, "clang::type_tag_for_datatype"},
};
static constexpr const char *TypeTagForDatatypeArgNames[] = {
"ArgumentKind","MatchingCType","LayoutCompatible","MustBeNull",};
struct ParsedAttrInfoTypeTagForDatatype final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoTypeTagForDatatype() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_TypeTagForDatatype,
    /*NumArgs=*/4,
    /*OptArgs=*/0,
    /*NumArgMembers=*/4,
    /*HasCustomParsing=*/1,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/TypeTagForDatatypeSpellings,
    /*ArgNames=*/TypeTagForDatatypeArgNames) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoTypeTagForDatatype Instance;
};
const ParsedAttrInfoTypeTagForDatatype ParsedAttrInfoTypeTagForDatatype::Instance;
static constexpr ParsedAttrInfo::Spelling TypeVisibilitySpellings[] = {
  {AttributeCommonInfo::AS_GNU, "type_visibility"},
  {AttributeCommonInfo::AS_CXX11, "clang::type_visibility"},
  {AttributeCommonInfo::AS_C23, "clang::type_visibility"},
};
static constexpr const char *TypeVisibilityArgNames[] = {
"Visibility",};
struct ParsedAttrInfoTypeVisibility final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoTypeVisibility() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_TypeVisibility,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/TypeVisibilitySpellings,
    /*ArgNames=*/TypeVisibilityArgNames) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoTypeVisibility Instance;
};
const ParsedAttrInfoTypeVisibility ParsedAttrInfoTypeVisibility::Instance;
static constexpr ParsedAttrInfo::Spelling UPtrSpellings[] = {
  {AttributeCommonInfo::AS_Keyword, "__uptr"},
};
struct ParsedAttrInfoUPtr final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoUPtr() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_UPtr,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/UPtrSpellings,
    /*ArgNames=*/{}) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoUPtr Instance;
};
const ParsedAttrInfoUPtr ParsedAttrInfoUPtr::Instance;
static constexpr ParsedAttrInfo::Spelling UnavailableSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "unavailable"},
  {AttributeCommonInfo::AS_CXX11, "clang::unavailable"},
  {AttributeCommonInfo::AS_C23, "clang::unavailable"},
};
static constexpr const char *UnavailableArgNames[] = {
"Message",};
struct ParsedAttrInfoUnavailable final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoUnavailable() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Unavailable,
    /*NumArgs=*/0,
    /*OptArgs=*/1,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/UnavailableSpellings,
    /*ArgNames=*/UnavailableArgNames) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoUnavailable Instance;
};
const ParsedAttrInfoUnavailable ParsedAttrInfoUnavailable::Instance;
static constexpr ParsedAttrInfo::Spelling UninitializedSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "uninitialized"},
  {AttributeCommonInfo::AS_CXX11, "clang::uninitialized"},
};
struct ParsedAttrInfoUninitialized final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoUninitialized() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Uninitialized,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/UninitializedSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isLocalVar(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "local variables";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable_is_local, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoUninitialized Instance;
};
const ParsedAttrInfoUninitialized ParsedAttrInfoUninitialized::Instance;
static constexpr ParsedAttrInfo::Spelling UnlikelySpellings[] = {
  {AttributeCommonInfo::AS_CXX11, "unlikely"},
  {AttributeCommonInfo::AS_C23, "clang::unlikely"},
};
struct ParsedAttrInfoUnlikely final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoUnlikely() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Unlikely,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/1,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/UnlikelySpellings,
    /*ArgNames=*/{}) {}
  using ParsedAttrInfo::diagMutualExclusion;

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoUnlikely Instance;
};
const ParsedAttrInfoUnlikely ParsedAttrInfoUnlikely::Instance;
static constexpr ParsedAttrInfo::Spelling UnsafeBufferUsageSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "unsafe_buffer_usage"},
  {AttributeCommonInfo::AS_CXX11, "clang::unsafe_buffer_usage"},
  {AttributeCommonInfo::AS_C23, "clang::unsafe_buffer_usage"},
};
struct ParsedAttrInfoUnsafeBufferUsage final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoUnsafeBufferUsage() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_UnsafeBufferUsage,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/UnsafeBufferUsageSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D) && !isa<FieldDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions and non-static data members";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_field, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoUnsafeBufferUsage Instance;
};
const ParsedAttrInfoUnsafeBufferUsage ParsedAttrInfoUnsafeBufferUsage::Instance;
static constexpr ParsedAttrInfo::Spelling UnusedSpellings[] = {
  {AttributeCommonInfo::AS_CXX11, "maybe_unused"},
  {AttributeCommonInfo::AS_GNU, "unused"},
  {AttributeCommonInfo::AS_CXX11, "gnu::unused"},
  {AttributeCommonInfo::AS_C23, "gnu::unused"},
  {AttributeCommonInfo::AS_C23, "maybe_unused"},
};
struct ParsedAttrInfoUnused final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoUnused() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Unused,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/UnusedSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<VarDecl>(D) && !isa<BindingDecl>(D) && !isa<ObjCIvarDecl>(D) && !isa<TypeDecl>(D) && !isa<EnumDecl>(D) && !isa<EnumConstantDecl>(D) && !isa<LabelDecl>(D) && !isa<FieldDecl>(D) && !isa<ObjCMethodDecl>(D) && !isFunctionLike(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "variables, value declarations, non-static data members, types, enums, enumerators, labels, non-static data members, Objective-C methods, functions, and function pointers";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

unsigned spellingIndexToSemanticSpelling(const ParsedAttr &Attr) const override {
  enum Spelling {
    CXX11_maybe_unused = 0,
    GNU_unused = 1,
    CXX11_gnu_unused = 2,
    C23_gnu_unused = 3,
    C23_maybe_unused = 4,
  SpellingNotCalculated = 15

  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return CXX11_maybe_unused;
    case 1: return GNU_unused;
    case 2: return CXX11_gnu_unused;
    case 3: return C23_gnu_unused;
    case 4: return C23_maybe_unused;
  }
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoUnused Instance;
};
const ParsedAttrInfoUnused ParsedAttrInfoUnused::Instance;
static constexpr ParsedAttrInfo::Spelling UseHandleSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "use_handle"},
  {AttributeCommonInfo::AS_CXX11, "clang::use_handle"},
  {AttributeCommonInfo::AS_C23, "clang::use_handle"},
};
static constexpr const char *UseHandleArgNames[] = {
"HandleType",};
struct ParsedAttrInfoUseHandle final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoUseHandle() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_UseHandle,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/UseHandleSpellings,
    /*ArgNames=*/UseHandleArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ParmVarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "parameters";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable_is_parameter, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoUseHandle Instance;
};
const ParsedAttrInfoUseHandle ParsedAttrInfoUseHandle::Instance;
static constexpr ParsedAttrInfo::Spelling UsedSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "used"},
  {AttributeCommonInfo::AS_CXX11, "gnu::used"},
  {AttributeCommonInfo::AS_C23, "gnu::used"},
};
struct ParsedAttrInfoUsed final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoUsed() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Used,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/UsedSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isNonLocalVar(D) && !isa<FunctionDecl>(D) && !isa<ObjCMethodDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "variables with non-local storage, functions, and Objective-C methods";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) UsedAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoUsed Instance;
};
const ParsedAttrInfoUsed ParsedAttrInfoUsed::Instance;
static constexpr ParsedAttrInfo::Spelling UsingIfExistsSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "using_if_exists"},
  {AttributeCommonInfo::AS_CXX11, "clang::using_if_exists"},
};
struct ParsedAttrInfoUsingIfExists final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoUsingIfExists() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_UsingIfExists,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/UsingIfExistsSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<UsingDecl>(D) && !isa<UnresolvedUsingTypenameDecl>(D) && !isa<UnresolvedUsingValueDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "named declarations, types, and value declarations";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoUsingIfExists Instance;
};
const ParsedAttrInfoUsingIfExists ParsedAttrInfoUsingIfExists::Instance;
static constexpr ParsedAttrInfo::Spelling UuidSpellings[] = {
  {AttributeCommonInfo::AS_Declspec, "uuid"},
  {AttributeCommonInfo::AS_Microsoft, "uuid"},
};
static constexpr const char *UuidArgNames[] = {
"Guid",};
struct ParsedAttrInfoUuid final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoUuid() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Uuid,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/UuidSpellings,
    /*ArgNames=*/UuidArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<RecordDecl>(D) && !isa<EnumDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "structs, unions, classes, and enums";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool acceptsLangOpts(const LangOptions &LangOpts) const override {
  return LangOpts.MicrosoftExt || LangOpts.Borland;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoUuid Instance;
};
const ParsedAttrInfoUuid ParsedAttrInfoUuid::Instance;
static constexpr ParsedAttrInfo::Spelling VTablePointerAuthenticationSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "ptrauth_vtable_pointer"},
  {AttributeCommonInfo::AS_CXX11, "clang::ptrauth_vtable_pointer"},
  {AttributeCommonInfo::AS_C23, "clang::ptrauth_vtable_pointer"},
};
static constexpr const char *VTablePointerAuthenticationArgNames[] = {
"Key","AddressDiscrimination","ExtraDiscrimination","CustomDiscriminationValue",};
struct ParsedAttrInfoVTablePointerAuthentication final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoVTablePointerAuthentication() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_VTablePointerAuthentication,
    /*NumArgs=*/3,
    /*OptArgs=*/1,
    /*NumArgMembers=*/4,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/VTablePointerAuthenticationSpellings,
    /*ArgNames=*/VTablePointerAuthenticationArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<CXXRecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "classes";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoVTablePointerAuthentication Instance;
};
const ParsedAttrInfoVTablePointerAuthentication ParsedAttrInfoVTablePointerAuthentication::Instance;
static constexpr ParsedAttrInfo::Spelling VecReturnSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "vecreturn"},
  {AttributeCommonInfo::AS_CXX11, "clang::vecreturn"},
};
struct ParsedAttrInfoVecReturn final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoVecReturn() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_VecReturn,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/VecReturnSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<CXXRecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "classes";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoVecReturn Instance;
};
const ParsedAttrInfoVecReturn ParsedAttrInfoVecReturn::Instance;
static constexpr ParsedAttrInfo::Spelling VecTypeHintSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "vec_type_hint"},
};
static constexpr const char *VecTypeHintArgNames[] = {
"TypeHint",};
struct ParsedAttrInfoVecTypeHint final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoVecTypeHint() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_VecTypeHint,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/VecTypeHintSpellings,
    /*ArgNames=*/VecTypeHintArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoVecTypeHint Instance;
};
const ParsedAttrInfoVecTypeHint ParsedAttrInfoVecTypeHint::Instance;
static constexpr ParsedAttrInfo::Spelling VectorCallSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "vectorcall"},
  {AttributeCommonInfo::AS_CXX11, "clang::vectorcall"},
  {AttributeCommonInfo::AS_C23, "clang::vectorcall"},
  {AttributeCommonInfo::AS_Keyword, "__vectorcall"},
  {AttributeCommonInfo::AS_Keyword, "_vectorcall"},
};
struct ParsedAttrInfoVectorCall final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoVectorCall() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_VectorCall,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/VectorCallSpellings,
    /*ArgNames=*/{}) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoVectorCall Instance;
};
const ParsedAttrInfoVectorCall ParsedAttrInfoVectorCall::Instance;
static constexpr ParsedAttrInfo::Spelling VectorSizeSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "vector_size"},
  {AttributeCommonInfo::AS_CXX11, "gnu::vector_size"},
  {AttributeCommonInfo::AS_C23, "gnu::vector_size"},
};
static constexpr const char *VectorSizeArgNames[] = {
"NumBytes",};
struct ParsedAttrInfoVectorSize final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoVectorSize() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_VectorSize,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/VectorSizeSpellings,
    /*ArgNames=*/VectorSizeArgNames) {}
bool isParamExpr(size_t N) const override {
  return (N == 0) || false;
}

static const ParsedAttrInfoVectorSize Instance;
};
const ParsedAttrInfoVectorSize ParsedAttrInfoVectorSize::Instance;
static constexpr ParsedAttrInfo::Spelling VisibilitySpellings[] = {
  {AttributeCommonInfo::AS_GNU, "visibility"},
  {AttributeCommonInfo::AS_CXX11, "gnu::visibility"},
  {AttributeCommonInfo::AS_C23, "gnu::visibility"},
};
static constexpr const char *VisibilityArgNames[] = {
"Visibility",};
struct ParsedAttrInfoVisibility final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoVisibility() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Visibility,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/VisibilitySpellings,
    /*ArgNames=*/VisibilityArgNames) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoVisibility Instance;
};
const ParsedAttrInfoVisibility ParsedAttrInfoVisibility::Instance;
static constexpr ParsedAttrInfo::Spelling WarnUnusedSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "warn_unused"},
  {AttributeCommonInfo::AS_CXX11, "gnu::warn_unused"},
  {AttributeCommonInfo::AS_C23, "gnu::warn_unused"},
};
struct ParsedAttrInfoWarnUnused final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoWarnUnused() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_WarnUnused,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/WarnUnusedSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<RecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "structs, unions, and classes";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) WarnUnusedAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoWarnUnused Instance;
};
const ParsedAttrInfoWarnUnused ParsedAttrInfoWarnUnused::Instance;
static constexpr ParsedAttrInfo::Spelling WarnUnusedResultSpellings[] = {
  {AttributeCommonInfo::AS_CXX11, "nodiscard"},
  {AttributeCommonInfo::AS_C23, "nodiscard"},
  {AttributeCommonInfo::AS_CXX11, "clang::warn_unused_result"},
  {AttributeCommonInfo::AS_GNU, "warn_unused_result"},
  {AttributeCommonInfo::AS_CXX11, "gnu::warn_unused_result"},
  {AttributeCommonInfo::AS_C23, "gnu::warn_unused_result"},
};
static constexpr const char *WarnUnusedResultArgNames[] = {
"Message",};
struct ParsedAttrInfoWarnUnusedResult final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoWarnUnusedResult() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_WarnUnusedResult,
    /*NumArgs=*/0,
    /*OptArgs=*/1,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/WarnUnusedResultSpellings,
    /*ArgNames=*/WarnUnusedResultArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<ObjCMethodDecl>(D) && !isa<EnumDecl>(D) && !isa<RecordDecl>(D) && !isFunctionLike(D) && !isa<TypedefNameDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "Objective-C methods, enums, structs, unions, classes, functions, function pointers, and typedefs";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

unsigned spellingIndexToSemanticSpelling(const ParsedAttr &Attr) const override {
  enum Spelling {
    CXX11_nodiscard = 0,
    C23_nodiscard = 1,
    CXX11_clang_warn_unused_result = 2,
    GNU_warn_unused_result = 3,
    CXX11_gnu_warn_unused_result = 4,
    C23_gnu_warn_unused_result = 5,
  SpellingNotCalculated = 15

  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return CXX11_nodiscard;
    case 1: return C23_nodiscard;
    case 2: return CXX11_clang_warn_unused_result;
    case 3: return GNU_warn_unused_result;
    case 4: return CXX11_gnu_warn_unused_result;
    case 5: return C23_gnu_warn_unused_result;
  }
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_enum, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_hasType_functionType, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_type_alias, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoWarnUnusedResult Instance;
};
const ParsedAttrInfoWarnUnusedResult ParsedAttrInfoWarnUnusedResult::Instance;
static constexpr ParsedAttrInfo::Spelling WeakSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "weak"},
  {AttributeCommonInfo::AS_CXX11, "gnu::weak"},
  {AttributeCommonInfo::AS_C23, "gnu::weak"},
};
struct ParsedAttrInfoWeak final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoWeak() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_Weak,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/WeakSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<VarDecl>(D) && !isa<FunctionDecl>(D) && !isa<CXXRecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "variables, functions, and classes";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) WeakAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoWeak Instance;
};
const ParsedAttrInfoWeak ParsedAttrInfoWeak::Instance;
static constexpr ParsedAttrInfo::Spelling WeakImportSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "weak_import"},
  {AttributeCommonInfo::AS_CXX11, "clang::weak_import"},
  {AttributeCommonInfo::AS_C23, "clang::weak_import"},
};
struct ParsedAttrInfoWeakImport final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoWeakImport() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_WeakImport,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/WeakImportSpellings,
    /*ArgNames=*/{}) {}
bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoWeakImport Instance;
};
const ParsedAttrInfoWeakImport ParsedAttrInfoWeakImport::Instance;
static constexpr ParsedAttrInfo::Spelling WeakRefSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "weakref"},
  {AttributeCommonInfo::AS_CXX11, "gnu::weakref"},
  {AttributeCommonInfo::AS_C23, "gnu::weakref"},
};
static constexpr const char *WeakRefArgNames[] = {
"Aliasee",};
struct ParsedAttrInfoWeakRef final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoWeakRef() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_WeakRef,
    /*NumArgs=*/0,
    /*OptArgs=*/1,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/WeakRefSpellings,
    /*ArgNames=*/WeakRefArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<VarDecl>(D) && !isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "variables and functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoWeakRef Instance;
};
const ParsedAttrInfoWeakRef ParsedAttrInfoWeakRef::Instance;
static constexpr ParsedAttrInfo::Spelling WebAssemblyExportNameSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "export_name"},
  {AttributeCommonInfo::AS_CXX11, "clang::export_name"},
  {AttributeCommonInfo::AS_C23, "clang::export_name"},
};
static constexpr const char *WebAssemblyExportNameArgNames[] = {
"ExportName",};
struct ParsedAttrInfoWebAssemblyExportName final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoWebAssemblyExportName() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_WebAssemblyExportName,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/WebAssemblyExportNameSpellings,
    /*ArgNames=*/WebAssemblyExportNameArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && (T.getArch() == llvm::Triple::wasm32 || T.getArch() == llvm::Triple::wasm64);
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoWebAssemblyExportName Instance;
};
const ParsedAttrInfoWebAssemblyExportName ParsedAttrInfoWebAssemblyExportName::Instance;
static constexpr ParsedAttrInfo::Spelling WebAssemblyFuncrefSpellings[] = {
  {AttributeCommonInfo::AS_Keyword, "__funcref"},
};
struct ParsedAttrInfoWebAssemblyFuncref final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoWebAssemblyFuncref() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_WebAssemblyFuncref,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/1,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/WebAssemblyFuncrefSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isFunctionPointer(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions pointers";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && (T.getArch() == llvm::Triple::wasm32 || T.getArch() == llvm::Triple::wasm64);
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoWebAssemblyFuncref Instance;
};
const ParsedAttrInfoWebAssemblyFuncref ParsedAttrInfoWebAssemblyFuncref::Instance;
static constexpr ParsedAttrInfo::Spelling WebAssemblyImportModuleSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "import_module"},
  {AttributeCommonInfo::AS_CXX11, "clang::import_module"},
  {AttributeCommonInfo::AS_C23, "clang::import_module"},
};
static constexpr const char *WebAssemblyImportModuleArgNames[] = {
"ImportModule",};
struct ParsedAttrInfoWebAssemblyImportModule final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoWebAssemblyImportModule() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_WebAssemblyImportModule,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/WebAssemblyImportModuleSpellings,
    /*ArgNames=*/WebAssemblyImportModuleArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && (T.getArch() == llvm::Triple::wasm32 || T.getArch() == llvm::Triple::wasm64);
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoWebAssemblyImportModule Instance;
};
const ParsedAttrInfoWebAssemblyImportModule ParsedAttrInfoWebAssemblyImportModule::Instance;
static constexpr ParsedAttrInfo::Spelling WebAssemblyImportNameSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "import_name"},
  {AttributeCommonInfo::AS_CXX11, "clang::import_name"},
  {AttributeCommonInfo::AS_C23, "clang::import_name"},
};
static constexpr const char *WebAssemblyImportNameArgNames[] = {
"ImportName",};
struct ParsedAttrInfoWebAssemblyImportName final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoWebAssemblyImportName() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_WebAssemblyImportName,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/WebAssemblyImportNameSpellings,
    /*ArgNames=*/WebAssemblyImportNameArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && (T.getArch() == llvm::Triple::wasm32 || T.getArch() == llvm::Triple::wasm64);
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoWebAssemblyImportName Instance;
};
const ParsedAttrInfoWebAssemblyImportName ParsedAttrInfoWebAssemblyImportName::Instance;
static constexpr ParsedAttrInfo::Spelling WorkGroupSizeHintSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "work_group_size_hint"},
};
static constexpr const char *WorkGroupSizeHintArgNames[] = {
"XDim","YDim","ZDim",};
struct ParsedAttrInfoWorkGroupSizeHint final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoWorkGroupSizeHint() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_WorkGroupSizeHint,
    /*NumArgs=*/3,
    /*OptArgs=*/0,
    /*NumArgMembers=*/3,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/WorkGroupSizeHintSpellings,
    /*ArgNames=*/WorkGroupSizeHintArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return (N == 0) || (N == 1) || (N == 2) || false;
}

static const ParsedAttrInfoWorkGroupSizeHint Instance;
};
const ParsedAttrInfoWorkGroupSizeHint ParsedAttrInfoWorkGroupSizeHint::Instance;
static constexpr ParsedAttrInfo::Spelling X86ForceAlignArgPointerSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "force_align_arg_pointer"},
  {AttributeCommonInfo::AS_CXX11, "gnu::force_align_arg_pointer"},
  {AttributeCommonInfo::AS_C23, "gnu::force_align_arg_pointer"},
};
struct ParsedAttrInfoX86ForceAlignArgPointer final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoX86ForceAlignArgPointer() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_X86ForceAlignArgPointer,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/1,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/0,
    /*Spellings=*/X86ForceAlignArgPointerSpellings,
    /*ArgNames=*/{}) {}
bool existsInTarget(const TargetInfo &Target) const override {
  const llvm::Triple &T = Target.getTriple(); (void)T;
  return true && (T.getArch() == llvm::Triple::x86 || T.getArch() == llvm::Triple::x86_64);
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoX86ForceAlignArgPointer Instance;
};
const ParsedAttrInfoX86ForceAlignArgPointer ParsedAttrInfoX86ForceAlignArgPointer::Instance;
static constexpr ParsedAttrInfo::Spelling XRayInstrumentSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "xray_always_instrument"},
  {AttributeCommonInfo::AS_CXX11, "clang::xray_always_instrument"},
  {AttributeCommonInfo::AS_C23, "clang::xray_always_instrument"},
  {AttributeCommonInfo::AS_GNU, "xray_never_instrument"},
  {AttributeCommonInfo::AS_CXX11, "clang::xray_never_instrument"},
  {AttributeCommonInfo::AS_C23, "clang::xray_never_instrument"},
};
struct ParsedAttrInfoXRayInstrument final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoXRayInstrument() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_XRayInstrument,
    /*NumArgs=*/0,
    /*OptArgs=*/0,
    /*NumArgMembers=*/0,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/XRayInstrumentSpellings,
    /*ArgNames=*/{}) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D) && !isa<ObjCMethodDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions and Objective-C methods";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

unsigned spellingIndexToSemanticSpelling(const ParsedAttr &Attr) const override {
  enum Spelling {
    GNU_xray_always_instrument = 0,
    CXX11_clang_xray_always_instrument = 1,
    C23_clang_xray_always_instrument = 2,
    GNU_xray_never_instrument = 3,
    CXX11_clang_xray_never_instrument = 4,
    C23_clang_xray_never_instrument = 5,
  SpellingNotCalculated = 15

  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return GNU_xray_always_instrument;
    case 1: return CXX11_clang_xray_always_instrument;
    case 2: return C23_clang_xray_always_instrument;
    case 3: return GNU_xray_never_instrument;
    case 4: return CXX11_clang_xray_never_instrument;
    case 5: return C23_clang_xray_never_instrument;
  }
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC));
}

AttrHandling handleDeclAttribute(Sema &S, Decl *D,const ParsedAttr &Attr) const override {
  D->addAttr(::new (S.Context) XRayInstrumentAttr(S.Context, Attr));
  return AttributeApplied;
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoXRayInstrument Instance;
};
const ParsedAttrInfoXRayInstrument ParsedAttrInfoXRayInstrument::Instance;
static constexpr ParsedAttrInfo::Spelling XRayLogArgsSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "xray_log_args"},
  {AttributeCommonInfo::AS_CXX11, "clang::xray_log_args"},
  {AttributeCommonInfo::AS_C23, "clang::xray_log_args"},
};
static constexpr const char *XRayLogArgsArgNames[] = {
"ArgumentCount",};
struct ParsedAttrInfoXRayLogArgs final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoXRayLogArgs() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_XRayLogArgs,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/0,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/XRayLogArgsSpellings,
    /*ArgNames=*/XRayLogArgsArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D) && !isa<ObjCMethodDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions and Objective-C methods";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoXRayLogArgs Instance;
};
const ParsedAttrInfoXRayLogArgs ParsedAttrInfoXRayLogArgs::Instance;
static constexpr ParsedAttrInfo::Spelling ZeroCallUsedRegsSpellings[] = {
  {AttributeCommonInfo::AS_GNU, "zero_call_used_regs"},
  {AttributeCommonInfo::AS_CXX11, "gnu::zero_call_used_regs"},
  {AttributeCommonInfo::AS_C23, "gnu::zero_call_used_regs"},
};
static constexpr const char *ZeroCallUsedRegsArgNames[] = {
"ZeroCallUsedRegs",};
struct ParsedAttrInfoZeroCallUsedRegs final : public ParsedAttrInfo {
  constexpr ParsedAttrInfoZeroCallUsedRegs() : ParsedAttrInfo(
    /*AttrKind=*/ParsedAttr::AT_ZeroCallUsedRegs,
    /*NumArgs=*/1,
    /*OptArgs=*/0,
    /*NumArgMembers=*/1,
    /*HasCustomParsing=*/0,
    /*AcceptsExprPack=*/0,
    /*IsTargetSpecific=*/0,
    /*IsType=*/0,
    /*IsStmt=*/0,
    /*IsKnownToGCC=*/1,
    /*IsSupportedByPragmaAttribute=*/1,
    /*Spellings=*/ZeroCallUsedRegsSpellings,
    /*ArgNames=*/ZeroCallUsedRegsArgNames) {}
bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << Attr << Attr.isRegularKeywordAttribute() << "functions";
    return false;
  }
  return true;
}

bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL, const Stmt *St) const override {
  S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
    << AL << AL.isRegularKeywordAttribute() << St->getBeginLoc();
  return false;
}

void getPragmaAttributeMatchRules(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) const override {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

bool isParamExpr(size_t N) const override {
  return false;
}

static const ParsedAttrInfoZeroCallUsedRegs Instance;
};
const ParsedAttrInfoZeroCallUsedRegs ParsedAttrInfoZeroCallUsedRegs::Instance;
static const ParsedAttrInfo *AttrInfoMap[] = {
&ParsedAttrInfoAArch64SVEPcs::Instance,
&ParsedAttrInfoAArch64VectorPcs::Instance,
&ParsedAttrInfoAMDGPUFlatWorkGroupSize::Instance,
&ParsedAttrInfoAMDGPUKernelCall::Instance,
&ParsedAttrInfoAMDGPUMaxNumWorkGroups::Instance,
&ParsedAttrInfoAMDGPUNumSGPR::Instance,
&ParsedAttrInfoAMDGPUNumVGPR::Instance,
&ParsedAttrInfoAMDGPUWavesPerEU::Instance,
&ParsedAttrInfoInterrupt::Instance,
&ParsedAttrInfoARMInterruptSaveFP::Instance,
&ParsedAttrInfoARMSaveFP::Instance,
&ParsedAttrInfoAVRSignal::Instance,
&ParsedAttrInfoAbiTag::Instance,
&ParsedAttrInfoAcquireCapability::Instance,
&ParsedAttrInfoAcquireHandle::Instance,
&ParsedAttrInfoAcquiredAfter::Instance,
&ParsedAttrInfoAcquiredBefore::Instance,
&ParsedAttrInfoAddressSpace::Instance,
&ParsedAttrInfoAlias::Instance,
&ParsedAttrInfoAlignValue::Instance,
&ParsedAttrInfoAligned::Instance,
&ParsedAttrInfoAllocAlign::Instance,
&ParsedAttrInfoAllocSize::Instance,
&ParsedAttrInfoAllocating::Instance,
&ParsedAttrInfoAlwaysDestroy::Instance,
&ParsedAttrInfoAlwaysInline::Instance,
&ParsedAttrInfoAnalyzerNoReturn::Instance,
&ParsedAttrInfoAnnotate::Instance,
&ParsedAttrInfoAnnotateType::Instance,
&ParsedAttrInfoAnyX86NoCallerSavedRegisters::Instance,
&ParsedAttrInfoAnyX86NoCfCheck::Instance,
&ParsedAttrInfoArcWeakrefUnavailable::Instance,
&ParsedAttrInfoArgumentWithTypeTag::Instance,
&ParsedAttrInfoArmAgnostic::Instance,
&ParsedAttrInfoArmBuiltinAlias::Instance,
&ParsedAttrInfoArmIn::Instance,
&ParsedAttrInfoArmInOut::Instance,
&ParsedAttrInfoArmLocallyStreaming::Instance,
&ParsedAttrInfoArmMveStrictPolymorphism::Instance,
&ParsedAttrInfoArmNew::Instance,
&ParsedAttrInfoArmOut::Instance,
&ParsedAttrInfoArmPreserves::Instance,
&ParsedAttrInfoArmStreaming::Instance,
&ParsedAttrInfoArmStreamingCompatible::Instance,
&ParsedAttrInfoArmSveVectorBits::Instance,
&ParsedAttrInfoArtificial::Instance,
&ParsedAttrInfoAssertCapability::Instance,
&ParsedAttrInfoAssumeAligned::Instance,
&ParsedAttrInfoAtomic::Instance,
&ParsedAttrInfoAvailability::Instance,
&ParsedAttrInfoAvailableOnlyInDefaultEvalMethod::Instance,
&ParsedAttrInfoBPFFastCall::Instance,
&ParsedAttrInfoBPFPreserveAccessIndex::Instance,
&ParsedAttrInfoBPFPreserveStaticOffset::Instance,
&ParsedAttrInfoBTFDeclTag::Instance,
&ParsedAttrInfoBTFTypeTag::Instance,
&ParsedAttrInfoBlocking::Instance,
&ParsedAttrInfoBlocks::Instance,
&ParsedAttrInfoBuiltinAlias::Instance,
&ParsedAttrInfoCDecl::Instance,
&ParsedAttrInfoCFAuditedTransfer::Instance,
&ParsedAttrInfoCFConsumed::Instance,
&ParsedAttrInfoCFGuard::Instance,
&ParsedAttrInfoCFICanonicalJumpTable::Instance,
&ParsedAttrInfoCFReturnsNotRetained::Instance,
&ParsedAttrInfoCFReturnsRetained::Instance,
&ParsedAttrInfoCFUnknownTransfer::Instance,
&ParsedAttrInfoCPUDispatch::Instance,
&ParsedAttrInfoCPUSpecific::Instance,
&ParsedAttrInfoCUDAConstant::Instance,
&ParsedAttrInfoCUDADevice::Instance,
&ParsedAttrInfoCUDADeviceBuiltinSurfaceType::Instance,
&ParsedAttrInfoCUDADeviceBuiltinTextureType::Instance,
&ParsedAttrInfoCUDAGlobal::Instance,
&ParsedAttrInfoCUDAGridConstant::Instance,
&ParsedAttrInfoCUDAHost::Instance,
&ParsedAttrInfoCUDAInvalidTarget::Instance,
&ParsedAttrInfoCUDALaunchBounds::Instance,
&ParsedAttrInfoCUDAShared::Instance,
&ParsedAttrInfoCXX11NoReturn::Instance,
&ParsedAttrInfoCXXAssume::Instance,
&ParsedAttrInfoCallableWhen::Instance,
&ParsedAttrInfoCallback::Instance,
&ParsedAttrInfoCalledOnce::Instance,
&ParsedAttrInfoCapability::Instance,
&ParsedAttrInfoCarriesDependency::Instance,
&ParsedAttrInfoCleanup::Instance,
&ParsedAttrInfoClspvLibclcBuiltin::Instance,
&ParsedAttrInfoCmseNSCall::Instance,
&ParsedAttrInfoCmseNSEntry::Instance,
&ParsedAttrInfoCodeAlign::Instance,
&ParsedAttrInfoCodeModel::Instance,
&ParsedAttrInfoCodeSeg::Instance,
&ParsedAttrInfoCold::Instance,
&ParsedAttrInfoCommon::Instance,
&ParsedAttrInfoConst::Instance,
&ParsedAttrInfoConstInit::Instance,
&ParsedAttrInfoConstructor::Instance,
&ParsedAttrInfoConsumable::Instance,
&ParsedAttrInfoConsumableAutoCast::Instance,
&ParsedAttrInfoConsumableSetOnRead::Instance,
&ParsedAttrInfoConvergent::Instance,
&ParsedAttrInfoCoroAwaitElidable::Instance,
&ParsedAttrInfoCoroAwaitElidableArgument::Instance,
&ParsedAttrInfoCoroDisableLifetimeBound::Instance,
&ParsedAttrInfoCoroLifetimeBound::Instance,
&ParsedAttrInfoCoroOnlyDestroyWhenComplete::Instance,
&ParsedAttrInfoCoroReturnType::Instance,
&ParsedAttrInfoCoroWrapper::Instance,
&ParsedAttrInfoCountedBy::Instance,
&ParsedAttrInfoCountedByOrNull::Instance,
&ParsedAttrInfoDLLExport::Instance,
&ParsedAttrInfoDLLExportStaticLocal::Instance,
&ParsedAttrInfoDLLImport::Instance,
&ParsedAttrInfoDLLImportStaticLocal::Instance,
&ParsedAttrInfoDeprecated::Instance,
&ParsedAttrInfoDestructor::Instance,
&ParsedAttrInfoDiagnoseAsBuiltin::Instance,
&ParsedAttrInfoDiagnoseIf::Instance,
&ParsedAttrInfoDisableSanitizerInstrumentation::Instance,
&ParsedAttrInfoDisableTailCalls::Instance,
&ParsedAttrInfoEmptyBases::Instance,
&ParsedAttrInfoEnableIf::Instance,
&ParsedAttrInfoEnforceTCB::Instance,
&ParsedAttrInfoEnforceTCBLeaf::Instance,
&ParsedAttrInfoEnumExtensibility::Instance,
&ParsedAttrInfoError::Instance,
&ParsedAttrInfoExcludeFromExplicitInstantiation::Instance,
&ParsedAttrInfoExplicitInit::Instance,
&ParsedAttrInfoExtVectorType::Instance,
&ParsedAttrInfoExternalSourceSymbol::Instance,
&ParsedAttrInfoFallThrough::Instance,
&ParsedAttrInfoFastCall::Instance,
&ParsedAttrInfoFlagEnum::Instance,
&ParsedAttrInfoFlatten::Instance,
&ParsedAttrInfoFormat::Instance,
&ParsedAttrInfoFormatArg::Instance,
&ParsedAttrInfoFormatMatches::Instance,
&ParsedAttrInfoFunctionReturnThunks::Instance,
&ParsedAttrInfoGNUInline::Instance,
&ParsedAttrInfoGuardedBy::Instance,
&ParsedAttrInfoGuardedVar::Instance,
&ParsedAttrInfoHIPManaged::Instance,
&ParsedAttrInfoHLSLContainedType::Instance,
&ParsedAttrInfoHLSLControlFlowHint::Instance,
&ParsedAttrInfoHLSLGroupSharedAddressSpace::Instance,
&ParsedAttrInfoHLSLLoopHint::Instance,
&ParsedAttrInfoHLSLNumThreads::Instance,
&ParsedAttrInfoHLSLPackOffset::Instance,
&ParsedAttrInfoHLSLParamModifier::Instance,
&ParsedAttrInfoHLSLROV::Instance,
&ParsedAttrInfoHLSLRawBuffer::Instance,
&ParsedAttrInfoHLSLResourceBinding::Instance,
&ParsedAttrInfoHLSLResourceClass::Instance,
&ParsedAttrInfoHLSLSV_DispatchThreadID::Instance,
&ParsedAttrInfoHLSLSV_GroupID::Instance,
&ParsedAttrInfoHLSLSV_GroupIndex::Instance,
&ParsedAttrInfoHLSLSV_GroupThreadID::Instance,
&ParsedAttrInfoHLSLShader::Instance,
&ParsedAttrInfoHLSLWaveSize::Instance,
&ParsedAttrInfoHot::Instance,
&ParsedAttrInfoHybridPatchable::Instance,
&ParsedAttrInfoIBAction::Instance,
&ParsedAttrInfoIBOutlet::Instance,
&ParsedAttrInfoIBOutletCollection::Instance,
&ParsedAttrInfoIFunc::Instance,
&ParsedAttrInfoInitPriority::Instance,
&ParsedAttrInfoIntelOclBicc::Instance,
&ParsedAttrInfoInternalLinkage::Instance,
&ParsedAttrInfoLTOVisibilityPublic::Instance,
&ParsedAttrInfoLayoutVersion::Instance,
&ParsedAttrInfoLeaf::Instance,
&ParsedAttrInfoLifetimeBound::Instance,
&ParsedAttrInfoLifetimeCaptureBy::Instance,
&ParsedAttrInfoLikely::Instance,
&ParsedAttrInfoLoaderUninitialized::Instance,
&ParsedAttrInfoLockReturned::Instance,
&ParsedAttrInfoLockable::Instance,
&ParsedAttrInfoLocksExcluded::Instance,
&ParsedAttrInfoLoopHint::Instance,
&ParsedAttrInfoM68kRTD::Instance,
&ParsedAttrInfoMIGServerRoutine::Instance,
&ParsedAttrInfoMSABI::Instance,
&ParsedAttrInfoMSAllocator::Instance,
&ParsedAttrInfoMSConstexpr::Instance,
&ParsedAttrInfoMSInheritance::Instance,
&ParsedAttrInfoMSNoVTable::Instance,
&ParsedAttrInfoMSStruct::Instance,
&ParsedAttrInfoMatrixType::Instance,
&ParsedAttrInfoMayAlias::Instance,
&ParsedAttrInfoMaybeUndef::Instance,
&ParsedAttrInfoMicroMips::Instance,
&ParsedAttrInfoMinSize::Instance,
&ParsedAttrInfoMinVectorWidth::Instance,
&ParsedAttrInfoMips16::Instance,
&ParsedAttrInfoMipsLongCall::Instance,
&ParsedAttrInfoMipsShortCall::Instance,
&ParsedAttrInfoMode::Instance,
&ParsedAttrInfoMustTail::Instance,
&ParsedAttrInfoNSConsumed::Instance,
&ParsedAttrInfoNSConsumesSelf::Instance,
&ParsedAttrInfoNSErrorDomain::Instance,
&ParsedAttrInfoNSReturnsAutoreleased::Instance,
&ParsedAttrInfoNSReturnsNotRetained::Instance,
&ParsedAttrInfoNSReturnsRetained::Instance,
&ParsedAttrInfoNVPTXKernel::Instance,
&ParsedAttrInfoNaked::Instance,
&ParsedAttrInfoNeonPolyVectorType::Instance,
&ParsedAttrInfoNeonVectorType::Instance,
&ParsedAttrInfoNoAlias::Instance,
&ParsedAttrInfoNoBuiltin::Instance,
&ParsedAttrInfoNoCommon::Instance,
&ParsedAttrInfoNoConvergent::Instance,
&ParsedAttrInfoNoDebug::Instance,
&ParsedAttrInfoNoDeref::Instance,
&ParsedAttrInfoNoDestroy::Instance,
&ParsedAttrInfoNoDuplicate::Instance,
&ParsedAttrInfoNoEscape::Instance,
&ParsedAttrInfoNoInline::Instance,
&ParsedAttrInfoNoInstrumentFunction::Instance,
&ParsedAttrInfoNoMerge::Instance,
&ParsedAttrInfoNoMicroMips::Instance,
&ParsedAttrInfoNoMips16::Instance,
&ParsedAttrInfoNoProfileFunction::Instance,
&ParsedAttrInfoNoRandomizeLayout::Instance,
&ParsedAttrInfoNoReturn::Instance,
&ParsedAttrInfoNoSanitize::Instance,
&ParsedAttrInfoNoSanitizeSpecific::Instance,
&ParsedAttrInfoNoSpecializations::Instance,
&ParsedAttrInfoNoSpeculativeLoadHardening::Instance,
&ParsedAttrInfoNoSplitStack::Instance,
&ParsedAttrInfoNoStackProtector::Instance,
&ParsedAttrInfoNoThreadSafetyAnalysis::Instance,
&ParsedAttrInfoNoThrow::Instance,
&ParsedAttrInfoNoTrivialAutoVarInit::Instance,
&ParsedAttrInfoNoUniqueAddress::Instance,
&ParsedAttrInfoNoUwtable::Instance,
&ParsedAttrInfoNonAllocating::Instance,
&ParsedAttrInfoNonBlocking::Instance,
&ParsedAttrInfoNonNull::Instance,
&ParsedAttrInfoNotTailCalled::Instance,
&ParsedAttrInfoOMPAssume::Instance,
&ParsedAttrInfoOSConsumed::Instance,
&ParsedAttrInfoOSConsumesThis::Instance,
&ParsedAttrInfoOSReturnsNotRetained::Instance,
&ParsedAttrInfoOSReturnsRetained::Instance,
&ParsedAttrInfoOSReturnsRetainedOnNonZero::Instance,
&ParsedAttrInfoOSReturnsRetainedOnZero::Instance,
&ParsedAttrInfoObjCBoxable::Instance,
&ParsedAttrInfoObjCBridge::Instance,
&ParsedAttrInfoObjCBridgeMutable::Instance,
&ParsedAttrInfoObjCBridgeRelated::Instance,
&ParsedAttrInfoObjCClassStub::Instance,
&ParsedAttrInfoObjCDesignatedInitializer::Instance,
&ParsedAttrInfoObjCDirect::Instance,
&ParsedAttrInfoObjCDirectMembers::Instance,
&ParsedAttrInfoObjCException::Instance,
&ParsedAttrInfoObjCExplicitProtocolImpl::Instance,
&ParsedAttrInfoObjCExternallyRetained::Instance,
&ParsedAttrInfoObjCGC::Instance,
&ParsedAttrInfoObjCIndependentClass::Instance,
&ParsedAttrInfoObjCInertUnsafeUnretained::Instance,
&ParsedAttrInfoObjCKindOf::Instance,
&ParsedAttrInfoObjCMethodFamily::Instance,
&ParsedAttrInfoObjCNSObject::Instance,
&ParsedAttrInfoObjCNonLazyClass::Instance,
&ParsedAttrInfoObjCNonRuntimeProtocol::Instance,
&ParsedAttrInfoObjCOwnership::Instance,
&ParsedAttrInfoObjCPreciseLifetime::Instance,
&ParsedAttrInfoObjCRequiresPropertyDefs::Instance,
&ParsedAttrInfoObjCRequiresSuper::Instance,
&ParsedAttrInfoObjCReturnsInnerPointer::Instance,
&ParsedAttrInfoObjCRootClass::Instance,
&ParsedAttrInfoObjCRuntimeName::Instance,
&ParsedAttrInfoObjCRuntimeVisible::Instance,
&ParsedAttrInfoObjCSubclassingRestricted::Instance,
&ParsedAttrInfoOpenACCRoutineAnnot::Instance,
&ParsedAttrInfoOpenCLAccess::Instance,
&ParsedAttrInfoOpenCLConstantAddressSpace::Instance,
&ParsedAttrInfoOpenCLGenericAddressSpace::Instance,
&ParsedAttrInfoOpenCLGlobalAddressSpace::Instance,
&ParsedAttrInfoOpenCLGlobalDeviceAddressSpace::Instance,
&ParsedAttrInfoOpenCLGlobalHostAddressSpace::Instance,
&ParsedAttrInfoOpenCLIntelReqdSubGroupSize::Instance,
&ParsedAttrInfoOpenCLKernel::Instance,
&ParsedAttrInfoOpenCLLocalAddressSpace::Instance,
&ParsedAttrInfoOpenCLNoSVM::Instance,
&ParsedAttrInfoOpenCLPrivateAddressSpace::Instance,
&ParsedAttrInfoOpenCLUnrollHint::Instance,
&ParsedAttrInfoOptimizeNone::Instance,
&ParsedAttrInfoOverloadable::Instance,
&ParsedAttrInfoOwner::Instance,
&ParsedAttrInfoOwnership::Instance,
&ParsedAttrInfoPacked::Instance,
&ParsedAttrInfoParamTypestate::Instance,
&ParsedAttrInfoPascal::Instance,
&ParsedAttrInfoPassObjectSize::Instance,
&ParsedAttrInfoPatchableFunctionEntry::Instance,
&ParsedAttrInfoPcs::Instance,
&ParsedAttrInfoPointer::Instance,
&ParsedAttrInfoPointerAuth::Instance,
&ParsedAttrInfoPragmaClangBSSSection::Instance,
&ParsedAttrInfoPragmaClangDataSection::Instance,
&ParsedAttrInfoPragmaClangRelroSection::Instance,
&ParsedAttrInfoPragmaClangRodataSection::Instance,
&ParsedAttrInfoPragmaClangTextSection::Instance,
&ParsedAttrInfoPreferredName::Instance,
&ParsedAttrInfoPreferredType::Instance,
&ParsedAttrInfoPreserveAll::Instance,
&ParsedAttrInfoPreserveMost::Instance,
&ParsedAttrInfoPreserveNone::Instance,
&ParsedAttrInfoPtGuardedBy::Instance,
&ParsedAttrInfoPtGuardedVar::Instance,
&ParsedAttrInfoPtr32::Instance,
&ParsedAttrInfoPtr64::Instance,
&ParsedAttrInfoPure::Instance,
&ParsedAttrInfoRISCVRVVVectorBits::Instance,
&ParsedAttrInfoRISCVVLSCC::Instance,
&ParsedAttrInfoRISCVVectorCC::Instance,
&ParsedAttrInfoRandomizeLayout::Instance,
&ParsedAttrInfoReadOnlyPlacement::Instance,
&ParsedAttrInfoRegCall::Instance,
&ParsedAttrInfoRegparm::Instance,
&ParsedAttrInfoReinitializes::Instance,
&ParsedAttrInfoReleaseCapability::Instance,
&ParsedAttrInfoReleaseHandle::Instance,
&ParsedAttrInfoReqdWorkGroupSize::Instance,
&ParsedAttrInfoRequiresCapability::Instance,
&ParsedAttrInfoRestrict::Instance,
&ParsedAttrInfoRetain::Instance,
&ParsedAttrInfoReturnTypestate::Instance,
&ParsedAttrInfoReturnsNonNull::Instance,
&ParsedAttrInfoReturnsTwice::Instance,
&ParsedAttrInfoSPtr::Instance,
&ParsedAttrInfoSYCLKernel::Instance,
&ParsedAttrInfoSYCLKernelEntryPoint::Instance,
&ParsedAttrInfoSYCLSpecialClass::Instance,
&ParsedAttrInfoScopedLockable::Instance,
&ParsedAttrInfoSection::Instance,
&ParsedAttrInfoSelectAny::Instance,
&ParsedAttrInfoSentinel::Instance,
&ParsedAttrInfoSetTypestate::Instance,
&ParsedAttrInfoSizedBy::Instance,
&ParsedAttrInfoSizedByOrNull::Instance,
&ParsedAttrInfoSpeculativeLoadHardening::Instance,
&ParsedAttrInfoStandaloneDebug::Instance,
&ParsedAttrInfoStdCall::Instance,
&ParsedAttrInfoStrictFP::Instance,
&ParsedAttrInfoStrictGuardStackCheck::Instance,
&ParsedAttrInfoSuppress::Instance,
&ParsedAttrInfoSwiftAsync::Instance,
&ParsedAttrInfoSwiftAsyncCall::Instance,
&ParsedAttrInfoSwiftAsyncContext::Instance,
&ParsedAttrInfoSwiftAsyncError::Instance,
&ParsedAttrInfoSwiftAsyncName::Instance,
&ParsedAttrInfoSwiftAttr::Instance,
&ParsedAttrInfoSwiftBridge::Instance,
&ParsedAttrInfoSwiftBridgedTypedef::Instance,
&ParsedAttrInfoSwiftCall::Instance,
&ParsedAttrInfoSwiftContext::Instance,
&ParsedAttrInfoSwiftError::Instance,
&ParsedAttrInfoSwiftErrorResult::Instance,
&ParsedAttrInfoSwiftIndirectResult::Instance,
&ParsedAttrInfoSwiftName::Instance,
&ParsedAttrInfoSwiftNewType::Instance,
&ParsedAttrInfoSwiftObjCMembers::Instance,
&ParsedAttrInfoSwiftPrivate::Instance,
&ParsedAttrInfoSysVABI::Instance,
&ParsedAttrInfoTLSModel::Instance,
&ParsedAttrInfoTarget::Instance,
&ParsedAttrInfoTargetClones::Instance,
&ParsedAttrInfoTargetVersion::Instance,
&ParsedAttrInfoTestTypestate::Instance,
&ParsedAttrInfoThisCall::Instance,
&ParsedAttrInfoThread::Instance,
&ParsedAttrInfoTransparentUnion::Instance,
&ParsedAttrInfoTrivialABI::Instance,
&ParsedAttrInfoTryAcquireCapability::Instance,
&ParsedAttrInfoTypeNonNull::Instance,
&ParsedAttrInfoTypeNullUnspecified::Instance,
&ParsedAttrInfoTypeNullable::Instance,
&ParsedAttrInfoTypeNullableResult::Instance,
&ParsedAttrInfoTypeTagForDatatype::Instance,
&ParsedAttrInfoTypeVisibility::Instance,
&ParsedAttrInfoUPtr::Instance,
&ParsedAttrInfoUnavailable::Instance,
&ParsedAttrInfoUninitialized::Instance,
&ParsedAttrInfoUnlikely::Instance,
&ParsedAttrInfoUnsafeBufferUsage::Instance,
&ParsedAttrInfoUnused::Instance,
&ParsedAttrInfoUseHandle::Instance,
&ParsedAttrInfoUsed::Instance,
&ParsedAttrInfoUsingIfExists::Instance,
&ParsedAttrInfoUuid::Instance,
&ParsedAttrInfoVTablePointerAuthentication::Instance,
&ParsedAttrInfoVecReturn::Instance,
&ParsedAttrInfoVecTypeHint::Instance,
&ParsedAttrInfoVectorCall::Instance,
&ParsedAttrInfoVectorSize::Instance,
&ParsedAttrInfoVisibility::Instance,
&ParsedAttrInfoWarnUnused::Instance,
&ParsedAttrInfoWarnUnusedResult::Instance,
&ParsedAttrInfoWeak::Instance,
&ParsedAttrInfoWeakImport::Instance,
&ParsedAttrInfoWeakRef::Instance,
&ParsedAttrInfoWebAssemblyExportName::Instance,
&ParsedAttrInfoWebAssemblyFuncref::Instance,
&ParsedAttrInfoWebAssemblyImportModule::Instance,
&ParsedAttrInfoWebAssemblyImportName::Instance,
&ParsedAttrInfoWorkGroupSizeHint::Instance,
&ParsedAttrInfoX86ForceAlignArgPointer::Instance,
&ParsedAttrInfoXRayInstrument::Instance,
&ParsedAttrInfoXRayLogArgs::Instance,
&ParsedAttrInfoZeroCallUsedRegs::Instance,
};

static void handleAttrWithDelayedArgs(Sema &S, Decl *D, const ParsedAttr &Attr) {
  SmallVector<Expr *, 4> ArgExprs;
  ArgExprs.reserve(Attr.getNumArgs());
  for (unsigned I = 0; I < Attr.getNumArgs(); ++I) {
    assert(!Attr.isArgIdent(I));
    ArgExprs.push_back(Attr.getArgAsExpr(I));
  }
  clang::Attr *CreatedAttr = nullptr;
  switch (Attr.getKind()) {
  default:
    llvm_unreachable("Attribute cannot hold delayed arguments.");
  case ParsedAttr::AT_Annotate: {
    CreatedAttr = AnnotateAttr::CreateWithDelayedArgs(S.Context, ArgExprs.data(), ArgExprs.size(), Attr);
    break;
  }
  case ParsedAttr::AT_AnnotateType: {
    CreatedAttr = AnnotateTypeAttr::CreateWithDelayedArgs(S.Context, ArgExprs.data(), ArgExprs.size(), Attr);
    break;
  }
  }
  D->addAttr(CreatedAttr);
}

static bool checkAttributeMatchRuleAppliesTo(const Decl *D, attr::SubjectMatchRule rule) {
  switch (rule) {
  case attr::SubjectMatchRule_block:
    return isa<BlockDecl>(D);
  case attr::SubjectMatchRule_enum:
    return isa<EnumDecl>(D);
  case attr::SubjectMatchRule_enum_constant:
    return isa<EnumConstantDecl>(D);
  case attr::SubjectMatchRule_field:
    return isa<FieldDecl>(D);
  case attr::SubjectMatchRule_function:
    return isa<FunctionDecl>(D);
  case attr::SubjectMatchRule_function_is_member:
    return isa<CXXMethodDecl>(D);
  case attr::SubjectMatchRule_namespace:
    return isa<NamespaceDecl>(D);
  case attr::SubjectMatchRule_objc_category:
    return isa<ObjCCategoryDecl>(D);
  case attr::SubjectMatchRule_objc_implementation:
    return isa<ObjCImplDecl>(D);
  case attr::SubjectMatchRule_objc_interface:
    return isa<ObjCInterfaceDecl>(D);
  case attr::SubjectMatchRule_objc_method:
    return isa<ObjCMethodDecl>(D);
  case attr::SubjectMatchRule_objc_method_is_instance:
    return isObjCInstanceMethod(D);
  case attr::SubjectMatchRule_objc_property:
    return isa<ObjCPropertyDecl>(D);
  case attr::SubjectMatchRule_objc_protocol:
    return isa<ObjCProtocolDecl>(D);
  case attr::SubjectMatchRule_record:
    return isa<RecordDecl>(D) || isa<CXXRecordDecl>(D);
  case attr::SubjectMatchRule_record_not_is_union:
    return isStruct(D);
  case attr::SubjectMatchRule_hasType_abstract:
    assert(false && "Abstract matcher rule isn't allowed");
    return false;
  case attr::SubjectMatchRule_hasType_functionType:
    return isFunctionLike(D);
  case attr::SubjectMatchRule_type_alias:
    return isa<TypedefNameDecl>(D);
  case attr::SubjectMatchRule_variable:
    return isa<VarDecl>(D);
  case attr::SubjectMatchRule_variable_is_thread_local:
    return isTLSVar(D);
  case attr::SubjectMatchRule_variable_is_global:
    return isGlobalVar(D);
  case attr::SubjectMatchRule_variable_is_local:
    return isLocalVar(D);
  case attr::SubjectMatchRule_variable_is_parameter:
    return isa<ParmVarDecl>(D);
  case attr::SubjectMatchRule_variable_not_is_parameter:
    return isNonParmVar(D);
  }
  llvm_unreachable("Invalid match rule");
return false;
}

#elif defined(WANT_DECL_MERGE_LOGIC)

static bool DiagnoseMutualExclusions(Sema &S, const NamedDecl *D, const Attr *A) {
  if (const auto *Second = dyn_cast<AlwaysDestroyAttr>(A)) {
    if (const auto *First = D->getAttr<NoDestroyAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    return true;
  }
  if (const auto *Second = dyn_cast<AlwaysInlineAttr>(A)) {
    if (const auto *First = D->getAttr<NotTailCalledAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    return true;
  }
  if (const auto *Second = dyn_cast<CFAuditedTransferAttr>(A)) {
    if (const auto *First = D->getAttr<CFUnknownTransferAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    return true;
  }
  if (const auto *Second = dyn_cast<CFUnknownTransferAttr>(A)) {
    if (const auto *First = D->getAttr<CFAuditedTransferAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    return true;
  }
  if (const auto *Second = dyn_cast<CPUDispatchAttr>(A)) {
    if (const auto *First = D->getAttr<TargetClonesAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    if (const auto *First = D->getAttr<TargetVersionAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    if (const auto *First = D->getAttr<TargetAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    if (const auto *First = D->getAttr<CPUSpecificAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    return true;
  }
  if (const auto *Second = dyn_cast<CPUSpecificAttr>(A)) {
    if (const auto *First = D->getAttr<TargetClonesAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    if (const auto *First = D->getAttr<TargetVersionAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    if (const auto *First = D->getAttr<TargetAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    if (const auto *First = D->getAttr<CPUDispatchAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    return true;
  }
  if (const auto *Second = dyn_cast<CUDAConstantAttr>(A)) {
    if (const auto *First = D->getAttr<CUDASharedAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    if (const auto *First = D->getAttr<HIPManagedAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    return true;
  }
  if (const auto *Second = dyn_cast<CUDADeviceAttr>(A)) {
    if (const auto *First = D->getAttr<CUDAGlobalAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    return true;
  }
  if (const auto *Second = dyn_cast<CUDADeviceBuiltinSurfaceTypeAttr>(A)) {
    if (const auto *First = D->getAttr<CUDADeviceBuiltinTextureTypeAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    return true;
  }
  if (const auto *Second = dyn_cast<CUDADeviceBuiltinTextureTypeAttr>(A)) {
    if (const auto *First = D->getAttr<CUDADeviceBuiltinSurfaceTypeAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    return true;
  }
  if (const auto *Second = dyn_cast<CUDAGlobalAttr>(A)) {
    if (const auto *First = D->getAttr<CUDADeviceAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    if (const auto *First = D->getAttr<CUDAHostAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    return true;
  }
  if (const auto *Second = dyn_cast<CUDAHostAttr>(A)) {
    if (const auto *First = D->getAttr<CUDAGlobalAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    return true;
  }
  if (const auto *Second = dyn_cast<CUDASharedAttr>(A)) {
    if (const auto *First = D->getAttr<CUDAConstantAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    if (const auto *First = D->getAttr<HIPManagedAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    return true;
  }
  if (const auto *Second = dyn_cast<ColdAttr>(A)) {
    if (const auto *First = D->getAttr<HotAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    return true;
  }
  if (const auto *Second = dyn_cast<CommonAttr>(A)) {
    if (const auto *First = D->getAttr<InternalLinkageAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    return true;
  }
  if (const auto *Second = dyn_cast<ConvergentAttr>(A)) {
    if (const auto *First = D->getAttr<NoConvergentAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    return true;
  }
  if (const auto *Second = dyn_cast<DisableTailCallsAttr>(A)) {
    if (const auto *First = D->getAttr<NakedAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    return true;
  }
  if (const auto *Second = dyn_cast<HIPManagedAttr>(A)) {
    if (const auto *First = D->getAttr<CUDAConstantAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    if (const auto *First = D->getAttr<CUDASharedAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    return true;
  }
  if (const auto *Second = dyn_cast<HotAttr>(A)) {
    if (const auto *First = D->getAttr<ColdAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    return true;
  }
  if (const auto *Second = dyn_cast<InternalLinkageAttr>(A)) {
    if (const auto *First = D->getAttr<CommonAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    return true;
  }
  if (const auto *Second = dyn_cast<MicroMipsAttr>(A)) {
    if (const auto *First = D->getAttr<Mips16Attr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    return true;
  }
  if (const auto *Second = dyn_cast<Mips16Attr>(A)) {
    if (const auto *First = D->getAttr<MipsInterruptAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    if (const auto *First = D->getAttr<MicroMipsAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    return true;
  }
  if (const auto *Second = dyn_cast<MipsLongCallAttr>(A)) {
    if (const auto *First = D->getAttr<MipsShortCallAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    return true;
  }
  if (const auto *Second = dyn_cast<MipsShortCallAttr>(A)) {
    if (const auto *First = D->getAttr<MipsLongCallAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    return true;
  }
  if (const auto *Second = dyn_cast<NakedAttr>(A)) {
    if (const auto *First = D->getAttr<DisableTailCallsAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    return true;
  }
  if (const auto *Second = dyn_cast<NoConvergentAttr>(A)) {
    if (const auto *First = D->getAttr<ConvergentAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    return true;
  }
  if (const auto *Second = dyn_cast<NoDestroyAttr>(A)) {
    if (const auto *First = D->getAttr<AlwaysDestroyAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    return true;
  }
  if (const auto *Second = dyn_cast<NoRandomizeLayoutAttr>(A)) {
    if (const auto *First = D->getAttr<RandomizeLayoutAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    return true;
  }
  if (const auto *Second = dyn_cast<NoSpeculativeLoadHardeningAttr>(A)) {
    if (const auto *First = D->getAttr<SpeculativeLoadHardeningAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    return true;
  }
  if (const auto *Second = dyn_cast<NotTailCalledAttr>(A)) {
    if (const auto *First = D->getAttr<AlwaysInlineAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    return true;
  }
  if (const auto *Second = dyn_cast<OwnerAttr>(A)) {
    if (const auto *First = D->getAttr<PointerAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    return true;
  }
  if (const auto *Second = dyn_cast<PointerAttr>(A)) {
    if (const auto *First = D->getAttr<OwnerAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    return true;
  }
  if (const auto *Second = dyn_cast<RandomizeLayoutAttr>(A)) {
    if (const auto *First = D->getAttr<NoRandomizeLayoutAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    return true;
  }
  if (const auto *Second = dyn_cast<SpeculativeLoadHardeningAttr>(A)) {
    if (const auto *First = D->getAttr<NoSpeculativeLoadHardeningAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    return true;
  }
  if (const auto *Second = dyn_cast<TargetAttr>(A)) {
    if (const auto *First = D->getAttr<TargetClonesAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    if (const auto *First = D->getAttr<TargetVersionAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    if (const auto *First = D->getAttr<CPUDispatchAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    if (const auto *First = D->getAttr<CPUSpecificAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    return true;
  }
  if (const auto *Second = dyn_cast<TargetClonesAttr>(A)) {
    if (const auto *First = D->getAttr<TargetVersionAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    if (const auto *First = D->getAttr<TargetAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    if (const auto *First = D->getAttr<CPUDispatchAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    if (const auto *First = D->getAttr<CPUSpecificAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    return true;
  }
  if (const auto *Second = dyn_cast<TargetVersionAttr>(A)) {
    if (const auto *First = D->getAttr<TargetClonesAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    if (const auto *First = D->getAttr<TargetAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    if (const auto *First = D->getAttr<CPUDispatchAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    if (const auto *First = D->getAttr<CPUSpecificAttr>()) {
      S.Diag(First->getLocation(), diag::err_attributes_are_not_compatible) << First << Second << (First->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
      S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
      return false;
    }
    return true;
  }
  return true;
}

#elif defined(WANT_STMT_MERGE_LOGIC)

static bool DiagnoseMutualExclusions(Sema &S, const SmallVectorImpl<const Attr *> &C) {
  for (const Attr *A : C) {
    if (const auto *Second = dyn_cast<AlwaysInlineAttr>(A)) {
      auto Iter = llvm::find_if(C, [](const Attr *Check) { return isa<NotTailCalledAttr>(Check); });
      if (Iter != C.end()) {
        S.Diag((*Iter)->getLocation(), diag::err_attributes_are_not_compatible) << *Iter << Second << ((*Iter)->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
        S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
        return false;
      }
    }
    if (const auto *Second = dyn_cast<LikelyAttr>(A)) {
      auto Iter = llvm::find_if(C, [](const Attr *Check) { return isa<UnlikelyAttr>(Check); });
      if (Iter != C.end()) {
        S.Diag((*Iter)->getLocation(), diag::err_attributes_are_not_compatible) << *Iter << Second << ((*Iter)->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
        S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
        return false;
      }
    }
    if (const auto *Second = dyn_cast<UnlikelyAttr>(A)) {
      auto Iter = llvm::find_if(C, [](const Attr *Check) { return isa<LikelyAttr>(Check); });
      if (Iter != C.end()) {
        S.Diag((*Iter)->getLocation(), diag::err_attributes_are_not_compatible) << *Iter << Second << ((*Iter)->isRegularKeywordAttribute() || Second->isRegularKeywordAttribute());
        S.Diag(Second->getLocation(), diag::note_conflicting_attribute);
        return false;
      }
    }
  }
  return true;
}

#endif
