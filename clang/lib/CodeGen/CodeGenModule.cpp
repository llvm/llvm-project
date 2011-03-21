//===--- CodeGenModule.cpp - Emit LLVM Code from ASTs for a Module --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This coordinates the per-module state used while generating code.
//
//===----------------------------------------------------------------------===//

#include "CodeGenModule.h"
#include "CGDebugInfo.h"
#include "CodeGenFunction.h"
#include "CodeGenTBAA.h"
#include "CGCall.h"
#include "CGCXXABI.h"
#include "CGObjCRuntime.h"
#include "TargetInfo.h"
#include "clang/Frontend/CodeGenOptions.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/RecordLayout.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/ConvertUTF.h"
#include "llvm/CallingConv.h"
#include "llvm/Module.h"
#include "llvm/Intrinsics.h"
#include "llvm/LLVMContext.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Target/Mangler.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/ErrorHandling.h"
using namespace clang;
using namespace CodeGen;

static CGCXXABI &createCXXABI(CodeGenModule &CGM) {
  switch (CGM.getContext().Target.getCXXABI()) {
  case CXXABI_ARM: return *CreateARMCXXABI(CGM);
  case CXXABI_Itanium: return *CreateItaniumCXXABI(CGM);
  case CXXABI_Microsoft: return *CreateMicrosoftCXXABI(CGM);
  }

  llvm_unreachable("invalid C++ ABI kind");
  return *CreateItaniumCXXABI(CGM);
}


CodeGenModule::CodeGenModule(ASTContext &C, const CodeGenOptions &CGO,
                             llvm::Module &M, const llvm::TargetData &TD,
                             Diagnostic &diags)
  : Context(C), Features(C.getLangOptions()), CodeGenOpts(CGO), TheModule(M),
    TheTargetData(TD), TheTargetCodeGenInfo(0), Diags(diags),
    ABI(createCXXABI(*this)), 
    Types(C, M, TD, getTargetCodeGenInfo().getABIInfo(), ABI),
    TBAA(0),
    VTables(*this), Runtime(0),
    CFConstantStringClassRef(0), ConstantStringClassRef(0),
    VMContext(M.getContext()),
    NSConcreteGlobalBlockDecl(0), NSConcreteStackBlockDecl(0),
    NSConcreteGlobalBlock(0), NSConcreteStackBlock(0),
    BlockObjectAssignDecl(0), BlockObjectDisposeDecl(0),
    BlockObjectAssign(0), BlockObjectDispose(0),
    BlockDescriptorType(0), GenericBlockLiteralType(0) {
  if (!Features.ObjC1)
    Runtime = 0;
  else if (!Features.NeXTRuntime)
    Runtime = CreateGNUObjCRuntime(*this);
  else if (Features.ObjCNonFragileABI)
    Runtime = CreateMacNonFragileABIObjCRuntime(*this);
  else
    Runtime = CreateMacObjCRuntime(*this);

  // Enable TBAA unless it's suppressed.
  if (!CodeGenOpts.RelaxedAliasing && CodeGenOpts.OptimizationLevel > 0)
    TBAA = new CodeGenTBAA(Context, VMContext, getLangOptions(),
                           ABI.getMangleContext());

  // If debug info generation is enabled, create the CGDebugInfo object.
  DebugInfo = CodeGenOpts.DebugInfo ? new CGDebugInfo(*this) : 0;

  Block.GlobalUniqueCount = 0;

  // Initialize the type cache.
  llvm::LLVMContext &LLVMContext = M.getContext();
  Int8Ty  = llvm::Type::getInt8Ty(LLVMContext);
  Int32Ty  = llvm::Type::getInt32Ty(LLVMContext);
  Int64Ty  = llvm::Type::getInt64Ty(LLVMContext);
  PointerWidthInBits = C.Target.getPointerWidth(0);
  PointerAlignInBytes =
    C.toCharUnitsFromBits(C.Target.getPointerAlign(0)).getQuantity();
  IntTy = llvm::IntegerType::get(LLVMContext, C.Target.getIntWidth());
  IntPtrTy = llvm::IntegerType::get(LLVMContext, PointerWidthInBits);
  Int8PtrTy = Int8Ty->getPointerTo(0);
  Int8PtrPtrTy = Int8PtrTy->getPointerTo(0);
}

CodeGenModule::~CodeGenModule() {
  delete Runtime;
  delete &ABI;
  delete TBAA;
  delete DebugInfo;
}

void CodeGenModule::createObjCRuntime() {
  if (!Features.NeXTRuntime)
    Runtime = CreateGNUObjCRuntime(*this);
  else if (Features.ObjCNonFragileABI)
    Runtime = CreateMacNonFragileABIObjCRuntime(*this);
  else
    Runtime = CreateMacObjCRuntime(*this);
}

void CodeGenModule::Release() {
  EmitDeferred();
  EmitCXXGlobalInitFunc();
  EmitCXXGlobalDtorFunc();
  if (Runtime)
    if (llvm::Function *ObjCInitFunction = Runtime->ModuleInitFunction())
      AddGlobalCtor(ObjCInitFunction);
  EmitCtorList(GlobalCtors, "llvm.global_ctors");
  EmitCtorList(GlobalDtors, "llvm.global_dtors");
  EmitAnnotations();
  EmitLLVMUsed();

  SimplifyPersonality();

  if (getCodeGenOpts().EmitDeclMetadata)
    EmitDeclMetadata();
}

llvm::MDNode *CodeGenModule::getTBAAInfo(QualType QTy) {
  if (!TBAA)
    return 0;
  return TBAA->getTBAAInfo(QTy);
}

void CodeGenModule::DecorateInstruction(llvm::Instruction *Inst,
                                        llvm::MDNode *TBAAInfo) {
  Inst->setMetadata(llvm::LLVMContext::MD_tbaa, TBAAInfo);
}

bool CodeGenModule::isTargetDarwin() const {
  return getContext().Target.getTriple().getOS() == llvm::Triple::Darwin;
}

/// ErrorUnsupported - Print out an error that codegen doesn't support the
/// specified stmt yet.
void CodeGenModule::ErrorUnsupported(const Stmt *S, const char *Type,
                                     bool OmitOnError) {
  if (OmitOnError && getDiags().hasErrorOccurred())
    return;
  unsigned DiagID = getDiags().getCustomDiagID(Diagnostic::Error,
                                               "cannot compile this %0 yet");
  std::string Msg = Type;
  getDiags().Report(Context.getFullLoc(S->getLocStart()), DiagID)
    << Msg << S->getSourceRange();
}

/// ErrorUnsupported - Print out an error that codegen doesn't support the
/// specified decl yet.
void CodeGenModule::ErrorUnsupported(const Decl *D, const char *Type,
                                     bool OmitOnError) {
  if (OmitOnError && getDiags().hasErrorOccurred())
    return;
  unsigned DiagID = getDiags().getCustomDiagID(Diagnostic::Error,
                                               "cannot compile this %0 yet");
  std::string Msg = Type;
  getDiags().Report(Context.getFullLoc(D->getLocation()), DiagID) << Msg;
}

void CodeGenModule::setGlobalVisibility(llvm::GlobalValue *GV,
                                        const NamedDecl *D) const {
  // Internal definitions always have default visibility.
  if (GV->hasLocalLinkage()) {
    GV->setVisibility(llvm::GlobalValue::DefaultVisibility);
    return;
  }

  // Set visibility for definitions.
  NamedDecl::LinkageInfo LV = D->getLinkageAndVisibility();
  if (LV.visibilityExplicit() || !GV->hasAvailableExternallyLinkage())
    GV->setVisibility(GetLLVMVisibility(LV.visibility()));
}

/// Set the symbol visibility of type information (vtable and RTTI)
/// associated with the given type.
void CodeGenModule::setTypeVisibility(llvm::GlobalValue *GV,
                                      const CXXRecordDecl *RD,
                                      TypeVisibilityKind TVK) const {
  setGlobalVisibility(GV, RD);

  if (!CodeGenOpts.HiddenWeakVTables)
    return;

  // We never want to drop the visibility for RTTI names.
  if (TVK == TVK_ForRTTIName)
    return;

  // We want to drop the visibility to hidden for weak type symbols.
  // This isn't possible if there might be unresolved references
  // elsewhere that rely on this symbol being visible.

  // This should be kept roughly in sync with setThunkVisibility
  // in CGVTables.cpp.

  // Preconditions.
  if (GV->getLinkage() != llvm::GlobalVariable::LinkOnceODRLinkage ||
      GV->getVisibility() != llvm::GlobalVariable::DefaultVisibility)
    return;

  // Don't override an explicit visibility attribute.
  if (RD->hasAttr<VisibilityAttr>())
    return;

  switch (RD->getTemplateSpecializationKind()) {
  // We have to disable the optimization if this is an EI definition
  // because there might be EI declarations in other shared objects.
  case TSK_ExplicitInstantiationDefinition:
  case TSK_ExplicitInstantiationDeclaration:
    return;

  // Every use of a non-template class's type information has to emit it.
  case TSK_Undeclared:
    break;

  // In theory, implicit instantiations can ignore the possibility of
  // an explicit instantiation declaration because there necessarily
  // must be an EI definition somewhere with default visibility.  In
  // practice, it's possible to have an explicit instantiation for
  // an arbitrary template class, and linkers aren't necessarily able
  // to deal with mixed-visibility symbols.
  case TSK_ExplicitSpecialization:
  case TSK_ImplicitInstantiation:
    if (!CodeGenOpts.HiddenWeakTemplateVTables)
      return;
    break;
  }

  // If there's a key function, there may be translation units
  // that don't have the key function's definition.  But ignore
  // this if we're emitting RTTI under -fno-rtti.
  if (!(TVK != TVK_ForRTTI) || Features.RTTI) {
    if (Context.getKeyFunction(RD))
      return;
  }

  // Otherwise, drop the visibility to hidden.
  GV->setVisibility(llvm::GlobalValue::HiddenVisibility);
  GV->setUnnamedAddr(true);
}

llvm::StringRef CodeGenModule::getMangledName(GlobalDecl GD) {
  const NamedDecl *ND = cast<NamedDecl>(GD.getDecl());

  llvm::StringRef &Str = MangledDeclNames[GD.getCanonicalDecl()];
  if (!Str.empty())
    return Str;

  if (!getCXXABI().getMangleContext().shouldMangleDeclName(ND)) {
    IdentifierInfo *II = ND->getIdentifier();
    assert(II && "Attempt to mangle unnamed decl.");

    Str = II->getName();
    return Str;
  }
  
  llvm::SmallString<256> Buffer;
  llvm::raw_svector_ostream Out(Buffer);
  if (const CXXConstructorDecl *D = dyn_cast<CXXConstructorDecl>(ND))
    getCXXABI().getMangleContext().mangleCXXCtor(D, GD.getCtorType(), Out);
  else if (const CXXDestructorDecl *D = dyn_cast<CXXDestructorDecl>(ND))
    getCXXABI().getMangleContext().mangleCXXDtor(D, GD.getDtorType(), Out);
  else if (const BlockDecl *BD = dyn_cast<BlockDecl>(ND))
    getCXXABI().getMangleContext().mangleBlock(BD, Out);
  else
    getCXXABI().getMangleContext().mangleName(ND, Out);

  // Allocate space for the mangled name.
  Out.flush();
  size_t Length = Buffer.size();
  char *Name = MangledNamesAllocator.Allocate<char>(Length);
  std::copy(Buffer.begin(), Buffer.end(), Name);
  
  Str = llvm::StringRef(Name, Length);
  
  return Str;
}

void CodeGenModule::getBlockMangledName(GlobalDecl GD, MangleBuffer &Buffer,
                                        const BlockDecl *BD) {
  MangleContext &MangleCtx = getCXXABI().getMangleContext();
  const Decl *D = GD.getDecl();
  llvm::raw_svector_ostream Out(Buffer.getBuffer());
  if (D == 0)
    MangleCtx.mangleGlobalBlock(BD, Out);
  else if (const CXXConstructorDecl *CD = dyn_cast<CXXConstructorDecl>(D))
    MangleCtx.mangleCtorBlock(CD, GD.getCtorType(), BD, Out);
  else if (const CXXDestructorDecl *DD = dyn_cast<CXXDestructorDecl>(D))
    MangleCtx.mangleDtorBlock(DD, GD.getDtorType(), BD, Out);
  else
    MangleCtx.mangleBlock(cast<DeclContext>(D), BD, Out);
}

llvm::GlobalValue *CodeGenModule::GetGlobalValue(llvm::StringRef Name) {
  return getModule().getNamedValue(Name);
}

/// AddGlobalCtor - Add a function to the list that will be called before
/// main() runs.
void CodeGenModule::AddGlobalCtor(llvm::Function * Ctor, int Priority) {
  // FIXME: Type coercion of void()* types.
  GlobalCtors.push_back(std::make_pair(Ctor, Priority));
}

/// AddGlobalDtor - Add a function to the list that will be called
/// when the module is unloaded.
void CodeGenModule::AddGlobalDtor(llvm::Function * Dtor, int Priority) {
  // FIXME: Type coercion of void()* types.
  GlobalDtors.push_back(std::make_pair(Dtor, Priority));
}

void CodeGenModule::EmitCtorList(const CtorList &Fns, const char *GlobalName) {
  // Ctor function type is void()*.
  llvm::FunctionType* CtorFTy =
    llvm::FunctionType::get(llvm::Type::getVoidTy(VMContext), false);
  llvm::Type *CtorPFTy = llvm::PointerType::getUnqual(CtorFTy);

  // Get the type of a ctor entry, { i32, void ()* }.
  llvm::StructType* CtorStructTy =
    llvm::StructType::get(VMContext, llvm::Type::getInt32Ty(VMContext),
                          llvm::PointerType::getUnqual(CtorFTy), NULL);

  // Construct the constructor and destructor arrays.
  std::vector<llvm::Constant*> Ctors;
  for (CtorList::const_iterator I = Fns.begin(), E = Fns.end(); I != E; ++I) {
    std::vector<llvm::Constant*> S;
    S.push_back(llvm::ConstantInt::get(llvm::Type::getInt32Ty(VMContext),
                I->second, false));
    S.push_back(llvm::ConstantExpr::getBitCast(I->first, CtorPFTy));
    Ctors.push_back(llvm::ConstantStruct::get(CtorStructTy, S));
  }

  if (!Ctors.empty()) {
    llvm::ArrayType *AT = llvm::ArrayType::get(CtorStructTy, Ctors.size());
    new llvm::GlobalVariable(TheModule, AT, false,
                             llvm::GlobalValue::AppendingLinkage,
                             llvm::ConstantArray::get(AT, Ctors),
                             GlobalName);
  }
}

void CodeGenModule::EmitAnnotations() {
  if (Annotations.empty())
    return;

  // Create a new global variable for the ConstantStruct in the Module.
  llvm::Constant *Array =
  llvm::ConstantArray::get(llvm::ArrayType::get(Annotations[0]->getType(),
                                                Annotations.size()),
                           Annotations);
  llvm::GlobalValue *gv =
  new llvm::GlobalVariable(TheModule, Array->getType(), false,
                           llvm::GlobalValue::AppendingLinkage, Array,
                           "llvm.global.annotations");
  gv->setSection("llvm.metadata");
}

llvm::GlobalValue::LinkageTypes
CodeGenModule::getFunctionLinkage(const FunctionDecl *D) {
  GVALinkage Linkage = getContext().GetGVALinkageForFunction(D);

  if (Linkage == GVA_Internal)
    return llvm::Function::InternalLinkage;
  
  if (D->hasAttr<DLLExportAttr>())
    return llvm::Function::DLLExportLinkage;
  
  if (D->hasAttr<WeakAttr>())
    return llvm::Function::WeakAnyLinkage;
  
  // In C99 mode, 'inline' functions are guaranteed to have a strong
  // definition somewhere else, so we can use available_externally linkage.
  if (Linkage == GVA_C99Inline)
    return llvm::Function::AvailableExternallyLinkage;
  
  // In C++, the compiler has to emit a definition in every translation unit
  // that references the function.  We should use linkonce_odr because
  // a) if all references in this translation unit are optimized away, we
  // don't need to codegen it.  b) if the function persists, it needs to be
  // merged with other definitions. c) C++ has the ODR, so we know the
  // definition is dependable.
  if (Linkage == GVA_CXXInline || Linkage == GVA_TemplateInstantiation)
    return !Context.getLangOptions().AppleKext 
             ? llvm::Function::LinkOnceODRLinkage 
             : llvm::Function::InternalLinkage;
  
  // An explicit instantiation of a template has weak linkage, since
  // explicit instantiations can occur in multiple translation units
  // and must all be equivalent. However, we are not allowed to
  // throw away these explicit instantiations.
  if (Linkage == GVA_ExplicitTemplateInstantiation)
    return !Context.getLangOptions().AppleKext
             ? llvm::Function::WeakODRLinkage
             : llvm::Function::InternalLinkage;
  
  // Otherwise, we have strong external linkage.
  assert(Linkage == GVA_StrongExternal);
  return llvm::Function::ExternalLinkage;
}


/// SetFunctionDefinitionAttributes - Set attributes for a global.
///
/// FIXME: This is currently only done for aliases and functions, but not for
/// variables (these details are set in EmitGlobalVarDefinition for variables).
void CodeGenModule::SetFunctionDefinitionAttributes(const FunctionDecl *D,
                                                    llvm::GlobalValue *GV) {
  SetCommonAttributes(D, GV);
}

void CodeGenModule::SetLLVMFunctionAttributes(const Decl *D,
                                              const CGFunctionInfo &Info,
                                              llvm::Function *F) {
  unsigned CallingConv;
  AttributeListType AttributeList;
  ConstructAttributeList(Info, D, AttributeList, CallingConv);
  F->setAttributes(llvm::AttrListPtr::get(AttributeList.begin(),
                                          AttributeList.size()));
  F->setCallingConv(static_cast<llvm::CallingConv::ID>(CallingConv));
}

void CodeGenModule::SetLLVMFunctionAttributesForDefinition(const Decl *D,
                                                           llvm::Function *F) {
  if (!Features.Exceptions && !Features.ObjCNonFragileABI)
    F->addFnAttr(llvm::Attribute::NoUnwind);

  if (D->hasAttr<AlwaysInlineAttr>())
    F->addFnAttr(llvm::Attribute::AlwaysInline);

  if (D->hasAttr<NakedAttr>())
    F->addFnAttr(llvm::Attribute::Naked);

  if (D->hasAttr<NoInlineAttr>())
    F->addFnAttr(llvm::Attribute::NoInline);

  if (isa<CXXConstructorDecl>(D) || isa<CXXDestructorDecl>(D))
    F->setUnnamedAddr(true);

  if (Features.getStackProtectorMode() == LangOptions::SSPOn)
    F->addFnAttr(llvm::Attribute::StackProtect);
  else if (Features.getStackProtectorMode() == LangOptions::SSPReq)
    F->addFnAttr(llvm::Attribute::StackProtectReq);
  
  unsigned alignment = D->getMaxAlignment() / Context.getCharWidth();
  if (alignment)
    F->setAlignment(alignment);

  // C++ ABI requires 2-byte alignment for member functions.
  if (F->getAlignment() < 2 && isa<CXXMethodDecl>(D))
    F->setAlignment(2);
}

void CodeGenModule::SetCommonAttributes(const Decl *D,
                                        llvm::GlobalValue *GV) {
  if (const NamedDecl *ND = dyn_cast<NamedDecl>(D))
    setGlobalVisibility(GV, ND);
  else
    GV->setVisibility(llvm::GlobalValue::DefaultVisibility);

  if (D->hasAttr<UsedAttr>())
    AddUsedGlobal(GV);

  if (const SectionAttr *SA = D->getAttr<SectionAttr>())
    GV->setSection(SA->getName());

  getTargetCodeGenInfo().SetTargetAttributes(D, GV, *this);
}

void CodeGenModule::SetInternalFunctionAttributes(const Decl *D,
                                                  llvm::Function *F,
                                                  const CGFunctionInfo &FI) {
  SetLLVMFunctionAttributes(D, FI, F);
  SetLLVMFunctionAttributesForDefinition(D, F);

  F->setLinkage(llvm::Function::InternalLinkage);

  SetCommonAttributes(D, F);
}

void CodeGenModule::SetFunctionAttributes(GlobalDecl GD,
                                          llvm::Function *F,
                                          bool IsIncompleteFunction) {
  const FunctionDecl *FD = cast<FunctionDecl>(GD.getDecl());

  if (!IsIncompleteFunction)
    SetLLVMFunctionAttributes(FD, getTypes().getFunctionInfo(GD), F);

  // Only a few attributes are set on declarations; these may later be
  // overridden by a definition.

  if (FD->hasAttr<DLLImportAttr>()) {
    F->setLinkage(llvm::Function::DLLImportLinkage);
  } else if (FD->hasAttr<WeakAttr>() ||
             FD->hasAttr<WeakImportAttr>()) {
    // "extern_weak" is overloaded in LLVM; we probably should have
    // separate linkage types for this.
    F->setLinkage(llvm::Function::ExternalWeakLinkage);
  } else {
    F->setLinkage(llvm::Function::ExternalLinkage);

    NamedDecl::LinkageInfo LV = FD->getLinkageAndVisibility();
    if (LV.linkage() == ExternalLinkage && LV.visibilityExplicit()) {
      F->setVisibility(GetLLVMVisibility(LV.visibility()));
    }
  }

  if (const SectionAttr *SA = FD->getAttr<SectionAttr>())
    F->setSection(SA->getName());
}

void CodeGenModule::AddUsedGlobal(llvm::GlobalValue *GV) {
  assert(!GV->isDeclaration() &&
         "Only globals with definition can force usage.");
  LLVMUsed.push_back(GV);
}

void CodeGenModule::EmitLLVMUsed() {
  // Don't create llvm.used if there is no need.
  if (LLVMUsed.empty())
    return;

  const llvm::Type *i8PTy = llvm::Type::getInt8PtrTy(VMContext);

  // Convert LLVMUsed to what ConstantArray needs.
  std::vector<llvm::Constant*> UsedArray;
  UsedArray.resize(LLVMUsed.size());
  for (unsigned i = 0, e = LLVMUsed.size(); i != e; ++i) {
    UsedArray[i] =
     llvm::ConstantExpr::getBitCast(cast<llvm::Constant>(&*LLVMUsed[i]),
                                      i8PTy);
  }

  if (UsedArray.empty())
    return;
  llvm::ArrayType *ATy = llvm::ArrayType::get(i8PTy, UsedArray.size());

  llvm::GlobalVariable *GV =
    new llvm::GlobalVariable(getModule(), ATy, false,
                             llvm::GlobalValue::AppendingLinkage,
                             llvm::ConstantArray::get(ATy, UsedArray),
                             "llvm.used");

  GV->setSection("llvm.metadata");
}

void CodeGenModule::EmitDeferred() {
  // Emit code for any potentially referenced deferred decls.  Since a
  // previously unused static decl may become used during the generation of code
  // for a static function, iterate until no  changes are made.

  while (!DeferredDeclsToEmit.empty() || !DeferredVTables.empty()) {
    if (!DeferredVTables.empty()) {
      const CXXRecordDecl *RD = DeferredVTables.back();
      DeferredVTables.pop_back();
      getVTables().GenerateClassData(getVTableLinkage(RD), RD);
      continue;
    }

    GlobalDecl D = DeferredDeclsToEmit.back();
    DeferredDeclsToEmit.pop_back();

    // Check to see if we've already emitted this.  This is necessary
    // for a couple of reasons: first, decls can end up in the
    // deferred-decls queue multiple times, and second, decls can end
    // up with definitions in unusual ways (e.g. by an extern inline
    // function acquiring a strong function redefinition).  Just
    // ignore these cases.
    //
    // TODO: That said, looking this up multiple times is very wasteful.
    llvm::StringRef Name = getMangledName(D);
    llvm::GlobalValue *CGRef = GetGlobalValue(Name);
    assert(CGRef && "Deferred decl wasn't referenced?");

    if (!CGRef->isDeclaration())
      continue;

    // GlobalAlias::isDeclaration() defers to the aliasee, but for our
    // purposes an alias counts as a definition.
    if (isa<llvm::GlobalAlias>(CGRef))
      continue;

    // Otherwise, emit the definition and move on to the next one.
    EmitGlobalDefinition(D);
  }
}

/// EmitAnnotateAttr - Generate the llvm::ConstantStruct which contains the
/// annotation information for a given GlobalValue.  The annotation struct is
/// {i8 *, i8 *, i8 *, i32}.  The first field is a constant expression, the
/// GlobalValue being annotated.  The second field is the constant string
/// created from the AnnotateAttr's annotation.  The third field is a constant
/// string containing the name of the translation unit.  The fourth field is
/// the line number in the file of the annotated value declaration.
///
/// FIXME: this does not unique the annotation string constants, as llvm-gcc
///        appears to.
///
llvm::Constant *CodeGenModule::EmitAnnotateAttr(llvm::GlobalValue *GV,
                                                const AnnotateAttr *AA,
                                                unsigned LineNo) {
  llvm::Module *M = &getModule();

  // get [N x i8] constants for the annotation string, and the filename string
  // which are the 2nd and 3rd elements of the global annotation structure.
  const llvm::Type *SBP = llvm::Type::getInt8PtrTy(VMContext);
  llvm::Constant *anno = llvm::ConstantArray::get(VMContext,
                                                  AA->getAnnotation(), true);
  llvm::Constant *unit = llvm::ConstantArray::get(VMContext,
                                                  M->getModuleIdentifier(),
                                                  true);

  // Get the two global values corresponding to the ConstantArrays we just
  // created to hold the bytes of the strings.
  llvm::GlobalValue *annoGV =
    new llvm::GlobalVariable(*M, anno->getType(), false,
                             llvm::GlobalValue::PrivateLinkage, anno,
                             GV->getName());
  // translation unit name string, emitted into the llvm.metadata section.
  llvm::GlobalValue *unitGV =
    new llvm::GlobalVariable(*M, unit->getType(), false,
                             llvm::GlobalValue::PrivateLinkage, unit,
                             ".str");
  unitGV->setUnnamedAddr(true);

  // Create the ConstantStruct for the global annotation.
  llvm::Constant *Fields[4] = {
    llvm::ConstantExpr::getBitCast(GV, SBP),
    llvm::ConstantExpr::getBitCast(annoGV, SBP),
    llvm::ConstantExpr::getBitCast(unitGV, SBP),
    llvm::ConstantInt::get(llvm::Type::getInt32Ty(VMContext), LineNo)
  };
  return llvm::ConstantStruct::get(VMContext, Fields, 4, false);
}

bool CodeGenModule::MayDeferGeneration(const ValueDecl *Global) {
  // Never defer when EmitAllDecls is specified.
  if (Features.EmitAllDecls)
    return false;

  return !getContext().DeclMustBeEmitted(Global);
}

llvm::Constant *CodeGenModule::GetWeakRefReference(const ValueDecl *VD) {
  const AliasAttr *AA = VD->getAttr<AliasAttr>();
  assert(AA && "No alias?");

  const llvm::Type *DeclTy = getTypes().ConvertTypeForMem(VD->getType());

  // See if there is already something with the target's name in the module.
  llvm::GlobalValue *Entry = GetGlobalValue(AA->getAliasee());

  llvm::Constant *Aliasee;
  if (isa<llvm::FunctionType>(DeclTy))
    Aliasee = GetOrCreateLLVMFunction(AA->getAliasee(), DeclTy, GlobalDecl(),
                                      /*ForVTable=*/false);
  else
    Aliasee = GetOrCreateLLVMGlobal(AA->getAliasee(),
                                    llvm::PointerType::getUnqual(DeclTy), 0);
  if (!Entry) {
    llvm::GlobalValue* F = cast<llvm::GlobalValue>(Aliasee);
    F->setLinkage(llvm::Function::ExternalWeakLinkage);    
    WeakRefReferences.insert(F);
  }

  return Aliasee;
}

void CodeGenModule::EmitGlobal(GlobalDecl GD) {
  const ValueDecl *Global = cast<ValueDecl>(GD.getDecl());

  // Weak references don't produce any output by themselves.
  if (Global->hasAttr<WeakRefAttr>())
    return;

  // If this is an alias definition (which otherwise looks like a declaration)
  // emit it now.
  if (Global->hasAttr<AliasAttr>())
    return EmitAliasDefinition(GD);

  // Ignore declarations, they will be emitted on their first use.
  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(Global)) {
    if (FD->getIdentifier()) {
      llvm::StringRef Name = FD->getName();
      if (Name == "_Block_object_assign") {
        BlockObjectAssignDecl = FD;
      } else if (Name == "_Block_object_dispose") {
        BlockObjectDisposeDecl = FD;
      }
    }

    // Forward declarations are emitted lazily on first use.
    if (!FD->isThisDeclarationADefinition())
      return;
  } else {
    const VarDecl *VD = cast<VarDecl>(Global);
    assert(VD->isFileVarDecl() && "Cannot emit local var decl as global.");

    if (VD->getIdentifier()) {
      llvm::StringRef Name = VD->getName();
      if (Name == "_NSConcreteGlobalBlock") {
        NSConcreteGlobalBlockDecl = VD;
      } else if (Name == "_NSConcreteStackBlock") {
        NSConcreteStackBlockDecl = VD;
      }
    }


    if (VD->isThisDeclarationADefinition() != VarDecl::Definition)
      return;
  }

  // Defer code generation when possible if this is a static definition, inline
  // function etc.  These we only want to emit if they are used.
  if (!MayDeferGeneration(Global)) {
    // Emit the definition if it can't be deferred.
    EmitGlobalDefinition(GD);
    return;
  }

  // If we're deferring emission of a C++ variable with an
  // initializer, remember the order in which it appeared in the file.
  if (getLangOptions().CPlusPlus && isa<VarDecl>(Global) &&
      cast<VarDecl>(Global)->hasInit()) {
    DelayedCXXInitPosition[Global] = CXXGlobalInits.size();
    CXXGlobalInits.push_back(0);
  }
  
  // If the value has already been used, add it directly to the
  // DeferredDeclsToEmit list.
  llvm::StringRef MangledName = getMangledName(GD);
  if (GetGlobalValue(MangledName))
    DeferredDeclsToEmit.push_back(GD);
  else {
    // Otherwise, remember that we saw a deferred decl with this name.  The
    // first use of the mangled name will cause it to move into
    // DeferredDeclsToEmit.
    DeferredDecls[MangledName] = GD;
  }
}

void CodeGenModule::EmitGlobalDefinition(GlobalDecl GD) {
  const ValueDecl *D = cast<ValueDecl>(GD.getDecl());

  PrettyStackTraceDecl CrashInfo(const_cast<ValueDecl *>(D), D->getLocation(), 
                                 Context.getSourceManager(),
                                 "Generating code for declaration");
  
  if (const FunctionDecl *Function = dyn_cast<FunctionDecl>(D)) {
    // At -O0, don't generate IR for functions with available_externally 
    // linkage.
    if (CodeGenOpts.OptimizationLevel == 0 && 
        !Function->hasAttr<AlwaysInlineAttr>() &&
        getFunctionLinkage(Function) 
                                  == llvm::Function::AvailableExternallyLinkage)
      return;

    if (const CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(D)) {
      if (Method->isVirtual())
        getVTables().EmitThunks(GD);

      if (const CXXConstructorDecl *CD = dyn_cast<CXXConstructorDecl>(Method))
        return EmitCXXConstructor(CD, GD.getCtorType());
  
      if (const CXXDestructorDecl *DD = dyn_cast<CXXDestructorDecl>(Method))
        return EmitCXXDestructor(DD, GD.getDtorType());
    }

    return EmitGlobalFunctionDefinition(GD);
  }
  
  if (const VarDecl *VD = dyn_cast<VarDecl>(D))
    return EmitGlobalVarDefinition(VD);
  
  assert(0 && "Invalid argument to EmitGlobalDefinition()");
}

/// GetOrCreateLLVMFunction - If the specified mangled name is not in the
/// module, create and return an llvm Function with the specified type. If there
/// is something in the module with the specified name, return it potentially
/// bitcasted to the right type.
///
/// If D is non-null, it specifies a decl that correspond to this.  This is used
/// to set the attributes on the function when it is first created.
llvm::Constant *
CodeGenModule::GetOrCreateLLVMFunction(llvm::StringRef MangledName,
                                       const llvm::Type *Ty,
                                       GlobalDecl D, bool ForVTable) {
  // Lookup the entry, lazily creating it if necessary.
  llvm::GlobalValue *Entry = GetGlobalValue(MangledName);
  if (Entry) {
    if (WeakRefReferences.count(Entry)) {
      const FunctionDecl *FD = cast_or_null<FunctionDecl>(D.getDecl());
      if (FD && !FD->hasAttr<WeakAttr>())
        Entry->setLinkage(llvm::Function::ExternalLinkage);

      WeakRefReferences.erase(Entry);
    }

    if (Entry->getType()->getElementType() == Ty)
      return Entry;

    // Make sure the result is of the correct type.
    const llvm::Type *PTy = llvm::PointerType::getUnqual(Ty);
    return llvm::ConstantExpr::getBitCast(Entry, PTy);
  }

  // This function doesn't have a complete type (for example, the return
  // type is an incomplete struct). Use a fake type instead, and make
  // sure not to try to set attributes.
  bool IsIncompleteFunction = false;

  const llvm::FunctionType *FTy;
  if (isa<llvm::FunctionType>(Ty)) {
    FTy = cast<llvm::FunctionType>(Ty);
  } else {
    FTy = llvm::FunctionType::get(llvm::Type::getVoidTy(VMContext), false);
    IsIncompleteFunction = true;
  }
  
  llvm::Function *F = llvm::Function::Create(FTy,
                                             llvm::Function::ExternalLinkage,
                                             MangledName, &getModule());
  assert(F->getName() == MangledName && "name was uniqued!");
  if (D.getDecl())
    SetFunctionAttributes(D, F, IsIncompleteFunction);

  // This is the first use or definition of a mangled name.  If there is a
  // deferred decl with this name, remember that we need to emit it at the end
  // of the file.
  llvm::StringMap<GlobalDecl>::iterator DDI = DeferredDecls.find(MangledName);
  if (DDI != DeferredDecls.end()) {
    // Move the potentially referenced deferred decl to the DeferredDeclsToEmit
    // list, and remove it from DeferredDecls (since we don't need it anymore).
    DeferredDeclsToEmit.push_back(DDI->second);
    DeferredDecls.erase(DDI);

  // Otherwise, there are cases we have to worry about where we're
  // using a declaration for which we must emit a definition but where
  // we might not find a top-level definition:
  //   - member functions defined inline in their classes
  //   - friend functions defined inline in some class
  //   - special member functions with implicit definitions
  // If we ever change our AST traversal to walk into class methods,
  // this will be unnecessary.
  //
  // We also don't emit a definition for a function if it's going to be an entry
  // in a vtable, unless it's already marked as used.
  } else if (getLangOptions().CPlusPlus && D.getDecl()) {
    // Look for a declaration that's lexically in a record.
    const FunctionDecl *FD = cast<FunctionDecl>(D.getDecl());
    do {
      if (isa<CXXRecordDecl>(FD->getLexicalDeclContext())) {
        if (FD->isImplicit() && !ForVTable) {
          assert(FD->isUsed() && "Sema didn't mark implicit function as used!");
          DeferredDeclsToEmit.push_back(D.getWithDecl(FD));
          break;
        } else if (FD->isThisDeclarationADefinition()) {
          DeferredDeclsToEmit.push_back(D.getWithDecl(FD));
          break;
        }
      }
      FD = FD->getPreviousDeclaration();
    } while (FD);
  }

  // Make sure the result is of the requested type.
  if (!IsIncompleteFunction) {
    assert(F->getType()->getElementType() == Ty);
    return F;
  }

  const llvm::Type *PTy = llvm::PointerType::getUnqual(Ty);
  return llvm::ConstantExpr::getBitCast(F, PTy);
}

/// GetAddrOfFunction - Return the address of the given function.  If Ty is
/// non-null, then this function will use the specified type if it has to
/// create it (this occurs when we see a definition of the function).
llvm::Constant *CodeGenModule::GetAddrOfFunction(GlobalDecl GD,
                                                 const llvm::Type *Ty,
                                                 bool ForVTable) {
  // If there was no specific requested type, just convert it now.
  if (!Ty)
    Ty = getTypes().ConvertType(cast<ValueDecl>(GD.getDecl())->getType());
  
  llvm::StringRef MangledName = getMangledName(GD);
  return GetOrCreateLLVMFunction(MangledName, Ty, GD, ForVTable);
}

/// CreateRuntimeFunction - Create a new runtime function with the specified
/// type and name.
llvm::Constant *
CodeGenModule::CreateRuntimeFunction(const llvm::FunctionType *FTy,
                                     llvm::StringRef Name) {
  return GetOrCreateLLVMFunction(Name, FTy, GlobalDecl(), /*ForVTable=*/false);
}

static bool DeclIsConstantGlobal(ASTContext &Context, const VarDecl *D) {
  if (!D->getType().isConstant(Context) && !D->getType()->isReferenceType())
    return false;
  if (Context.getLangOptions().CPlusPlus &&
      Context.getBaseElementType(D->getType())->getAs<RecordType>()) {
    // FIXME: We should do something fancier here!
    return false;
  }
  return true;
}

/// GetOrCreateLLVMGlobal - If the specified mangled name is not in the module,
/// create and return an llvm GlobalVariable with the specified type.  If there
/// is something in the module with the specified name, return it potentially
/// bitcasted to the right type.
///
/// If D is non-null, it specifies a decl that correspond to this.  This is used
/// to set the attributes on the global when it is first created.
llvm::Constant *
CodeGenModule::GetOrCreateLLVMGlobal(llvm::StringRef MangledName,
                                     const llvm::PointerType *Ty,
                                     const VarDecl *D,
                                     bool UnnamedAddr) {
  // Lookup the entry, lazily creating it if necessary.
  llvm::GlobalValue *Entry = GetGlobalValue(MangledName);
  if (Entry) {
    if (WeakRefReferences.count(Entry)) {
      if (D && !D->hasAttr<WeakAttr>())
        Entry->setLinkage(llvm::Function::ExternalLinkage);

      WeakRefReferences.erase(Entry);
    }

    if (UnnamedAddr)
      Entry->setUnnamedAddr(true);

    if (Entry->getType() == Ty)
      return Entry;

    // Make sure the result is of the correct type.
    return llvm::ConstantExpr::getBitCast(Entry, Ty);
  }

  // This is the first use or definition of a mangled name.  If there is a
  // deferred decl with this name, remember that we need to emit it at the end
  // of the file.
  llvm::StringMap<GlobalDecl>::iterator DDI = DeferredDecls.find(MangledName);
  if (DDI != DeferredDecls.end()) {
    // Move the potentially referenced deferred decl to the DeferredDeclsToEmit
    // list, and remove it from DeferredDecls (since we don't need it anymore).
    DeferredDeclsToEmit.push_back(DDI->second);
    DeferredDecls.erase(DDI);
  }

  llvm::GlobalVariable *GV =
    new llvm::GlobalVariable(getModule(), Ty->getElementType(), false,
                             llvm::GlobalValue::ExternalLinkage,
                             0, MangledName, 0,
                             false, Ty->getAddressSpace());

  // Handle things which are present even on external declarations.
  if (D) {
    // FIXME: This code is overly simple and should be merged with other global
    // handling.
    GV->setConstant(DeclIsConstantGlobal(Context, D));

    // Set linkage and visibility in case we never see a definition.
    NamedDecl::LinkageInfo LV = D->getLinkageAndVisibility();
    if (LV.linkage() != ExternalLinkage) {
      // Don't set internal linkage on declarations.
    } else {
      if (D->hasAttr<DLLImportAttr>())
        GV->setLinkage(llvm::GlobalValue::DLLImportLinkage);
      else if (D->hasAttr<WeakAttr>() || D->hasAttr<WeakImportAttr>())
        GV->setLinkage(llvm::GlobalValue::ExternalWeakLinkage);

      // Set visibility on a declaration only if it's explicit.
      if (LV.visibilityExplicit())
        GV->setVisibility(GetLLVMVisibility(LV.visibility()));
    }

    GV->setThreadLocal(D->isThreadSpecified());
  }

  return GV;
}


llvm::GlobalVariable *
CodeGenModule::CreateOrReplaceCXXRuntimeVariable(llvm::StringRef Name, 
                                      const llvm::Type *Ty,
                                      llvm::GlobalValue::LinkageTypes Linkage) {
  llvm::GlobalVariable *GV = getModule().getNamedGlobal(Name);
  llvm::GlobalVariable *OldGV = 0;

  
  if (GV) {
    // Check if the variable has the right type.
    if (GV->getType()->getElementType() == Ty)
      return GV;

    // Because C++ name mangling, the only way we can end up with an already
    // existing global with the same name is if it has been declared extern "C".
      assert(GV->isDeclaration() && "Declaration has wrong type!");
    OldGV = GV;
  }
  
  // Create a new variable.
  GV = new llvm::GlobalVariable(getModule(), Ty, /*isConstant=*/true,
                                Linkage, 0, Name);
  
  if (OldGV) {
    // Replace occurrences of the old variable if needed.
    GV->takeName(OldGV);
    
    if (!OldGV->use_empty()) {
      llvm::Constant *NewPtrForOldDecl =
      llvm::ConstantExpr::getBitCast(GV, OldGV->getType());
      OldGV->replaceAllUsesWith(NewPtrForOldDecl);
    }
    
    OldGV->eraseFromParent();
  }
  
  return GV;
}

/// GetAddrOfGlobalVar - Return the llvm::Constant for the address of the
/// given global variable.  If Ty is non-null and if the global doesn't exist,
/// then it will be greated with the specified type instead of whatever the
/// normal requested type would be.
llvm::Constant *CodeGenModule::GetAddrOfGlobalVar(const VarDecl *D,
                                                  const llvm::Type *Ty) {
  assert(D->hasGlobalStorage() && "Not a global variable");
  QualType ASTTy = D->getType();
  if (Ty == 0)
    Ty = getTypes().ConvertTypeForMem(ASTTy);

  const llvm::PointerType *PTy =
    llvm::PointerType::get(Ty, ASTTy.getAddressSpace());

  llvm::StringRef MangledName = getMangledName(D);
  return GetOrCreateLLVMGlobal(MangledName, PTy, D);
}

/// CreateRuntimeVariable - Create a new runtime global variable with the
/// specified type and name.
llvm::Constant *
CodeGenModule::CreateRuntimeVariable(const llvm::Type *Ty,
                                     llvm::StringRef Name) {
  return GetOrCreateLLVMGlobal(Name,  llvm::PointerType::getUnqual(Ty), 0,
                               true);
}

void CodeGenModule::EmitTentativeDefinition(const VarDecl *D) {
  assert(!D->getInit() && "Cannot emit definite definitions here!");

  if (MayDeferGeneration(D)) {
    // If we have not seen a reference to this variable yet, place it
    // into the deferred declarations table to be emitted if needed
    // later.
    llvm::StringRef MangledName = getMangledName(D);
    if (!GetGlobalValue(MangledName)) {
      DeferredDecls[MangledName] = D;
      return;
    }
  }

  // The tentative definition is the only definition.
  EmitGlobalVarDefinition(D);
}

void CodeGenModule::EmitVTable(CXXRecordDecl *Class, bool DefinitionRequired) {
  if (DefinitionRequired)
    getVTables().GenerateClassData(getVTableLinkage(Class), Class);
}

llvm::GlobalVariable::LinkageTypes 
CodeGenModule::getVTableLinkage(const CXXRecordDecl *RD) {
  if (RD->isInAnonymousNamespace() || !RD->hasLinkage())
    return llvm::GlobalVariable::InternalLinkage;

  if (const CXXMethodDecl *KeyFunction
                                    = RD->getASTContext().getKeyFunction(RD)) {
    // If this class has a key function, use that to determine the linkage of
    // the vtable.
    const FunctionDecl *Def = 0;
    if (KeyFunction->hasBody(Def))
      KeyFunction = cast<CXXMethodDecl>(Def);
    
    switch (KeyFunction->getTemplateSpecializationKind()) {
      case TSK_Undeclared:
      case TSK_ExplicitSpecialization:
        // When compiling with optimizations turned on, we emit all vtables,
        // even if the key function is not defined in the current translation
        // unit. If this is the case, use available_externally linkage.
        if (!Def && CodeGenOpts.OptimizationLevel)
          return llvm::GlobalVariable::AvailableExternallyLinkage;

        if (KeyFunction->isInlined())
          return !Context.getLangOptions().AppleKext ?
                   llvm::GlobalVariable::LinkOnceODRLinkage :
                   llvm::Function::InternalLinkage;
        
        return llvm::GlobalVariable::ExternalLinkage;
        
      case TSK_ImplicitInstantiation:
        return !Context.getLangOptions().AppleKext ?
                 llvm::GlobalVariable::LinkOnceODRLinkage :
                 llvm::Function::InternalLinkage;

      case TSK_ExplicitInstantiationDefinition:
        return !Context.getLangOptions().AppleKext ?
                 llvm::GlobalVariable::WeakODRLinkage :
                 llvm::Function::InternalLinkage;
  
      case TSK_ExplicitInstantiationDeclaration:
        // FIXME: Use available_externally linkage. However, this currently
        // breaks LLVM's build due to undefined symbols.
        //      return llvm::GlobalVariable::AvailableExternallyLinkage;
        return !Context.getLangOptions().AppleKext ?
                 llvm::GlobalVariable::LinkOnceODRLinkage :
                 llvm::Function::InternalLinkage;
    }
  }
  
  if (Context.getLangOptions().AppleKext)
    return llvm::Function::InternalLinkage;
  
  switch (RD->getTemplateSpecializationKind()) {
  case TSK_Undeclared:
  case TSK_ExplicitSpecialization:
  case TSK_ImplicitInstantiation:
    // FIXME: Use available_externally linkage. However, this currently
    // breaks LLVM's build due to undefined symbols.
    //   return llvm::GlobalVariable::AvailableExternallyLinkage;
  case TSK_ExplicitInstantiationDeclaration:
    return llvm::GlobalVariable::LinkOnceODRLinkage;

  case TSK_ExplicitInstantiationDefinition:
      return llvm::GlobalVariable::WeakODRLinkage;
  }
  
  // Silence GCC warning.
  return llvm::GlobalVariable::LinkOnceODRLinkage;
}

CharUnits CodeGenModule::GetTargetTypeStoreSize(const llvm::Type *Ty) const {
    return Context.toCharUnitsFromBits(
      TheTargetData.getTypeStoreSizeInBits(Ty));
}

void CodeGenModule::EmitGlobalVarDefinition(const VarDecl *D) {
  llvm::Constant *Init = 0;
  QualType ASTTy = D->getType();
  bool NonConstInit = false;

  const Expr *InitExpr = D->getAnyInitializer();
  
  if (!InitExpr) {
    // This is a tentative definition; tentative definitions are
    // implicitly initialized with { 0 }.
    //
    // Note that tentative definitions are only emitted at the end of
    // a translation unit, so they should never have incomplete
    // type. In addition, EmitTentativeDefinition makes sure that we
    // never attempt to emit a tentative definition if a real one
    // exists. A use may still exists, however, so we still may need
    // to do a RAUW.
    assert(!ASTTy->isIncompleteType() && "Unexpected incomplete type");
    Init = EmitNullConstant(D->getType());
  } else {
    Init = EmitConstantExpr(InitExpr, D->getType());       
    if (!Init) {
      QualType T = InitExpr->getType();
      if (D->getType()->isReferenceType())
        T = D->getType();
      
      if (getLangOptions().CPlusPlus) {
        Init = EmitNullConstant(T);
        NonConstInit = true;
      } else {
        ErrorUnsupported(D, "static initializer");
        Init = llvm::UndefValue::get(getTypes().ConvertType(T));
      }
    } else {
      // We don't need an initializer, so remove the entry for the delayed
      // initializer position (just in case this entry was delayed).
      if (getLangOptions().CPlusPlus)
        DelayedCXXInitPosition.erase(D);
    }
  }

  const llvm::Type* InitType = Init->getType();
  llvm::Constant *Entry = GetAddrOfGlobalVar(D, InitType);

  // Strip off a bitcast if we got one back.
  if (llvm::ConstantExpr *CE = dyn_cast<llvm::ConstantExpr>(Entry)) {
    assert(CE->getOpcode() == llvm::Instruction::BitCast ||
           // all zero index gep.
           CE->getOpcode() == llvm::Instruction::GetElementPtr);
    Entry = CE->getOperand(0);
  }

  // Entry is now either a Function or GlobalVariable.
  llvm::GlobalVariable *GV = dyn_cast<llvm::GlobalVariable>(Entry);

  // We have a definition after a declaration with the wrong type.
  // We must make a new GlobalVariable* and update everything that used OldGV
  // (a declaration or tentative definition) with the new GlobalVariable*
  // (which will be a definition).
  //
  // This happens if there is a prototype for a global (e.g.
  // "extern int x[];") and then a definition of a different type (e.g.
  // "int x[10];"). This also happens when an initializer has a different type
  // from the type of the global (this happens with unions).
  if (GV == 0 ||
      GV->getType()->getElementType() != InitType ||
      GV->getType()->getAddressSpace() != ASTTy.getAddressSpace()) {

    // Move the old entry aside so that we'll create a new one.
    Entry->setName(llvm::StringRef());

    // Make a new global with the correct type, this is now guaranteed to work.
    GV = cast<llvm::GlobalVariable>(GetAddrOfGlobalVar(D, InitType));

    // Replace all uses of the old global with the new global
    llvm::Constant *NewPtrForOldDecl =
        llvm::ConstantExpr::getBitCast(GV, Entry->getType());
    Entry->replaceAllUsesWith(NewPtrForOldDecl);

    // Erase the old global, since it is no longer used.
    cast<llvm::GlobalValue>(Entry)->eraseFromParent();
  }

  if (const AnnotateAttr *AA = D->getAttr<AnnotateAttr>()) {
    SourceManager &SM = Context.getSourceManager();
    AddAnnotation(EmitAnnotateAttr(GV, AA,
                              SM.getInstantiationLineNumber(D->getLocation())));
  }

  GV->setInitializer(Init);

  // If it is safe to mark the global 'constant', do so now.
  GV->setConstant(false);
  if (!NonConstInit && DeclIsConstantGlobal(Context, D))
    GV->setConstant(true);

  GV->setAlignment(getContext().getDeclAlign(D).getQuantity());
  
  // Set the llvm linkage type as appropriate.
  llvm::GlobalValue::LinkageTypes Linkage = 
    GetLLVMLinkageVarDefinition(D, GV);
  GV->setLinkage(Linkage);
  if (Linkage == llvm::GlobalVariable::CommonLinkage)
    // common vars aren't constant even if declared const.
    GV->setConstant(false);

  SetCommonAttributes(D, GV);

  // Emit the initializer function if necessary.
  if (NonConstInit)
    EmitCXXGlobalVarDeclInitFunc(D, GV);

  // Emit global variable debug information.
  if (CGDebugInfo *DI = getModuleDebugInfo()) {
    DI->setLocation(D->getLocation());
    DI->EmitGlobalVariable(GV, D);
  }
}

llvm::GlobalValue::LinkageTypes
CodeGenModule::GetLLVMLinkageVarDefinition(const VarDecl *D,
                                           llvm::GlobalVariable *GV) {
  GVALinkage Linkage = getContext().GetGVALinkageForVariable(D);
  if (Linkage == GVA_Internal)
    return llvm::Function::InternalLinkage;
  else if (D->hasAttr<DLLImportAttr>())
    return llvm::Function::DLLImportLinkage;
  else if (D->hasAttr<DLLExportAttr>())
    return llvm::Function::DLLExportLinkage;
  else if (D->hasAttr<WeakAttr>()) {
    if (GV->isConstant())
      return llvm::GlobalVariable::WeakODRLinkage;
    else
      return llvm::GlobalVariable::WeakAnyLinkage;
  } else if (Linkage == GVA_TemplateInstantiation ||
             Linkage == GVA_ExplicitTemplateInstantiation)
    // FIXME: It seems like we can provide more specific linkage here
    // (LinkOnceODR, WeakODR).
    return llvm::GlobalVariable::WeakAnyLinkage;
  else if (!getLangOptions().CPlusPlus && 
           ((!CodeGenOpts.NoCommon && !D->getAttr<NoCommonAttr>()) ||
             D->getAttr<CommonAttr>()) &&
           !D->hasExternalStorage() && !D->getInit() &&
           !D->getAttr<SectionAttr>() && !D->isThreadSpecified()) {
    // Thread local vars aren't considered common linkage.
    return llvm::GlobalVariable::CommonLinkage;
  }
  return llvm::GlobalVariable::ExternalLinkage;
}

/// ReplaceUsesOfNonProtoTypeWithRealFunction - This function is called when we
/// implement a function with no prototype, e.g. "int foo() {}".  If there are
/// existing call uses of the old function in the module, this adjusts them to
/// call the new function directly.
///
/// This is not just a cleanup: the always_inline pass requires direct calls to
/// functions to be able to inline them.  If there is a bitcast in the way, it
/// won't inline them.  Instcombine normally deletes these calls, but it isn't
/// run at -O0.
static void ReplaceUsesOfNonProtoTypeWithRealFunction(llvm::GlobalValue *Old,
                                                      llvm::Function *NewFn) {
  // If we're redefining a global as a function, don't transform it.
  llvm::Function *OldFn = dyn_cast<llvm::Function>(Old);
  if (OldFn == 0) return;

  const llvm::Type *NewRetTy = NewFn->getReturnType();
  llvm::SmallVector<llvm::Value*, 4> ArgList;

  for (llvm::Value::use_iterator UI = OldFn->use_begin(), E = OldFn->use_end();
       UI != E; ) {
    // TODO: Do invokes ever occur in C code?  If so, we should handle them too.
    llvm::Value::use_iterator I = UI++; // Increment before the CI is erased.
    llvm::CallInst *CI = dyn_cast<llvm::CallInst>(*I);
    if (!CI) continue; // FIXME: when we allow Invoke, just do CallSite CS(*I)
    llvm::CallSite CS(CI);
    if (!CI || !CS.isCallee(I)) continue;

    // If the return types don't match exactly, and if the call isn't dead, then
    // we can't transform this call.
    if (CI->getType() != NewRetTy && !CI->use_empty())
      continue;

    // If the function was passed too few arguments, don't transform.  If extra
    // arguments were passed, we silently drop them.  If any of the types
    // mismatch, we don't transform.
    unsigned ArgNo = 0;
    bool DontTransform = false;
    for (llvm::Function::arg_iterator AI = NewFn->arg_begin(),
         E = NewFn->arg_end(); AI != E; ++AI, ++ArgNo) {
      if (CS.arg_size() == ArgNo ||
          CS.getArgument(ArgNo)->getType() != AI->getType()) {
        DontTransform = true;
        break;
      }
    }
    if (DontTransform)
      continue;

    // Okay, we can transform this.  Create the new call instruction and copy
    // over the required information.
    ArgList.append(CS.arg_begin(), CS.arg_begin() + ArgNo);
    llvm::CallInst *NewCall = llvm::CallInst::Create(NewFn, ArgList.begin(),
                                                     ArgList.end(), "", CI);
    ArgList.clear();
    if (!NewCall->getType()->isVoidTy())
      NewCall->takeName(CI);
    NewCall->setAttributes(CI->getAttributes());
    NewCall->setCallingConv(CI->getCallingConv());

    // Finally, remove the old call, replacing any uses with the new one.
    if (!CI->use_empty())
      CI->replaceAllUsesWith(NewCall);

    // Copy debug location attached to CI.
    if (!CI->getDebugLoc().isUnknown())
      NewCall->setDebugLoc(CI->getDebugLoc());
    CI->eraseFromParent();
  }
}


void CodeGenModule::EmitGlobalFunctionDefinition(GlobalDecl GD) {
  const FunctionDecl *D = cast<FunctionDecl>(GD.getDecl());
  const llvm::FunctionType *Ty = getTypes().GetFunctionType(GD);
  // Get or create the prototype for the function.
  llvm::Constant *Entry = GetAddrOfFunction(GD, Ty);

  // Strip off a bitcast if we got one back.
  if (llvm::ConstantExpr *CE = dyn_cast<llvm::ConstantExpr>(Entry)) {
    assert(CE->getOpcode() == llvm::Instruction::BitCast);
    Entry = CE->getOperand(0);
  }


  if (cast<llvm::GlobalValue>(Entry)->getType()->getElementType() != Ty) {
    llvm::GlobalValue *OldFn = cast<llvm::GlobalValue>(Entry);

    // If the types mismatch then we have to rewrite the definition.
    assert(OldFn->isDeclaration() &&
           "Shouldn't replace non-declaration");

    // F is the Function* for the one with the wrong type, we must make a new
    // Function* and update everything that used F (a declaration) with the new
    // Function* (which will be a definition).
    //
    // This happens if there is a prototype for a function
    // (e.g. "int f()") and then a definition of a different type
    // (e.g. "int f(int x)").  Move the old function aside so that it
    // doesn't interfere with GetAddrOfFunction.
    OldFn->setName(llvm::StringRef());
    llvm::Function *NewFn = cast<llvm::Function>(GetAddrOfFunction(GD, Ty));

    // If this is an implementation of a function without a prototype, try to
    // replace any existing uses of the function (which may be calls) with uses
    // of the new function
    if (D->getType()->isFunctionNoProtoType()) {
      ReplaceUsesOfNonProtoTypeWithRealFunction(OldFn, NewFn);
      OldFn->removeDeadConstantUsers();
    }

    // Replace uses of F with the Function we will endow with a body.
    if (!Entry->use_empty()) {
      llvm::Constant *NewPtrForOldDecl =
        llvm::ConstantExpr::getBitCast(NewFn, Entry->getType());
      Entry->replaceAllUsesWith(NewPtrForOldDecl);
    }

    // Ok, delete the old function now, which is dead.
    OldFn->eraseFromParent();

    Entry = NewFn;
  }

  // We need to set linkage and visibility on the function before
  // generating code for it because various parts of IR generation
  // want to propagate this information down (e.g. to local static
  // declarations).
  llvm::Function *Fn = cast<llvm::Function>(Entry);
  setFunctionLinkage(D, Fn);

  // FIXME: this is redundant with part of SetFunctionDefinitionAttributes
  setGlobalVisibility(Fn, D);

  CodeGenFunction(*this).GenerateCode(D, Fn);

  SetFunctionDefinitionAttributes(D, Fn);
  SetLLVMFunctionAttributesForDefinition(D, Fn);

  if (const ConstructorAttr *CA = D->getAttr<ConstructorAttr>())
    AddGlobalCtor(Fn, CA->getPriority());
  if (const DestructorAttr *DA = D->getAttr<DestructorAttr>())
    AddGlobalDtor(Fn, DA->getPriority());
}

void CodeGenModule::EmitAliasDefinition(GlobalDecl GD) {
  const ValueDecl *D = cast<ValueDecl>(GD.getDecl());
  const AliasAttr *AA = D->getAttr<AliasAttr>();
  assert(AA && "Not an alias?");

  llvm::StringRef MangledName = getMangledName(GD);

  // If there is a definition in the module, then it wins over the alias.
  // This is dubious, but allow it to be safe.  Just ignore the alias.
  llvm::GlobalValue *Entry = GetGlobalValue(MangledName);
  if (Entry && !Entry->isDeclaration())
    return;

  const llvm::Type *DeclTy = getTypes().ConvertTypeForMem(D->getType());

  // Create a reference to the named value.  This ensures that it is emitted
  // if a deferred decl.
  llvm::Constant *Aliasee;
  if (isa<llvm::FunctionType>(DeclTy))
    Aliasee = GetOrCreateLLVMFunction(AA->getAliasee(), DeclTy, GlobalDecl(),
                                      /*ForVTable=*/false);
  else
    Aliasee = GetOrCreateLLVMGlobal(AA->getAliasee(),
                                    llvm::PointerType::getUnqual(DeclTy), 0);

  // Create the new alias itself, but don't set a name yet.
  llvm::GlobalValue *GA =
    new llvm::GlobalAlias(Aliasee->getType(),
                          llvm::Function::ExternalLinkage,
                          "", Aliasee, &getModule());

  if (Entry) {
    assert(Entry->isDeclaration());

    // If there is a declaration in the module, then we had an extern followed
    // by the alias, as in:
    //   extern int test6();
    //   ...
    //   int test6() __attribute__((alias("test7")));
    //
    // Remove it and replace uses of it with the alias.
    GA->takeName(Entry);

    Entry->replaceAllUsesWith(llvm::ConstantExpr::getBitCast(GA,
                                                          Entry->getType()));
    Entry->eraseFromParent();
  } else {
    GA->setName(MangledName);
  }

  // Set attributes which are particular to an alias; this is a
  // specialization of the attributes which may be set on a global
  // variable/function.
  if (D->hasAttr<DLLExportAttr>()) {
    if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
      // The dllexport attribute is ignored for undefined symbols.
      if (FD->hasBody())
        GA->setLinkage(llvm::Function::DLLExportLinkage);
    } else {
      GA->setLinkage(llvm::Function::DLLExportLinkage);
    }
  } else if (D->hasAttr<WeakAttr>() ||
             D->hasAttr<WeakRefAttr>() ||
             D->hasAttr<WeakImportAttr>()) {
    GA->setLinkage(llvm::Function::WeakAnyLinkage);
  }

  SetCommonAttributes(D, GA);
}

/// getBuiltinLibFunction - Given a builtin id for a function like
/// "__builtin_fabsf", return a Function* for "fabsf".
llvm::Value *CodeGenModule::getBuiltinLibFunction(const FunctionDecl *FD,
                                                  unsigned BuiltinID) {
  assert((Context.BuiltinInfo.isLibFunction(BuiltinID) ||
          Context.BuiltinInfo.isPredefinedLibFunction(BuiltinID)) &&
         "isn't a lib fn");

  // Get the name, skip over the __builtin_ prefix (if necessary).
  const char *Name = Context.BuiltinInfo.GetName(BuiltinID);
  if (Context.BuiltinInfo.isLibFunction(BuiltinID))
    Name += 10;

  const llvm::FunctionType *Ty =
    cast<llvm::FunctionType>(getTypes().ConvertType(FD->getType()));

  return GetOrCreateLLVMFunction(Name, Ty, GlobalDecl(FD), /*ForVTable=*/false);
}

llvm::Function *CodeGenModule::getIntrinsic(unsigned IID,const llvm::Type **Tys,
                                            unsigned NumTys) {
  return llvm::Intrinsic::getDeclaration(&getModule(),
                                         (llvm::Intrinsic::ID)IID, Tys, NumTys);
}

static llvm::StringMapEntry<llvm::Constant*> &
GetConstantCFStringEntry(llvm::StringMap<llvm::Constant*> &Map,
                         const StringLiteral *Literal,
                         bool TargetIsLSB,
                         bool &IsUTF16,
                         unsigned &StringLength) {
  llvm::StringRef String = Literal->getString();
  unsigned NumBytes = String.size();

  // Check for simple case.
  if (!Literal->containsNonAsciiOrNull()) {
    StringLength = NumBytes;
    return Map.GetOrCreateValue(String);
  }

  // Otherwise, convert the UTF8 literals into a byte string.
  llvm::SmallVector<UTF16, 128> ToBuf(NumBytes);
  const UTF8 *FromPtr = (UTF8 *)String.data();
  UTF16 *ToPtr = &ToBuf[0];

  (void)ConvertUTF8toUTF16(&FromPtr, FromPtr + NumBytes,
                           &ToPtr, ToPtr + NumBytes,
                           strictConversion);

  // ConvertUTF8toUTF16 returns the length in ToPtr.
  StringLength = ToPtr - &ToBuf[0];

  // Render the UTF-16 string into a byte array and convert to the target byte
  // order.
  //
  // FIXME: This isn't something we should need to do here.
  llvm::SmallString<128> AsBytes;
  AsBytes.reserve(StringLength * 2);
  for (unsigned i = 0; i != StringLength; ++i) {
    unsigned short Val = ToBuf[i];
    if (TargetIsLSB) {
      AsBytes.push_back(Val & 0xFF);
      AsBytes.push_back(Val >> 8);
    } else {
      AsBytes.push_back(Val >> 8);
      AsBytes.push_back(Val & 0xFF);
    }
  }
  // Append one extra null character, the second is automatically added by our
  // caller.
  AsBytes.push_back(0);

  IsUTF16 = true;
  return Map.GetOrCreateValue(llvm::StringRef(AsBytes.data(), AsBytes.size()));
}

llvm::Constant *
CodeGenModule::GetAddrOfConstantCFString(const StringLiteral *Literal) {
  unsigned StringLength = 0;
  bool isUTF16 = false;
  llvm::StringMapEntry<llvm::Constant*> &Entry =
    GetConstantCFStringEntry(CFConstantStringMap, Literal,
                             getTargetData().isLittleEndian(),
                             isUTF16, StringLength);

  if (llvm::Constant *C = Entry.getValue())
    return C;

  llvm::Constant *Zero =
      llvm::Constant::getNullValue(llvm::Type::getInt32Ty(VMContext));
  llvm::Constant *Zeros[] = { Zero, Zero };

  // If we don't already have it, get __CFConstantStringClassReference.
  if (!CFConstantStringClassRef) {
    const llvm::Type *Ty = getTypes().ConvertType(getContext().IntTy);
    Ty = llvm::ArrayType::get(Ty, 0);
    llvm::Constant *GV = CreateRuntimeVariable(Ty,
                                           "__CFConstantStringClassReference");
    // Decay array -> ptr
    CFConstantStringClassRef =
      llvm::ConstantExpr::getGetElementPtr(GV, Zeros, 2);
  }

  QualType CFTy = getContext().getCFConstantStringType();

  const llvm::StructType *STy =
    cast<llvm::StructType>(getTypes().ConvertType(CFTy));

  std::vector<llvm::Constant*> Fields(4);

  // Class pointer.
  Fields[0] = CFConstantStringClassRef;

  // Flags.
  const llvm::Type *Ty = getTypes().ConvertType(getContext().UnsignedIntTy);
  Fields[1] = isUTF16 ? llvm::ConstantInt::get(Ty, 0x07d0) :
    llvm::ConstantInt::get(Ty, 0x07C8);

  // String pointer.
  llvm::Constant *C = llvm::ConstantArray::get(VMContext, Entry.getKey().str());

  llvm::GlobalValue::LinkageTypes Linkage;
  bool isConstant;
  if (isUTF16) {
    // FIXME: why do utf strings get "_" labels instead of "L" labels?
    Linkage = llvm::GlobalValue::InternalLinkage;
    // Note: -fwritable-strings doesn't make unicode CFStrings writable, but
    // does make plain ascii ones writable.
    isConstant = true;
  } else {
    // FIXME: With OS X ld 123.2 (xcode 4) and LTO we would get a linker error
    // when using private linkage. It is not clear if this is a bug in ld
    // or a reasonable new restriction.
    Linkage = llvm::GlobalValue::LinkerPrivateLinkage;
    isConstant = !Features.WritableStrings;
  }
  
  llvm::GlobalVariable *GV =
    new llvm::GlobalVariable(getModule(), C->getType(), isConstant, Linkage, C,
                             ".str");
  GV->setUnnamedAddr(true);
  if (isUTF16) {
    CharUnits Align = getContext().getTypeAlignInChars(getContext().ShortTy);
    GV->setAlignment(Align.getQuantity());
  }
  Fields[2] = llvm::ConstantExpr::getGetElementPtr(GV, Zeros, 2);

  // String length.
  Ty = getTypes().ConvertType(getContext().LongTy);
  Fields[3] = llvm::ConstantInt::get(Ty, StringLength);

  // The struct.
  C = llvm::ConstantStruct::get(STy, Fields);
  GV = new llvm::GlobalVariable(getModule(), C->getType(), true,
                                llvm::GlobalVariable::PrivateLinkage, C,
                                "_unnamed_cfstring_");
  if (const char *Sect = getContext().Target.getCFStringSection())
    GV->setSection(Sect);
  Entry.setValue(GV);

  return GV;
}

llvm::Constant *
CodeGenModule::GetAddrOfConstantString(const StringLiteral *Literal) {
  unsigned StringLength = 0;
  bool isUTF16 = false;
  llvm::StringMapEntry<llvm::Constant*> &Entry =
    GetConstantCFStringEntry(CFConstantStringMap, Literal,
                             getTargetData().isLittleEndian(),
                             isUTF16, StringLength);
  
  if (llvm::Constant *C = Entry.getValue())
    return C;
  
  llvm::Constant *Zero =
  llvm::Constant::getNullValue(llvm::Type::getInt32Ty(VMContext));
  llvm::Constant *Zeros[] = { Zero, Zero };
  
  // If we don't already have it, get _NSConstantStringClassReference.
  if (!ConstantStringClassRef) {
    std::string StringClass(getLangOptions().ObjCConstantStringClass);
    const llvm::Type *Ty = getTypes().ConvertType(getContext().IntTy);
    Ty = llvm::ArrayType::get(Ty, 0);
    llvm::Constant *GV;
    if (StringClass.empty())
      GV = CreateRuntimeVariable(Ty, 
                                 Features.ObjCNonFragileABI ?
                                 "OBJC_CLASS_$_NSConstantString" :
                                 "_NSConstantStringClassReference");
    else {
      std::string str;
      if (Features.ObjCNonFragileABI)
        str = "OBJC_CLASS_$_" + StringClass;
      else
        str = "_" + StringClass + "ClassReference";
      GV = CreateRuntimeVariable(Ty, str);
    }
    // Decay array -> ptr
    ConstantStringClassRef = 
    llvm::ConstantExpr::getGetElementPtr(GV, Zeros, 2);
  }
  
  QualType NSTy = getContext().getNSConstantStringType();
  
  const llvm::StructType *STy =
  cast<llvm::StructType>(getTypes().ConvertType(NSTy));
  
  std::vector<llvm::Constant*> Fields(3);
  
  // Class pointer.
  Fields[0] = ConstantStringClassRef;
  
  // String pointer.
  llvm::Constant *C = llvm::ConstantArray::get(VMContext, Entry.getKey().str());
  
  llvm::GlobalValue::LinkageTypes Linkage;
  bool isConstant;
  if (isUTF16) {
    // FIXME: why do utf strings get "_" labels instead of "L" labels?
    Linkage = llvm::GlobalValue::InternalLinkage;
    // Note: -fwritable-strings doesn't make unicode NSStrings writable, but
    // does make plain ascii ones writable.
    isConstant = true;
  } else {
    Linkage = llvm::GlobalValue::PrivateLinkage;
    isConstant = !Features.WritableStrings;
  }
  
  llvm::GlobalVariable *GV =
  new llvm::GlobalVariable(getModule(), C->getType(), isConstant, Linkage, C,
                           ".str");
  GV->setUnnamedAddr(true);
  if (isUTF16) {
    CharUnits Align = getContext().getTypeAlignInChars(getContext().ShortTy);
    GV->setAlignment(Align.getQuantity());
  }
  Fields[1] = llvm::ConstantExpr::getGetElementPtr(GV, Zeros, 2);
  
  // String length.
  const llvm::Type *Ty = getTypes().ConvertType(getContext().UnsignedIntTy);
  Fields[2] = llvm::ConstantInt::get(Ty, StringLength);
  
  // The struct.
  C = llvm::ConstantStruct::get(STy, Fields);
  GV = new llvm::GlobalVariable(getModule(), C->getType(), true,
                                llvm::GlobalVariable::PrivateLinkage, C,
                                "_unnamed_nsstring_");
  // FIXME. Fix section.
  if (const char *Sect = 
        Features.ObjCNonFragileABI 
          ? getContext().Target.getNSStringNonFragileABISection() 
          : getContext().Target.getNSStringSection())
    GV->setSection(Sect);
  Entry.setValue(GV);
  
  return GV;
}

/// GetStringForStringLiteral - Return the appropriate bytes for a
/// string literal, properly padded to match the literal type.
std::string CodeGenModule::GetStringForStringLiteral(const StringLiteral *E) {
  const ASTContext &Context = getContext();
  const ConstantArrayType *CAT =
    Context.getAsConstantArrayType(E->getType());
  assert(CAT && "String isn't pointer or array!");

  // Resize the string to the right size.
  uint64_t RealLen = CAT->getSize().getZExtValue();

  if (E->isWide())
    RealLen *= Context.Target.getWCharWidth() / Context.getCharWidth();

  std::string Str = E->getString().str();
  Str.resize(RealLen, '\0');

  return Str;
}

/// GetAddrOfConstantStringFromLiteral - Return a pointer to a
/// constant array for the given string literal.
llvm::Constant *
CodeGenModule::GetAddrOfConstantStringFromLiteral(const StringLiteral *S) {
  // FIXME: This can be more efficient.
  // FIXME: We shouldn't need to bitcast the constant in the wide string case.
  llvm::Constant *C = GetAddrOfConstantString(GetStringForStringLiteral(S));
  if (S->isWide()) {
    llvm::Type *DestTy =
        llvm::PointerType::getUnqual(getTypes().ConvertType(S->getType()));
    C = llvm::ConstantExpr::getBitCast(C, DestTy);
  }
  return C;
}

/// GetAddrOfConstantStringFromObjCEncode - Return a pointer to a constant
/// array for the given ObjCEncodeExpr node.
llvm::Constant *
CodeGenModule::GetAddrOfConstantStringFromObjCEncode(const ObjCEncodeExpr *E) {
  std::string Str;
  getContext().getObjCEncodingForType(E->getEncodedType(), Str);

  return GetAddrOfConstantCString(Str);
}


/// GenerateWritableString -- Creates storage for a string literal.
static llvm::Constant *GenerateStringLiteral(llvm::StringRef str,
                                             bool constant,
                                             CodeGenModule &CGM,
                                             const char *GlobalName) {
  // Create Constant for this string literal. Don't add a '\0'.
  llvm::Constant *C =
      llvm::ConstantArray::get(CGM.getLLVMContext(), str, false);

  // Create a global variable for this string
  llvm::GlobalVariable *GV =
    new llvm::GlobalVariable(CGM.getModule(), C->getType(), constant,
                             llvm::GlobalValue::PrivateLinkage,
                             C, GlobalName);
  GV->setUnnamedAddr(true);
  return GV;
}

/// GetAddrOfConstantString - Returns a pointer to a character array
/// containing the literal. This contents are exactly that of the
/// given string, i.e. it will not be null terminated automatically;
/// see GetAddrOfConstantCString. Note that whether the result is
/// actually a pointer to an LLVM constant depends on
/// Feature.WriteableStrings.
///
/// The result has pointer to array type.
llvm::Constant *CodeGenModule::GetAddrOfConstantString(llvm::StringRef Str,
                                                       const char *GlobalName) {
  bool IsConstant = !Features.WritableStrings;

  // Get the default prefix if a name wasn't specified.
  if (!GlobalName)
    GlobalName = ".str";

  // Don't share any string literals if strings aren't constant.
  if (!IsConstant)
    return GenerateStringLiteral(Str, false, *this, GlobalName);

  llvm::StringMapEntry<llvm::Constant *> &Entry =
    ConstantStringMap.GetOrCreateValue(Str);

  if (Entry.getValue())
    return Entry.getValue();

  // Create a global variable for this.
  llvm::Constant *C = GenerateStringLiteral(Str, true, *this, GlobalName);
  Entry.setValue(C);
  return C;
}

/// GetAddrOfConstantCString - Returns a pointer to a character
/// array containing the literal and a terminating '\0'
/// character. The result has pointer to array type.
llvm::Constant *CodeGenModule::GetAddrOfConstantCString(const std::string &Str,
                                                        const char *GlobalName){
  llvm::StringRef StrWithNull(Str.c_str(), Str.size() + 1);
  return GetAddrOfConstantString(StrWithNull, GlobalName);
}

/// EmitObjCPropertyImplementations - Emit information for synthesized
/// properties for an implementation.
void CodeGenModule::EmitObjCPropertyImplementations(const
                                                    ObjCImplementationDecl *D) {
  for (ObjCImplementationDecl::propimpl_iterator
         i = D->propimpl_begin(), e = D->propimpl_end(); i != e; ++i) {
    ObjCPropertyImplDecl *PID = *i;

    // Dynamic is just for type-checking.
    if (PID->getPropertyImplementation() == ObjCPropertyImplDecl::Synthesize) {
      ObjCPropertyDecl *PD = PID->getPropertyDecl();

      // Determine which methods need to be implemented, some may have
      // been overridden. Note that ::isSynthesized is not the method
      // we want, that just indicates if the decl came from a
      // property. What we want to know is if the method is defined in
      // this implementation.
      if (!D->getInstanceMethod(PD->getGetterName()))
        CodeGenFunction(*this).GenerateObjCGetter(
                                 const_cast<ObjCImplementationDecl *>(D), PID);
      if (!PD->isReadOnly() &&
          !D->getInstanceMethod(PD->getSetterName()))
        CodeGenFunction(*this).GenerateObjCSetter(
                                 const_cast<ObjCImplementationDecl *>(D), PID);
    }
  }
}

/// EmitObjCIvarInitializations - Emit information for ivar initialization
/// for an implementation.
void CodeGenModule::EmitObjCIvarInitializations(ObjCImplementationDecl *D) {
  if (D->getNumIvarInitializers() == 0)
    return;
  DeclContext* DC = const_cast<DeclContext*>(dyn_cast<DeclContext>(D));
  assert(DC && "EmitObjCIvarInitializations - null DeclContext");
  IdentifierInfo *II = &getContext().Idents.get(".cxx_destruct");
  Selector cxxSelector = getContext().Selectors.getSelector(0, &II);
  ObjCMethodDecl *DTORMethod = ObjCMethodDecl::Create(getContext(), 
                                                  D->getLocation(),
                                                  D->getLocation(), cxxSelector,
                                                  getContext().VoidTy, 0, 
                                                  DC, true, false, true, false,
                                                  ObjCMethodDecl::Required);
  D->addInstanceMethod(DTORMethod);
  CodeGenFunction(*this).GenerateObjCCtorDtorMethod(D, DTORMethod, false);
  
  II = &getContext().Idents.get(".cxx_construct");
  cxxSelector = getContext().Selectors.getSelector(0, &II);
  // The constructor returns 'self'.
  ObjCMethodDecl *CTORMethod = ObjCMethodDecl::Create(getContext(), 
                                                D->getLocation(),
                                                D->getLocation(), cxxSelector,
                                                getContext().getObjCIdType(), 0, 
                                                DC, true, false, true, false,
                                                ObjCMethodDecl::Required);
  D->addInstanceMethod(CTORMethod);
  CodeGenFunction(*this).GenerateObjCCtorDtorMethod(D, CTORMethod, true);
  

}

/// EmitNamespace - Emit all declarations in a namespace.
void CodeGenModule::EmitNamespace(const NamespaceDecl *ND) {
  for (RecordDecl::decl_iterator I = ND->decls_begin(), E = ND->decls_end();
       I != E; ++I)
    EmitTopLevelDecl(*I);
}

// EmitLinkageSpec - Emit all declarations in a linkage spec.
void CodeGenModule::EmitLinkageSpec(const LinkageSpecDecl *LSD) {
  if (LSD->getLanguage() != LinkageSpecDecl::lang_c &&
      LSD->getLanguage() != LinkageSpecDecl::lang_cxx) {
    ErrorUnsupported(LSD, "linkage spec");
    return;
  }

  for (RecordDecl::decl_iterator I = LSD->decls_begin(), E = LSD->decls_end();
       I != E; ++I)
    EmitTopLevelDecl(*I);
}

/// EmitTopLevelDecl - Emit code for a single top level declaration.
void CodeGenModule::EmitTopLevelDecl(Decl *D) {
  // If an error has occurred, stop code generation, but continue
  // parsing and semantic analysis (to ensure all warnings and errors
  // are emitted).
  if (Diags.hasErrorOccurred())
    return;

  // Ignore dependent declarations.
  if (D->getDeclContext() && D->getDeclContext()->isDependentContext())
    return;

  switch (D->getKind()) {
  case Decl::CXXConversion:
  case Decl::CXXMethod:
  case Decl::Function:
    // Skip function templates
    if (cast<FunctionDecl>(D)->getDescribedFunctionTemplate())
      return;

    EmitGlobal(cast<FunctionDecl>(D));
    break;
      
  case Decl::Var:
    EmitGlobal(cast<VarDecl>(D));
    break;

  // C++ Decls
  case Decl::Namespace:
    EmitNamespace(cast<NamespaceDecl>(D));
    break;
    // No code generation needed.
  case Decl::UsingShadow:
  case Decl::Using:
  case Decl::UsingDirective:
  case Decl::ClassTemplate:
  case Decl::FunctionTemplate:
  case Decl::NamespaceAlias:
    break;
  case Decl::CXXConstructor:
    // Skip function templates
    if (cast<FunctionDecl>(D)->getDescribedFunctionTemplate())
      return;
      
    EmitCXXConstructors(cast<CXXConstructorDecl>(D));
    break;
  case Decl::CXXDestructor:
    EmitCXXDestructors(cast<CXXDestructorDecl>(D));
    break;

  case Decl::StaticAssert:
    // Nothing to do.
    break;

  // Objective-C Decls

  // Forward declarations, no (immediate) code generation.
  case Decl::ObjCClass:
  case Decl::ObjCForwardProtocol:
  case Decl::ObjCInterface:
    break;
  
    case Decl::ObjCCategory: {
      ObjCCategoryDecl *CD = cast<ObjCCategoryDecl>(D);
      if (CD->IsClassExtension() && CD->hasSynthBitfield())
        Context.ResetObjCLayout(CD->getClassInterface());
      break;
    }
      

  case Decl::ObjCProtocol:
    Runtime->GenerateProtocol(cast<ObjCProtocolDecl>(D));
    break;

  case Decl::ObjCCategoryImpl:
    // Categories have properties but don't support synthesize so we
    // can ignore them here.
    Runtime->GenerateCategory(cast<ObjCCategoryImplDecl>(D));
    break;

  case Decl::ObjCImplementation: {
    ObjCImplementationDecl *OMD = cast<ObjCImplementationDecl>(D);
    if (Features.ObjCNonFragileABI2 && OMD->hasSynthBitfield())
      Context.ResetObjCLayout(OMD->getClassInterface());
    EmitObjCPropertyImplementations(OMD);
    EmitObjCIvarInitializations(OMD);
    Runtime->GenerateClass(OMD);
    break;
  }
  case Decl::ObjCMethod: {
    ObjCMethodDecl *OMD = cast<ObjCMethodDecl>(D);
    // If this is not a prototype, emit the body.
    if (OMD->getBody())
      CodeGenFunction(*this).GenerateObjCMethod(OMD);
    break;
  }
  case Decl::ObjCCompatibleAlias:
    // compatibility-alias is a directive and has no code gen.
    break;

  case Decl::LinkageSpec:
    EmitLinkageSpec(cast<LinkageSpecDecl>(D));
    break;

  case Decl::FileScopeAsm: {
    FileScopeAsmDecl *AD = cast<FileScopeAsmDecl>(D);
    llvm::StringRef AsmString = AD->getAsmString()->getString();

    const std::string &S = getModule().getModuleInlineAsm();
    if (S.empty())
      getModule().setModuleInlineAsm(AsmString);
    else
      getModule().setModuleInlineAsm(S + '\n' + AsmString.str());
    break;
  }

  default:
    // Make sure we handled everything we should, every other kind is a
    // non-top-level decl.  FIXME: Would be nice to have an isTopLevelDeclKind
    // function. Need to recode Decl::Kind to do that easily.
    assert(isa<TypeDecl>(D) && "Unsupported decl kind");
  }
}

/// Turns the given pointer into a constant.
static llvm::Constant *GetPointerConstant(llvm::LLVMContext &Context,
                                          const void *Ptr) {
  uintptr_t PtrInt = reinterpret_cast<uintptr_t>(Ptr);
  const llvm::Type *i64 = llvm::Type::getInt64Ty(Context);
  return llvm::ConstantInt::get(i64, PtrInt);
}

static void EmitGlobalDeclMetadata(CodeGenModule &CGM,
                                   llvm::NamedMDNode *&GlobalMetadata,
                                   GlobalDecl D,
                                   llvm::GlobalValue *Addr) {
  if (!GlobalMetadata)
    GlobalMetadata =
      CGM.getModule().getOrInsertNamedMetadata("clang.global.decl.ptrs");

  // TODO: should we report variant information for ctors/dtors?
  llvm::Value *Ops[] = {
    Addr,
    GetPointerConstant(CGM.getLLVMContext(), D.getDecl())
  };
  GlobalMetadata->addOperand(llvm::MDNode::get(CGM.getLLVMContext(), Ops, 2));
}

/// Emits metadata nodes associating all the global values in the
/// current module with the Decls they came from.  This is useful for
/// projects using IR gen as a subroutine.
///
/// Since there's currently no way to associate an MDNode directly
/// with an llvm::GlobalValue, we create a global named metadata
/// with the name 'clang.global.decl.ptrs'.
void CodeGenModule::EmitDeclMetadata() {
  llvm::NamedMDNode *GlobalMetadata = 0;

  // StaticLocalDeclMap
  for (llvm::DenseMap<GlobalDecl,llvm::StringRef>::iterator
         I = MangledDeclNames.begin(), E = MangledDeclNames.end();
       I != E; ++I) {
    llvm::GlobalValue *Addr = getModule().getNamedValue(I->second);
    EmitGlobalDeclMetadata(*this, GlobalMetadata, I->first, Addr);
  }
}

/// Emits metadata nodes for all the local variables in the current
/// function.
void CodeGenFunction::EmitDeclMetadata() {
  if (LocalDeclMap.empty()) return;

  llvm::LLVMContext &Context = getLLVMContext();

  // Find the unique metadata ID for this name.
  unsigned DeclPtrKind = Context.getMDKindID("clang.decl.ptr");

  llvm::NamedMDNode *GlobalMetadata = 0;

  for (llvm::DenseMap<const Decl*, llvm::Value*>::iterator
         I = LocalDeclMap.begin(), E = LocalDeclMap.end(); I != E; ++I) {
    const Decl *D = I->first;
    llvm::Value *Addr = I->second;

    if (llvm::AllocaInst *Alloca = dyn_cast<llvm::AllocaInst>(Addr)) {
      llvm::Value *DAddr = GetPointerConstant(getLLVMContext(), D);
      Alloca->setMetadata(DeclPtrKind, llvm::MDNode::get(Context, &DAddr, 1));
    } else if (llvm::GlobalValue *GV = dyn_cast<llvm::GlobalValue>(Addr)) {
      GlobalDecl GD = GlobalDecl(cast<VarDecl>(D));
      EmitGlobalDeclMetadata(CGM, GlobalMetadata, GD, GV);
    }
  }
}

///@name Custom Runtime Function Interfaces
///@{
//
// FIXME: These can be eliminated once we can have clients just get the required
// AST nodes from the builtin tables.

llvm::Constant *CodeGenModule::getBlockObjectDispose() {
  if (BlockObjectDispose)
    return BlockObjectDispose;

  // If we saw an explicit decl, use that.
  if (BlockObjectDisposeDecl) {
    return BlockObjectDispose = GetAddrOfFunction(
      BlockObjectDisposeDecl,
      getTypes().GetFunctionType(BlockObjectDisposeDecl));
  }

  // Otherwise construct the function by hand.
  const llvm::FunctionType *FTy;
  std::vector<const llvm::Type*> ArgTys;
  const llvm::Type *ResultType = llvm::Type::getVoidTy(VMContext);
  ArgTys.push_back(Int8PtrTy);
  ArgTys.push_back(llvm::Type::getInt32Ty(VMContext));
  FTy = llvm::FunctionType::get(ResultType, ArgTys, false);
  return BlockObjectDispose =
    CreateRuntimeFunction(FTy, "_Block_object_dispose");
}

llvm::Constant *CodeGenModule::getBlockObjectAssign() {
  if (BlockObjectAssign)
    return BlockObjectAssign;

  // If we saw an explicit decl, use that.
  if (BlockObjectAssignDecl) {
    return BlockObjectAssign = GetAddrOfFunction(
      BlockObjectAssignDecl,
      getTypes().GetFunctionType(BlockObjectAssignDecl));
  }

  // Otherwise construct the function by hand.
  const llvm::FunctionType *FTy;
  std::vector<const llvm::Type*> ArgTys;
  const llvm::Type *ResultType = llvm::Type::getVoidTy(VMContext);
  ArgTys.push_back(Int8PtrTy);
  ArgTys.push_back(Int8PtrTy);
  ArgTys.push_back(llvm::Type::getInt32Ty(VMContext));
  FTy = llvm::FunctionType::get(ResultType, ArgTys, false);
  return BlockObjectAssign =
    CreateRuntimeFunction(FTy, "_Block_object_assign");
}

llvm::Constant *CodeGenModule::getNSConcreteGlobalBlock() {
  if (NSConcreteGlobalBlock)
    return NSConcreteGlobalBlock;

  // If we saw an explicit decl, use that.
  if (NSConcreteGlobalBlockDecl) {
    return NSConcreteGlobalBlock = GetAddrOfGlobalVar(
      NSConcreteGlobalBlockDecl,
      getTypes().ConvertType(NSConcreteGlobalBlockDecl->getType()));
  }

  // Otherwise construct the variable by hand.
  return NSConcreteGlobalBlock =
    CreateRuntimeVariable(Int8PtrTy, "_NSConcreteGlobalBlock");
}

llvm::Constant *CodeGenModule::getNSConcreteStackBlock() {
  if (NSConcreteStackBlock)
    return NSConcreteStackBlock;

  // If we saw an explicit decl, use that.
  if (NSConcreteStackBlockDecl) {
    return NSConcreteStackBlock = GetAddrOfGlobalVar(
      NSConcreteStackBlockDecl,
      getTypes().ConvertType(NSConcreteStackBlockDecl->getType()));
  }

  // Otherwise construct the variable by hand.
  return NSConcreteStackBlock =
    CreateRuntimeVariable(Int8PtrTy, "_NSConcreteStackBlock");
}

///@}
