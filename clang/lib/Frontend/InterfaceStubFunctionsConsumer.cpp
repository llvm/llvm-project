//===--- InterfaceStubFunctionsConsumer.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Mangle.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Sema/TemplateInstCallback.h"
#include "llvm/BinaryFormat/ELF.h"

using namespace clang;

class InterfaceStubFunctionsConsumer : public ASTConsumer {
  CompilerInstance &Instance;
  StringRef InFile;
  StringRef Format;
  std::set<std::string> ParsedTemplates;

  enum RootDeclOrigin { TopLevel = 0, FromTU = 1, IsLate = 2 };
  struct MangledSymbol {
    std::string ParentName;
    uint8_t Type;
    uint8_t Binding;
    std::vector<std::string> Names;
    MangledSymbol() = delete;

    MangledSymbol(const std::string &ParentName, uint8_t Type, uint8_t Binding,
                  std::vector<std::string> Names)
        : ParentName(ParentName), Type(Type), Binding(Binding), Names(Names) {}
  };
  using MangledSymbols = std::map<const NamedDecl *, MangledSymbol>;

  bool WriteNamedDecl(const NamedDecl *ND, MangledSymbols &Symbols, int RDO) {
    // Here we filter out anything that's not set to DefaultVisibility.
    // DefaultVisibility is set on a decl when -fvisibility is not specified on
    // the command line (or specified as default) and the decl does not have
    // __attribute__((visibility("hidden"))) set or when the command line
    // argument is set to hidden but the decl explicitly has
    // __attribute__((visibility ("default"))) set. We do this so that the user
    // can have fine grain control of what they want to expose in the stub.
    auto isVisible = [](const NamedDecl *ND) -> bool {
      return ND->getVisibility() == DefaultVisibility;
    };

    auto ignoreDecl = [this, isVisible](const NamedDecl *ND) -> bool {
      if (!isVisible(ND))
        return true;

      if (const VarDecl *VD = dyn_cast<VarDecl>(ND))
        if ((VD->getStorageClass() == StorageClass::SC_Extern) ||
            (VD->getStorageClass() == StorageClass::SC_Static &&
             VD->getParentFunctionOrMethod() == nullptr))
          return true;

      if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(ND)) {
        if (FD->isInlined() && !isa<CXXMethodDecl>(FD) &&
            !Instance.getLangOpts().GNUInline)
          return true;
        if (const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(FD)) {
          if (const auto *RC = dyn_cast<CXXRecordDecl>(MD->getParent()))
            if (isa<ClassTemplateDecl>(RC->getParent()) || !isVisible(RC))
              return true;
          if (MD->isDependentContext() || !MD->hasBody())
            return true;
        }
        if (FD->getStorageClass() == StorageClass::SC_Static)
          return true;
      }
      return false;
    };

    auto getParentFunctionDecl = [](const NamedDecl *ND) -> const NamedDecl * {
      if (const VarDecl *VD = dyn_cast<VarDecl>(ND))
        if (const auto *FD =
                dyn_cast_or_null<FunctionDecl>(VD->getParentFunctionOrMethod()))
          return FD;
      return nullptr;
    };

    auto getMangledNames = [](const NamedDecl *ND) -> std::vector<std::string> {
      if (!ND)
        return {""};
      ASTNameGenerator NameGen(ND->getASTContext());
      std::vector<std::string> MangledNames = NameGen.getAllManglings(ND);
      if (isa<CXXConstructorDecl>(ND) || isa<CXXDestructorDecl>(ND))
        return MangledNames;
#ifdef EXPENSIVE_CHECKS
      assert(MangledNames.size() <= 1 && "Expected only one name mangling.");
#endif
      return {NameGen.getName(ND)};
    };

    if (!(RDO & FromTU))
      return true;
    if (Symbols.find(ND) != Symbols.end())
      return true;
    // - Currently have not figured out how to produce the names for FieldDecls.
    // - Do not want to produce symbols for function paremeters.
    if (isa<FieldDecl>(ND) || isa<ParmVarDecl>(ND))
      return true;

    const NamedDecl *ParentDecl = getParentFunctionDecl(ND);
    if ((ParentDecl && ignoreDecl(ParentDecl)) || ignoreDecl(ND))
      return true;

    if (RDO & IsLate) {
      Instance.getDiagnostics().Report(diag::err_asm_invalid_type_in_input)
          << "Generating Interface Stubs is not supported with "
             "delayed template parsing.";
    } else {
      if (const auto *FD = dyn_cast<FunctionDecl>(ND))
        if (FD->isDependentContext())
          return true;

      const bool IsWeak = (ND->hasAttr<WeakAttr>() ||
                           ND->hasAttr<WeakRefAttr>() || ND->isWeakImported());

      Symbols.insert(std::make_pair(
          ND,
          MangledSymbol(getMangledNames(ParentDecl).front(),
                        // Type:
                        isa<VarDecl>(ND) ? llvm::ELF::STT_OBJECT
                                         : llvm::ELF::STT_FUNC,
                        // Binding:
                        IsWeak ? llvm::ELF::STB_WEAK : llvm::ELF::STB_GLOBAL,
                        getMangledNames(ND))));
    }
    return true;
  }

  void
  HandleDecls(const llvm::iterator_range<DeclContext::decl_iterator> &Decls,
              MangledSymbols &Symbols, int RDO) {
    for (const auto *D : Decls)
      HandleNamedDecl(dyn_cast<NamedDecl>(D), Symbols, RDO);
  }

  void HandleTemplateSpecializations(const FunctionTemplateDecl &FTD,
                                     MangledSymbols &Symbols, int RDO) {
    for (const auto *D : FTD.specializations())
      HandleNamedDecl(dyn_cast<NamedDecl>(D), Symbols, RDO);
  }

  void HandleTemplateSpecializations(const ClassTemplateDecl &CTD,
                                     MangledSymbols &Symbols, int RDO) {
    for (const auto *D : CTD.specializations())
      HandleNamedDecl(dyn_cast<NamedDecl>(D), Symbols, RDO);
  }

  bool HandleNamedDecl(const NamedDecl *ND, MangledSymbols &Symbols, int RDO) {
    if (!ND)
      return false;

    switch (ND->getKind()) {
    default:
      break;
    case Decl::Kind::Namespace:
      HandleDecls(cast<NamespaceDecl>(ND)->decls(), Symbols, RDO);
      return true;
    case Decl::Kind::CXXRecord:
      HandleDecls(cast<CXXRecordDecl>(ND)->decls(), Symbols, RDO);
      return true;
    case Decl::Kind::ClassTemplateSpecialization:
      HandleDecls(cast<ClassTemplateSpecializationDecl>(ND)->decls(), Symbols,
                  RDO);
      return true;
    case Decl::Kind::ClassTemplate:
      HandleTemplateSpecializations(*cast<ClassTemplateDecl>(ND), Symbols, RDO);
      return true;
    case Decl::Kind::FunctionTemplate:
      HandleTemplateSpecializations(*cast<FunctionTemplateDecl>(ND), Symbols,
                                    RDO);
      return true;
    case Decl::Kind::TemplateTypeParm:
      return true;
    case Decl::Kind::Var:
    case Decl::Kind::ParmVar:
    case Decl::Kind::CXXMethod:
    case Decl::Kind::CXXConstructor:
    case Decl::Kind::CXXDestructor:
    case Decl::Kind::Function:
    case Decl::Kind::Field:
      if (WriteNamedDecl(ND, Symbols, RDO))
        return true;
    }

    // While interface stubs are in the development stage, it's probably best to
    // catch anything that's not a VarDecl or Template/FunctionDecl.
    Instance.getDiagnostics().Report(diag::err_asm_invalid_type_in_input)
        << "Expected a function or function template decl.";
    return false;
  }

public:
  InterfaceStubFunctionsConsumer(CompilerInstance &Instance, StringRef InFile,
                                 StringRef Format)
      : Instance(Instance), InFile(InFile), Format(Format) {}

  void HandleTranslationUnit(ASTContext &context) override {
    struct Visitor : public RecursiveASTVisitor<Visitor> {
      bool VisitNamedDecl(NamedDecl *ND) {
        if (const auto *FD = dyn_cast<FunctionDecl>(ND))
          if (FD->isLateTemplateParsed()) {
            LateParsedDecls.insert(FD);
            return true;
          }

        if (const auto *VD = dyn_cast<ValueDecl>(ND)) {
          ValueDecls.insert(VD);
          return true;
        }

        NamedDecls.insert(ND);
        return true;
      }

      std::set<const NamedDecl *> LateParsedDecls;
      std::set<NamedDecl *> NamedDecls;
      std::set<const ValueDecl *> ValueDecls;
    } v;

    v.TraverseDecl(context.getTranslationUnitDecl());

    MangledSymbols Symbols;
    auto OS = Instance.createDefaultOutputFile(/*Binary=*/false, InFile, "ifs");
    if (!OS)
      return;

    if (Instance.getLangOpts().DelayedTemplateParsing) {
      clang::Sema &S = Instance.getSema();
      for (const auto *FD : v.LateParsedDecls) {
        clang::LateParsedTemplate &LPT =
            *S.LateParsedTemplateMap.find(cast<FunctionDecl>(FD))->second;
        S.LateTemplateParser(S.OpaqueParser, LPT);
        HandleNamedDecl(FD, Symbols, (FromTU | IsLate));
      }
    }

    for (const NamedDecl *ND : v.ValueDecls)
      HandleNamedDecl(ND, Symbols, FromTU);
    for (const NamedDecl *ND : v.NamedDecls)
      HandleNamedDecl(ND, Symbols, FromTU);

    auto writeIfoYaml = [this](const llvm::Triple &T,
                               const MangledSymbols &Symbols,
                               const ASTContext &context, StringRef Format,
                               raw_ostream &OS) -> void {
      OS << "--- !" << Format << "\n";
      OS << "FileHeader:\n";
      OS << "  Class:           ELFCLASS";
      OS << (T.isArch64Bit() ? "64" : "32");
      OS << "\n";
      OS << "  Data:            ELFDATA2";
      OS << (T.isLittleEndian() ? "LSB" : "MSB");
      OS << "\n";
      OS << "  Type:            ET_REL\n";
      OS << "  Machine:         "
         << llvm::StringSwitch<llvm::StringRef>(T.getArchName())
                .Case("x86_64", "EM_X86_64")
                .Case("i386", "EM_386")
                .Case("i686", "EM_386")
                .Case("aarch64", "EM_AARCH64")
                .Case("amdgcn", "EM_AMDGPU")
                .Case("r600", "EM_AMDGPU")
                .Case("arm", "EM_ARM")
                .Case("thumb", "EM_ARM")
                .Case("avr", "EM_AVR")
                .Case("mips", "EM_MIPS")
                .Case("mipsel", "EM_MIPS")
                .Case("mips64", "EM_MIPS")
                .Case("mips64el", "EM_MIPS")
                .Case("msp430", "EM_MSP430")
                .Case("ppc", "EM_PPC")
                .Case("ppc64", "EM_PPC64")
                .Case("ppc64le", "EM_PPC64")
                .Case("x86", T.isOSIAMCU() ? "EM_IAMCU" : "EM_386")
                .Case("x86_64", "EM_X86_64")
                .Default("EM_NONE")
         << "\nSymbols:\n";
      for (const auto &E : Symbols) {
        const MangledSymbol &Symbol = E.second;
        for (auto Name : Symbol.Names) {
          OS << "  - Name:            "
             << (Symbol.ParentName.empty() || Instance.getLangOpts().CPlusPlus
                     ? ""
                     : (Symbol.ParentName + "."))
             << Name << "\n"
             << "    Type:            STT_";
          switch (Symbol.Type) {
          default:
          case llvm::ELF::STT_NOTYPE:
            OS << "NOTYPE";
            break;
          case llvm::ELF::STT_OBJECT:
            OS << "OBJECT";
            break;
          case llvm::ELF::STT_FUNC:
            OS << "FUNC";
            break;
          }
          OS << "\n    Binding:         STB_"
             << ((Symbol.Binding == llvm::ELF::STB_WEAK) ? "WEAK" : "GLOBAL")
             << "\n";
        }
      }
      OS << "...\n";
      OS.flush();
    };

    auto writeIfoElfAbiYaml =
        [this](const llvm::Triple &T, const MangledSymbols &Symbols,
               const ASTContext &context, StringRef Format,
               raw_ostream &OS) -> void {
      OS << "--- !" << Format << "\n";
      OS << "TbeVersion: 1.0\n";
      OS << "Arch: " << T.getArchName() << "\n";
      OS << "Symbols:\n";
      for (const auto &E : Symbols) {
        const MangledSymbol &Symbol = E.second;
        for (auto Name : Symbol.Names) {
          OS << "  "
             << (Symbol.ParentName.empty() || Instance.getLangOpts().CPlusPlus
                     ? ""
                     : (Symbol.ParentName + "."))
             << Name << ": { Type: ";
          switch (Symbol.Type) {
          default:
            llvm_unreachable(
                "clang -emit-iterface-stubs: Unexpected symbol type.");
          case llvm::ELF::STT_NOTYPE:
            OS << "NoType";
            break;
          case llvm::ELF::STT_OBJECT: {
            auto VD = cast<ValueDecl>(E.first)->getType();
            OS << "Object, Size: "
               << context.getTypeSizeInChars(VD).getQuantity();
            break;
          }
          case llvm::ELF::STT_FUNC:
            OS << "Func";
            break;
          }
          if (Symbol.Binding == llvm::ELF::STB_WEAK)
            OS << ", Weak: true";
          OS << " }\n";
        }
      }
      OS << "...\n";
      OS.flush();
    };

    if (Format == "experimental-yaml-elf-v1")
      writeIfoYaml(Instance.getTarget().getTriple(), Symbols, context, Format,
                   *OS);
    else
      writeIfoElfAbiYaml(Instance.getTarget().getTriple(), Symbols, context,
                         Format, *OS);
  }
};

std::unique_ptr<ASTConsumer>
GenerateInterfaceYAMLExpV1Action::CreateASTConsumer(CompilerInstance &CI,
                                                    StringRef InFile) {
  return llvm::make_unique<InterfaceStubFunctionsConsumer>(
      CI, InFile, "experimental-yaml-elf-v1");
}

std::unique_ptr<ASTConsumer>
GenerateInterfaceTBEExpV1Action::CreateASTConsumer(CompilerInstance &CI,
                                                   StringRef InFile) {
  return llvm::make_unique<InterfaceStubFunctionsConsumer>(
      CI, InFile, "experimental-tapi-elf-v1");
}
