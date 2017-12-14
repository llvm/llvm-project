//===--- SymbolYAML.cpp ------------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SymbolYAML.h"
#include "Index.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"

LLVM_YAML_IS_DOCUMENT_LIST_VECTOR(clang::clangd::Symbol)

namespace llvm {
namespace yaml {

using clang::clangd::Symbol;
using clang::clangd::SymbolID;
using clang::clangd::SymbolLocation;
using clang::index::SymbolInfo;
using clang::index::SymbolLanguage;
using clang::index::SymbolKind;

// Helper to (de)serialize the SymbolID. We serialize it as a hex string.
struct NormalizedSymbolID {
  NormalizedSymbolID(IO &) {}
  NormalizedSymbolID(IO &, const SymbolID& ID) {
    llvm::raw_string_ostream OS(HexString);
    OS << ID;
  }

  SymbolID denormalize(IO&) {
    SymbolID ID;
    HexString >> ID;
    return ID;
  }

  std::string HexString;
};

template <> struct MappingTraits<SymbolLocation> {
  static void mapping(IO &IO, SymbolLocation &Value) {
    IO.mapRequired("StartOffset", Value.StartOffset);
    IO.mapRequired("EndOffset", Value.EndOffset);
    IO.mapRequired("FilePath", Value.FilePath);
  }
};

template <> struct MappingTraits<SymbolInfo> {
  static void mapping(IO &io, SymbolInfo &SymInfo) {
    // FIXME: expose other fields?
    io.mapRequired("Kind", SymInfo.Kind);
    io.mapRequired("Lang", SymInfo.Lang);
  }
};

template<> struct MappingTraits<Symbol> {
  static void mapping(IO &IO, Symbol &Sym) {
    MappingNormalization<NormalizedSymbolID, SymbolID> NSymbolID(
        IO, Sym.ID);
    IO.mapRequired("ID", NSymbolID->HexString);
    IO.mapRequired("QualifiedName", Sym.QualifiedName);
    IO.mapRequired("SymInfo", Sym.SymInfo);
    IO.mapRequired("CanonicalDeclaration", Sym.CanonicalDeclaration);
  }
};

template <> struct ScalarEnumerationTraits<SymbolLanguage> {
  static void enumeration(IO &IO, SymbolLanguage &Value) {
    IO.enumCase(Value, "C", SymbolLanguage::C);
    IO.enumCase(Value, "Cpp", SymbolLanguage::CXX);
    IO.enumCase(Value, "ObjC", SymbolLanguage::ObjC);
    IO.enumCase(Value, "Swift", SymbolLanguage::Swift);
  }
};

template <> struct ScalarEnumerationTraits<SymbolKind> {
  static void enumeration(IO &IO, SymbolKind &Value) {
#define DEFINE_ENUM(name) IO.enumCase(Value, #name, SymbolKind::name)

    DEFINE_ENUM(Unknown);
    DEFINE_ENUM(Function);
    DEFINE_ENUM(Module);
    DEFINE_ENUM(Namespace);
    DEFINE_ENUM(NamespaceAlias);
    DEFINE_ENUM(Macro);
    DEFINE_ENUM(Enum);
    DEFINE_ENUM(Struct);
    DEFINE_ENUM(Class);
    DEFINE_ENUM(Protocol);
    DEFINE_ENUM(Extension);
    DEFINE_ENUM(Union);
    DEFINE_ENUM(TypeAlias);
    DEFINE_ENUM(Function);
    DEFINE_ENUM(Variable);
    DEFINE_ENUM(Field);
    DEFINE_ENUM(EnumConstant);
    DEFINE_ENUM(InstanceMethod);
    DEFINE_ENUM(ClassMethod);
    DEFINE_ENUM(StaticMethod);
    DEFINE_ENUM(InstanceProperty);
    DEFINE_ENUM(ClassProperty);
    DEFINE_ENUM(StaticProperty);
    DEFINE_ENUM(Constructor);
    DEFINE_ENUM(Destructor);
    DEFINE_ENUM(ConversionFunction);
    DEFINE_ENUM(Parameter);
    DEFINE_ENUM(Using);

#undef DEFINE_ENUM
  }
};

} // namespace yaml
} // namespace llvm

namespace clang {
namespace clangd {

SymbolSlab SymbolFromYAML(llvm::StringRef YAMLContent) {
  std::vector<Symbol> S;
  llvm::yaml::Input Yin(YAMLContent);
  Yin >> S;
  SymbolSlab Syms;
  for (auto& Sym : S)
    Syms.insert(std::move(Sym));
  return Syms;
}

std::string SymbolToYAML(const SymbolSlab& Symbols) {
  std::string Str;
  llvm::raw_string_ostream OS(Str);
  llvm::yaml::Output Yout(OS);
  for (auto &Pair : Symbols) {
    Symbol MutableSymbol = Pair.second;
    Yout<< MutableSymbol;
  }
  return OS.str();
}

} // namespace clangd
} // namespace clang
