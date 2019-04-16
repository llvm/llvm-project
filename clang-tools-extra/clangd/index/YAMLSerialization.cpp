//===-- YAMLSerialization.cpp ------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A YAML index file is a sequence of tagged entries.
// Each entry either encodes a Symbol or the list of references to a symbol
// (a "ref bundle").
//
//===----------------------------------------------------------------------===//

#include "Index.h"
#include "Serialization.h"
#include "SymbolLocation.h"
#include "SymbolOrigin.h"
#include "Trace.h"
#include "dex/Dex.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>

LLVM_YAML_IS_SEQUENCE_VECTOR(clang::clangd::Symbol::IncludeHeaderWithReferences)
LLVM_YAML_IS_SEQUENCE_VECTOR(clang::clangd::Ref)

namespace {
using RefBundle =
    std::pair<clang::clangd::SymbolID, std::vector<clang::clangd::Ref>>;
// This is a pale imitation of std::variant<Symbol, RefBundle>
struct VariantEntry {
  llvm::Optional<clang::clangd::Symbol> Symbol;
  llvm::Optional<RefBundle> Refs;
};
// A class helps YAML to serialize the 32-bit encoded position (Line&Column),
// as YAMLIO can't directly map bitfields.
struct YPosition {
  uint32_t Line;
  uint32_t Column;
};

} // namespace
namespace llvm {
namespace yaml {

using clang::clangd::Ref;
using clang::clangd::RefKind;
using clang::clangd::Symbol;
using clang::clangd::SymbolID;
using clang::clangd::SymbolLocation;
using clang::clangd::SymbolOrigin;
using clang::index::SymbolInfo;
using clang::index::SymbolKind;
using clang::index::SymbolLanguage;

// Helper to (de)serialize the SymbolID. We serialize it as a hex string.
struct NormalizedSymbolID {
  NormalizedSymbolID(IO &) {}
  NormalizedSymbolID(IO &, const SymbolID &ID) {
    llvm::raw_string_ostream OS(HexString);
    OS << ID;
  }

  SymbolID denormalize(IO &I) {
    auto ID = SymbolID::fromStr(HexString);
    if (!ID) {
      I.setError(llvm::toString(ID.takeError()));
      return SymbolID();
    }
    return *ID;
  }

  std::string HexString;
};

struct NormalizedSymbolFlag {
  NormalizedSymbolFlag(IO &) {}
  NormalizedSymbolFlag(IO &, Symbol::SymbolFlag F) {
    Flag = static_cast<uint8_t>(F);
  }

  Symbol::SymbolFlag denormalize(IO &) {
    return static_cast<Symbol::SymbolFlag>(Flag);
  }

  uint8_t Flag = 0;
};

struct NormalizedSymbolOrigin {
  NormalizedSymbolOrigin(IO &) {}
  NormalizedSymbolOrigin(IO &, SymbolOrigin O) {
    Origin = static_cast<uint8_t>(O);
  }

  SymbolOrigin denormalize(IO &) { return static_cast<SymbolOrigin>(Origin); }

  uint8_t Origin = 0;
};

template <> struct MappingTraits<YPosition> {
  static void mapping(IO &IO, YPosition &Value) {
    IO.mapRequired("Line", Value.Line);
    IO.mapRequired("Column", Value.Column);
  }
};

struct NormalizedPosition {
  using Position = clang::clangd::SymbolLocation::Position;
  NormalizedPosition(IO &) {}
  NormalizedPosition(IO &, const Position &Pos) {
    P.Line = Pos.line();
    P.Column = Pos.column();
  }

  Position denormalize(IO &) {
    Position Pos;
    Pos.setLine(P.Line);
    Pos.setColumn(P.Column);
    return Pos;
  }
  YPosition P;
};

struct NormalizedFileURI {
  NormalizedFileURI(IO &) {}
  NormalizedFileURI(IO &, const char *FileURI) { URI = FileURI; }

  const char *denormalize(IO &IO) {
    assert(IO.getContext() &&
           "Expecting an UniqueStringSaver to allocate data");
    return static_cast<llvm::UniqueStringSaver *>(IO.getContext())
        ->save(URI)
        .data();
  }

  std::string URI;
};

template <> struct MappingTraits<SymbolLocation> {
  static void mapping(IO &IO, SymbolLocation &Value) {
    MappingNormalization<NormalizedFileURI, const char *> NFile(IO,
                                                                Value.FileURI);
    IO.mapRequired("FileURI", NFile->URI);
    MappingNormalization<NormalizedPosition, SymbolLocation::Position> NStart(
        IO, Value.Start);
    IO.mapRequired("Start", NStart->P);
    MappingNormalization<NormalizedPosition, SymbolLocation::Position> NEnd(
        IO, Value.End);
    IO.mapRequired("End", NEnd->P);
  }
};

template <> struct MappingTraits<SymbolInfo> {
  static void mapping(IO &io, SymbolInfo &SymInfo) {
    // FIXME: expose other fields?
    io.mapRequired("Kind", SymInfo.Kind);
    io.mapRequired("Lang", SymInfo.Lang);
  }
};

template <>
struct MappingTraits<clang::clangd::Symbol::IncludeHeaderWithReferences> {
  static void mapping(IO &io,
                      clang::clangd::Symbol::IncludeHeaderWithReferences &Inc) {
    io.mapRequired("Header", Inc.IncludeHeader);
    io.mapRequired("References", Inc.References);
  }
};

template <> struct MappingTraits<Symbol> {
  static void mapping(IO &IO, Symbol &Sym) {
    MappingNormalization<NormalizedSymbolID, SymbolID> NSymbolID(IO, Sym.ID);
    MappingNormalization<NormalizedSymbolFlag, Symbol::SymbolFlag> NSymbolFlag(
        IO, Sym.Flags);
    MappingNormalization<NormalizedSymbolOrigin, SymbolOrigin> NSymbolOrigin(
        IO, Sym.Origin);
    IO.mapRequired("ID", NSymbolID->HexString);
    IO.mapRequired("Name", Sym.Name);
    IO.mapRequired("Scope", Sym.Scope);
    IO.mapRequired("SymInfo", Sym.SymInfo);
    IO.mapOptional("CanonicalDeclaration", Sym.CanonicalDeclaration,
                   SymbolLocation());
    IO.mapOptional("Definition", Sym.Definition, SymbolLocation());
    IO.mapOptional("References", Sym.References, 0u);
    IO.mapOptional("Origin", NSymbolOrigin->Origin);
    IO.mapOptional("Flags", NSymbolFlag->Flag);
    IO.mapOptional("Signature", Sym.Signature);
    IO.mapOptional("TemplateSpecializationArgs",
                   Sym.TemplateSpecializationArgs);
    IO.mapOptional("CompletionSnippetSuffix", Sym.CompletionSnippetSuffix);
    IO.mapOptional("Documentation", Sym.Documentation);
    IO.mapOptional("ReturnType", Sym.ReturnType);
    IO.mapOptional("Type", Sym.Type);
    IO.mapOptional("IncludeHeaders", Sym.IncludeHeaders);
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

template <> struct MappingTraits<RefBundle> {
  static void mapping(IO &IO, RefBundle &Refs) {
    MappingNormalization<NormalizedSymbolID, SymbolID> NSymbolID(IO,
                                                                 Refs.first);
    IO.mapRequired("ID", NSymbolID->HexString);
    IO.mapRequired("References", Refs.second);
  }
};

struct NormalizedRefKind {
  NormalizedRefKind(IO &) {}
  NormalizedRefKind(IO &, RefKind O) { Kind = static_cast<uint8_t>(O); }

  RefKind denormalize(IO &) { return static_cast<RefKind>(Kind); }

  uint8_t Kind = 0;
};

template <> struct MappingTraits<Ref> {
  static void mapping(IO &IO, Ref &R) {
    MappingNormalization<NormalizedRefKind, RefKind> NKind(IO, R.Kind);
    IO.mapRequired("Kind", NKind->Kind);
    IO.mapRequired("Location", R.Location);
  }
};

template <> struct MappingTraits<VariantEntry> {
  static void mapping(IO &IO, VariantEntry &Variant) {
    if (IO.mapTag("!Symbol", Variant.Symbol.hasValue())) {
      if (!IO.outputting())
        Variant.Symbol.emplace();
      MappingTraits<Symbol>::mapping(IO, *Variant.Symbol);
    } else if (IO.mapTag("!Refs", Variant.Refs.hasValue())) {
      if (!IO.outputting())
        Variant.Refs.emplace();
      MappingTraits<RefBundle>::mapping(IO, *Variant.Refs);
    }
  }
};

} // namespace yaml
} // namespace llvm

namespace clang {
namespace clangd {

void writeYAML(const IndexFileOut &O, llvm::raw_ostream &OS) {
  llvm::yaml::Output Yout(OS);
  for (const auto &Sym : *O.Symbols) {
    VariantEntry Entry;
    Entry.Symbol = Sym;
    Yout << Entry;
  }
  if (O.Refs)
    for (auto &Sym : *O.Refs) {
      VariantEntry Entry;
      Entry.Refs = Sym;
      Yout << Entry;
    }
}

llvm::Expected<IndexFileIn> readYAML(llvm::StringRef Data) {
  SymbolSlab::Builder Symbols;
  RefSlab::Builder Refs;
  llvm::BumpPtrAllocator
      Arena; // store the underlying data of Position::FileURI.
  llvm::UniqueStringSaver Strings(Arena);
  llvm::yaml::Input Yin(Data, &Strings);
  while (Yin.setCurrentDocument()) {
    llvm::yaml::EmptyContext Ctx;
    VariantEntry Variant;
    yamlize(Yin, Variant, true, Ctx);
    if (Yin.error())
      return llvm::errorCodeToError(Yin.error());

    if (Variant.Symbol)
      Symbols.insert(*Variant.Symbol);
    if (Variant.Refs)
      for (const auto &Ref : Variant.Refs->second)
        Refs.insert(Variant.Refs->first, Ref);
    Yin.nextDocument();
  }

  IndexFileIn Result;
  Result.Symbols.emplace(std::move(Symbols).build());
  Result.Refs.emplace(std::move(Refs).build());
  return std::move(Result);
}

std::string toYAML(const Symbol &S) {
  std::string Buf;
  {
    llvm::raw_string_ostream OS(Buf);
    llvm::yaml::Output Yout(OS);
    Symbol Sym = S; // copy: Yout<< requires mutability.
    Yout << Sym;
  }
  return Buf;
}

std::string toYAML(const std::pair<SymbolID, llvm::ArrayRef<Ref>> &Data) {
  RefBundle Refs = {Data.first, Data.second};
  std::string Buf;
  {
    llvm::raw_string_ostream OS(Buf);
    llvm::yaml::Output Yout(OS);
    Yout << Refs;
  }
  return Buf;
}

} // namespace clangd
} // namespace clang
