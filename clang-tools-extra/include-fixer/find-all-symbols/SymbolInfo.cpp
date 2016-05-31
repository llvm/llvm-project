//===-- SymbolInfo.cpp - Symbol Info ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SymbolInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"

using llvm::yaml::MappingTraits;
using llvm::yaml::IO;
using llvm::yaml::Input;
using ContextType = clang::find_all_symbols::SymbolInfo::ContextType;
using clang::find_all_symbols::SymbolInfo;
using SymbolKind = clang::find_all_symbols::SymbolInfo::SymbolKind;

LLVM_YAML_IS_DOCUMENT_LIST_VECTOR(SymbolInfo)
LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(std::string)
LLVM_YAML_IS_SEQUENCE_VECTOR(SymbolInfo::Context)

namespace llvm {
namespace yaml {
template <> struct MappingTraits<SymbolInfo> {
  static void mapping(IO &io, SymbolInfo &Symbol) {
    io.mapRequired("Name", Symbol.Name);
    io.mapRequired("Contexts", Symbol.Contexts);
    io.mapRequired("FilePath", Symbol.FilePath);
    io.mapRequired("LineNumber", Symbol.LineNumber);
    io.mapRequired("Type", Symbol.Type);
    io.mapRequired("NumOccurrences", Symbol.NumOccurrences);
  }
};

template <> struct ScalarEnumerationTraits<ContextType> {
  static void enumeration(IO &io, ContextType &value) {
    io.enumCase(value, "Record", ContextType::Record);
    io.enumCase(value, "Namespace", ContextType::Namespace);
    io.enumCase(value, "EnumDecl", ContextType::EnumDecl);
  }
};

template <> struct ScalarEnumerationTraits<SymbolKind> {
  static void enumeration(IO &io, SymbolKind &value) {
    io.enumCase(value, "Variable", SymbolKind::Variable);
    io.enumCase(value, "Function", SymbolKind::Function);
    io.enumCase(value, "Class", SymbolKind::Class);
    io.enumCase(value, "TypedefName", SymbolKind::TypedefName);
    io.enumCase(value, "EnumDecl", SymbolKind::EnumDecl);
    io.enumCase(value, "EnumConstantDecl", SymbolKind::EnumConstantDecl);
    io.enumCase(value, "Macro", SymbolKind::Macro);
    io.enumCase(value, "Unknown", SymbolKind::Unknown);
  }
};

template <> struct MappingTraits<SymbolInfo::Context> {
  static void mapping(IO &io, SymbolInfo::Context &Context) {
    io.mapRequired("ContextType", Context.first);
    io.mapRequired("ContextName", Context.second);
  }
};

} // namespace yaml
} // namespace llvm

namespace clang {
namespace find_all_symbols {

SymbolInfo::SymbolInfo(llvm::StringRef Name, SymbolKind Type,
                       llvm::StringRef FilePath, int LineNumber,
                       const std::vector<Context> &Contexts,
                       unsigned NumOccurrences)
    : Name(Name), Type(Type), FilePath(FilePath), Contexts(Contexts),
      LineNumber(LineNumber), NumOccurrences(NumOccurrences) {}

bool SymbolInfo::operator==(const SymbolInfo &Symbol) const {
  return std::tie(Name, Type, FilePath, LineNumber, Contexts) ==
         std::tie(Symbol.Name, Symbol.Type, Symbol.FilePath, Symbol.LineNumber,
                  Symbol.Contexts);
}

bool SymbolInfo::operator<(const SymbolInfo &Symbol) const {
  return std::tie(Name, Type, FilePath, LineNumber, Contexts) <
         std::tie(Symbol.Name, Symbol.Type, Symbol.FilePath, Symbol.LineNumber,
                  Symbol.Contexts);
}

bool WriteSymbolInfosToStream(llvm::raw_ostream &OS,
                              const std::set<SymbolInfo> &Symbols) {
  llvm::yaml::Output yout(OS);
  for (auto Symbol : Symbols)
    yout << Symbol;
  return true;
}

std::vector<SymbolInfo> ReadSymbolInfosFromYAML(llvm::StringRef Yaml) {
  std::vector<SymbolInfo> Symbols;
  llvm::yaml::Input yin(Yaml);
  yin >> Symbols;
  return Symbols;
}

} // namespace find_all_symbols
} // namespace clang
