//===-- Implementation of PublicAPICommand --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PublicAPICommand.h"

#include "utils/LibcTableGenUtil/APIIndexer.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/TableGen/Record.h"
#include <algorithm>
#include <vector>

// Text blocks for macro definitions and type decls can be indented to
// suit the surrounding tablegen listing. We need to dedent such blocks
// before writing them out.
static void dedentAndWrite(llvm::StringRef Text, llvm::raw_ostream &OS) {
  llvm::SmallVector<llvm::StringRef, 10> Lines;
  llvm::SplitString(Text, Lines, "\n");
  size_t shortest_indent = 1024;
  for (llvm::StringRef L : Lines) {
    llvm::StringRef Indent = L.take_while([](char c) { return c == ' '; });
    size_t IndentSize = Indent.size();
    if (Indent.size() == L.size()) {
      // Line is all spaces so no point noting the indent.
      continue;
    }
    if (IndentSize < shortest_indent)
      shortest_indent = IndentSize;
  }
  for (llvm::StringRef L : Lines) {
    if (L.size() >= shortest_indent)
      OS << L.drop_front(shortest_indent) << '\n';
  }
}

static std::string getTypeHdrName(const std::string &Name) {
  llvm::SmallVector<llvm::StringRef> Parts;
  llvm::SplitString(llvm::StringRef(Name), Parts);
  return llvm::join(Parts.begin(), Parts.end(), "_");
}

namespace llvm_libc {

static bool isAsciiStart(char C) {
  return (C >= 'A' && C <= 'Z') || (C >= 'a' && C <= 'z') || C == '_';
}

static bool isAsciiContinue(char C) {
  return isAsciiStart(C) || (C >= '0' && C <= '9');
}

static bool isAsciiIdentifier(llvm::StringRef S) {
  if (S.empty())
    return false;
  if (!isAsciiStart(S[0]))
    return false;
  for (char C : S.drop_front())
    if (!isAsciiContinue(C))
      return false;
  return true;
}

static AttributeStyle getAttributeStyle(const llvm::Record *Instance) {
  llvm::StringRef Style = Instance->getValueAsString("Style");
  return llvm::StringSwitch<AttributeStyle>(Style)
      .Case("cxx11", AttributeStyle::Cxx11)
      .Case("gnu", AttributeStyle::Gnu)
      .Case("declspec", AttributeStyle::Declspec)
      .Default(AttributeStyle::Gnu);
}

static AttributeNamespace getAttributeNamespace(const llvm::Record *Instance) {
  llvm::StringRef Namespace = Instance->getValueAsString("Namespace");
  return llvm::StringSwitch<AttributeNamespace>(Namespace)
      .Case("clang", AttributeNamespace::Clang)
      .Case("gnu", AttributeNamespace::Gnu)
      .Default(AttributeNamespace::None);
}

using AttributeMap = llvm::DenseMap<llvm::StringRef, const llvm::Record *>;

template <class SpecMap, class FuncList>
static AttributeMap collectAttributeMacros(const SpecMap &Spec,
                                           const FuncList &Funcs) {
  llvm::DenseMap<llvm::StringRef, const llvm::Record *> MacroAttr;
  for (const auto &Name : Funcs) {
    auto Iter = Spec.find(Name);
    if (Iter == Spec.end())
      continue;

    const llvm::Record *FunctionSpec = Iter->second;
    for (const llvm::Record *Attr :
         FunctionSpec->getValueAsListOfDefs("Attributes"))
      MacroAttr[Attr->getValueAsString("Macro")] = Attr;
  }
  return MacroAttr;
}

static void emitAttributeMacroDecls(const AttributeMap &MacroAttr,
                                    llvm::raw_ostream &OS) {
  for (auto &[Macro, Attr] : MacroAttr) {
    std::vector<const llvm::Record *> Instances =
        Attr->getValueAsListOfDefs("Instances");
    llvm::SmallVector<std::pair<AttributeStyle, const llvm::Record *>> Styles;
    std::transform(Instances.begin(), Instances.end(),
                   std::back_inserter(Styles),
                   [&](const llvm::Record *Instance)
                       -> std::pair<AttributeStyle, const llvm::Record *> {
                     auto Style = getAttributeStyle(Instance);
                     return {Style, Instance};
                   });
    // 1. If __cplusplus is defined and cxx11 style is provided, define the
    // macro using cxx11 version with the following priority:
    //    1a. If there is no namespace (so the macro is supposed to be
    //        compiler-independent), use this version first. This macro will be
    //        tested via __has_cpp_attribute.
    //    1b. If the attribute is a clang attribute, check for __clang__.
    //    1c. If the attribute is a gnu attribute, check for __GNUC__.
    // 2. Otherwise, if __GNUC__ is defined and gnu style is provided,
    //    define the macro using gnu version;
    // 3. Otherwise, if _MSC_VER is defined and __declspec is provided, define
    //    the macro using __declspec version;
    // 4. Fallback to empty macro.
    std::sort(Styles.begin(), Styles.end(), [&](auto &a, auto &b) {
      if (a.first == AttributeStyle::Cxx11 && b.first == AttributeStyle::Cxx11)
        return getAttributeNamespace(a.second) <
               getAttributeNamespace(b.second);
      return a.first < b.first;
    });
    for (auto &[Style, Instance] : Styles) {
      llvm::StringRef Attr = Instance->getValueAsString("Attr");
      if (Style == AttributeStyle::Cxx11) {
        OS << "#if !defined(" << Macro << ") && defined(__cplusplus)";
        AttributeNamespace Namespace = getAttributeNamespace(Instance);
        if (Namespace == AttributeNamespace::Clang)
          OS << " && defined(__clang__)\n";
        else if (Namespace == AttributeNamespace::Gnu)
          OS << " && defined(__GNUC__)\n";
        else
          OS << '\n';
        if (isAsciiIdentifier(Attr) && Namespace != AttributeNamespace::None)
          OS << "#if __has_attribute(" << Attr << ")\n";
        else
          OS << "#if __has_cpp_attribute(" << Attr << ")\n";
        OS << "#define " << Macro << " [[";
        if (Namespace == AttributeNamespace::Clang)
          OS << "clang::";
        else if (Namespace == AttributeNamespace::Gnu)
          OS << "gnu::";
        OS << Attr << "]]\n";
        if (isAsciiIdentifier(Attr))
          OS << "#endif\n";
        OS << "#endif\n";
      }
      if (Style == AttributeStyle::Gnu) {
        OS << "#if !defined(" << Macro << ") && defined(__GNUC__)\n";
        if (isAsciiIdentifier(Attr))
          OS << "#if __has_attribute(" << Attr << ")\n";
        OS << "#define " << Macro << " __attribute__((";
        OS << Attr << "))\n";
        if (isAsciiIdentifier(Attr))
          OS << "#endif\n";
        OS << "#endif\n";
      }
      if (Style == AttributeStyle::Declspec) {
        OS << "#if !defined(" << Macro << ") && defined(_MSC_VER)\n";
        OS << "#define " << Macro << " __declspec(";
        OS << Attr << ")\n";
        OS << "#endif\n";
      }
    }
    OS << "#if !defined(" << Macro << ")\n";
    OS << "#define " << Macro << '\n';
    OS << "#endif\n";
  }

  if (!MacroAttr.empty())
    OS << '\n';
}

static void emitAttributeMacroForFunction(const llvm::Record *FunctionSpec,
                                          llvm::raw_ostream &OS) {
  std::vector<const llvm::Record *> Attributes =
      FunctionSpec->getValueAsListOfDefs("Attributes");
  llvm::interleave(
      Attributes.begin(), Attributes.end(),
      [&](const llvm::Record *Attr) { OS << Attr->getValueAsString("Macro"); },
      [&]() { OS << ' '; });
  if (!Attributes.empty())
    OS << ' ';
}

static void emitUndefsForAttributeMacros(const AttributeMap &MacroAttr,
                                         llvm::raw_ostream &OS) {
  if (!MacroAttr.empty())
    OS << '\n';
  for (auto &[Macro, Attr] : MacroAttr)
    OS << "#undef " << Macro << '\n';
}

static void writeAPIFromIndex(APIIndexer &G,
                              std::vector<std::string> EntrypointNameList,
                              llvm::raw_ostream &OS) {
  for (auto &Pair : G.MacroDefsMap) {
    const std::string &Name = Pair.first;
    if (!G.MacroSpecMap.count(Name))
      llvm::PrintFatalError(Name + " not found in any standard spec.\n");

    const llvm::Record *MacroDef = Pair.second;
    dedentAndWrite(MacroDef->getValueAsString("Defn"), OS);

    OS << '\n';
  }

  for (auto &TypeName : G.RequiredTypes) {
    if (!G.TypeSpecMap.count(TypeName))
      llvm::PrintFatalError(TypeName + " not found in any standard spec.\n");
    OS << "#include <llvm-libc-types/" << getTypeHdrName(TypeName) << ".h>\n";
  }
  OS << '\n';

  if (G.Enumerations.size() != 0)
    OS << "enum {" << '\n';
  for (const auto &Name : G.Enumerations) {
    if (!G.EnumerationSpecMap.count(Name))
      llvm::PrintFatalError(
          Name + " is not listed as an enumeration in any standard spec.\n");

    const llvm::Record *EnumerationSpec = G.EnumerationSpecMap[Name];
    OS << "  " << EnumerationSpec->getValueAsString("Name");
    auto Value = EnumerationSpec->getValueAsString("Value");
    if (Value == "__default__") {
      OS << ",\n";
    } else {
      OS << " = " << Value << ",\n";
    }
  }
  if (G.Enumerations.size() != 0)
    OS << "};\n\n";

  // Collect and declare macros for attributes
  AttributeMap MacroAttr =
      collectAttributeMacros(G.FunctionSpecMap, EntrypointNameList);
  emitAttributeMacroDecls(MacroAttr, OS);

  OS << "__BEGIN_C_DECLS\n\n";
  for (auto &Name : EntrypointNameList) {
    auto Iter = G.FunctionSpecMap.find(Name);

    // Functions that aren't in this header file are skipped as
    // opposed to erroring out because the list of functions being
    // iterated over is the complete list of functions with
    // entrypoints. Thus this is filtering out the functions that
    // don't go to this header file, whereas the other, similar
    // conditionals above are more of a sanity check.
    if (Iter == G.FunctionSpecMap.end())
      continue;

    const llvm::Record *FunctionSpec = Iter->second;
    const llvm::Record *RetValSpec = FunctionSpec->getValueAsDef("Return");
    const llvm::Record *ReturnType = RetValSpec->getValueAsDef("ReturnType");

    // TODO: https://github.com/llvm/llvm-project/issues/81208
    //   Ideally, we should group functions based on their guarding macros.
    bool Guarded =
        (FunctionSpec->getType()->getAsString() == "GuardedFunctionSpec");

    if (Guarded)
      OS << "#ifdef " << FunctionSpec->getValueAsString("Guard") << "\n";

    // Emit attribute macros for the function. Space is automatically added.
    emitAttributeMacroForFunction(FunctionSpec, OS);
    OS << G.getTypeAsString(ReturnType) << " " << Name << "(";

    auto ArgsList = FunctionSpec->getValueAsListOfDefs("Args");
    for (size_t i = 0; i < ArgsList.size(); ++i) {
      const llvm::Record *ArgType = ArgsList[i]->getValueAsDef("ArgType");
      OS << G.getTypeAsString(ArgType);
      if (i < ArgsList.size() - 1)
        OS << ", ";
    }

    OS << ") __NOEXCEPT;\n";

    if (Guarded)
      OS << "#endif // " << FunctionSpec->getValueAsString("Guard") << "\n";

    OS << "\n";
  }

  // Make another pass over entrypoints to emit object declarations.
  for (const auto &Name : EntrypointNameList) {
    auto Iter = G.ObjectSpecMap.find(Name);
    if (Iter == G.ObjectSpecMap.end())
      continue;
    const llvm::Record *ObjectSpec = Iter->second;
    auto Type = ObjectSpec->getValueAsString("Type");
    OS << "extern " << Type << " " << Name << ";\n";
  }
  OS << "__END_C_DECLS\n";

  // Undef file-level attribute macros.
  emitUndefsForAttributeMacros(MacroAttr, OS);
}

void writePublicAPI(llvm::raw_ostream &OS, const llvm::RecordKeeper &Records) {}

const char PublicAPICommand::Name[] = "public_api";

void PublicAPICommand::run(llvm::raw_ostream &OS, const ArgVector &Args,
                           llvm::StringRef StdHeader,
                           const llvm::RecordKeeper &Records,
                           const Command::ErrorReporter &Reporter) const {
  if (Args.size() != 0)
    Reporter.printFatalError("public_api command does not take any arguments.");

  APIIndexer G(StdHeader, Records);
  writeAPIFromIndex(G, EntrypointNameList, OS);
}

} // namespace llvm_libc
