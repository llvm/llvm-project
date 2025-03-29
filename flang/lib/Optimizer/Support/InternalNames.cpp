//===-- InternalNames.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/Support/CommandLine.h"
#include <optional>
#include <regex>

static llvm::cl::opt<std::string> mainEntryName(
    "main-entry-name",
    llvm::cl::desc("override the name of the default PROGRAM entry (may be "
                   "helpful for using other runtimes)"));

constexpr std::int64_t badValue = -1;

inline std::string prefix() { return "_Q"; }

/// Generate a mangling prefix from module, submodule, procedure, and
/// statement function names, plus an (innermost) block scope id.
static std::string doAncestors(llvm::ArrayRef<llvm::StringRef> modules,
                               llvm::ArrayRef<llvm::StringRef> procs,
                               std::int64_t blockId = 0) {
  std::string prefix;
  const char *tag = "M";
  for (auto mod : modules) {
    prefix.append(tag).append(mod.lower());
    tag = "S";
  }
  for (auto proc : procs)
    prefix.append("F").append(proc.lower());
  if (blockId)
    prefix.append("B").append(std::to_string(blockId));
  return prefix;
}

inline llvm::SmallVector<llvm::StringRef>
convertToStringRef(llvm::ArrayRef<std::string> from) {
  return {from.begin(), from.end()};
}

inline std::optional<llvm::StringRef>
convertToStringRef(const std::optional<std::string> &from) {
  std::optional<llvm::StringRef> to;
  if (from)
    to = *from;
  return to;
}

static std::string readName(llvm::StringRef uniq, std::size_t &i,
                            std::size_t init, std::size_t end) {
  // Allow 'X' to be part of the mangled name, which
  // can happen after the special symbols are replaced
  // in the mangled names by CompilerGeneratedNamesConversionPass.
  for (i = init; i < end && (uniq[i] < 'A' || uniq[i] > 'Z' || uniq[i] == 'X');
       ++i) {
    // do nothing
  }
  return uniq.substr(init, i - init).str();
}

static std::int64_t readInt(llvm::StringRef uniq, std::size_t &i,
                            std::size_t init, std::size_t end) {
  for (i = init; i < end && uniq[i] >= '0' && uniq[i] <= '9'; ++i) {
    // do nothing
  }
  std::int64_t result = badValue;
  if (uniq.substr(init, i - init).getAsInteger(10, result))
    return badValue;
  return result;
}

std::string fir::NameUniquer::toLower(llvm::StringRef name) {
  return name.lower();
}

std::string fir::NameUniquer::intAsString(std::int64_t i) {
  assert(i >= 0);
  return std::to_string(i);
}

std::string fir::NameUniquer::doKind(std::int64_t kind) {
  std::string result = "K";
  if (kind < 0)
    return result.append("N").append(intAsString(-kind));
  return result.append(intAsString(kind));
}

std::string fir::NameUniquer::doKinds(llvm::ArrayRef<std::int64_t> kinds) {
  std::string result;
  for (auto i : kinds)
    result.append(doKind(i));
  return result;
}

std::string fir::NameUniquer::doCommonBlock(llvm::StringRef name) {
  return prefix().append("C").append(toLower(name));
}

std::string
fir::NameUniquer::doConstant(llvm::ArrayRef<llvm::StringRef> modules,
                             llvm::ArrayRef<llvm::StringRef> procs,
                             std::int64_t blockId, llvm::StringRef name) {
  return prefix()
      .append(doAncestors(modules, procs, blockId))
      .append("EC")
      .append(toLower(name));
}

std::string
fir::NameUniquer::doDispatchTable(llvm::ArrayRef<llvm::StringRef> modules,
                                  llvm::ArrayRef<llvm::StringRef> procs,
                                  std::int64_t blockId, llvm::StringRef name,
                                  llvm::ArrayRef<std::int64_t> kinds) {
  return prefix()
      .append(doAncestors(modules, procs, blockId))
      .append("DT")
      .append(toLower(name))
      .append(doKinds(kinds));
}

std::string fir::NameUniquer::doGenerated(llvm::StringRef name) {
  return prefix().append("Q").append(name);
}

std::string
fir::NameUniquer::doGenerated(llvm::ArrayRef<llvm::StringRef> modules,
                              llvm::ArrayRef<llvm::StringRef> procs,
                              std::int64_t blockId, llvm::StringRef name) {
  return prefix()
      .append("Q")
      .append(doAncestors(modules, procs, blockId))
      .append(name);
}

std::string fir::NameUniquer::doIntrinsicTypeDescriptor(
    llvm::ArrayRef<llvm::StringRef> modules,
    llvm::ArrayRef<llvm::StringRef> procs, std::int64_t blockId,
    IntrinsicType type, std::int64_t kind) {
  const char *name = nullptr;
  switch (type) {
  case IntrinsicType::CHARACTER:
    name = "character";
    break;
  case IntrinsicType::COMPLEX:
    name = "complex";
    break;
  case IntrinsicType::INTEGER:
    name = "integer";
    break;
  case IntrinsicType::LOGICAL:
    name = "logical";
    break;
  case IntrinsicType::REAL:
    name = "real";
    break;
  }
  assert(name && "unknown intrinsic type");
  return prefix()
      .append(doAncestors(modules, procs, blockId))
      .append("YI")
      .append(name)
      .append(doKind(kind));
}

std::string
fir::NameUniquer::doProcedure(llvm::ArrayRef<llvm::StringRef> modules,
                              llvm::ArrayRef<llvm::StringRef> procs,
                              llvm::StringRef name) {
  return prefix()
      .append(doAncestors(modules, procs))
      .append("P")
      .append(toLower(name));
}

std::string fir::NameUniquer::doType(llvm::ArrayRef<llvm::StringRef> modules,
                                     llvm::ArrayRef<llvm::StringRef> procs,
                                     std::int64_t blockId, llvm::StringRef name,
                                     llvm::ArrayRef<std::int64_t> kinds) {
  return prefix()
      .append(doAncestors(modules, procs, blockId))
      .append("T")
      .append(toLower(name))
      .append(doKinds(kinds));
}

std::string
fir::NameUniquer::doTypeDescriptor(llvm::ArrayRef<llvm::StringRef> modules,
                                   llvm::ArrayRef<llvm::StringRef> procs,
                                   std::int64_t blockId, llvm::StringRef name,
                                   llvm::ArrayRef<std::int64_t> kinds) {
  return prefix()
      .append(doAncestors(modules, procs, blockId))
      .append("CT")
      .append(toLower(name))
      .append(doKinds(kinds));
}

std::string
fir::NameUniquer::doTypeDescriptor(llvm::ArrayRef<std::string> modules,
                                   llvm::ArrayRef<std::string> procs,
                                   std::int64_t blockId, llvm::StringRef name,
                                   llvm::ArrayRef<std::int64_t> kinds) {
  auto rmodules = convertToStringRef(modules);
  auto rprocs = convertToStringRef(procs);
  return doTypeDescriptor(rmodules, rprocs, blockId, name, kinds);
}

std::string
fir::NameUniquer::doVariable(llvm::ArrayRef<llvm::StringRef> modules,
                             llvm::ArrayRef<llvm::StringRef> procs,
                             std::int64_t blockId, llvm::StringRef name) {
  return prefix()
      .append(doAncestors(modules, procs, blockId))
      .append("E")
      .append(toLower(name));
}

std::string
fir::NameUniquer::doNamelistGroup(llvm::ArrayRef<llvm::StringRef> modules,
                                  llvm::ArrayRef<llvm::StringRef> procs,
                                  llvm::StringRef name) {
  return prefix()
      .append(doAncestors(modules, procs))
      .append("N")
      .append(toLower(name));
}

llvm::StringRef fir::NameUniquer::doProgramEntry() {
  if (mainEntryName.size())
    return mainEntryName;
  return "_QQmain";
}

std::pair<fir::NameUniquer::NameKind, fir::NameUniquer::DeconstructedName>
fir::NameUniquer::deconstruct(llvm::StringRef uniq) {
  uniq = fir::NameUniquer::dropTypeConversionMarkers(uniq);
  if (uniq.starts_with("_Q")) {
    llvm::SmallVector<std::string> modules;
    llvm::SmallVector<std::string> procs;
    std::int64_t blockId = 0;
    std::string name;
    llvm::SmallVector<std::int64_t> kinds;
    NameKind nk = NameKind::NOT_UNIQUED;
    for (std::size_t i = 2, end{uniq.size()}; i != end;) {
      switch (uniq[i]) {
      case 'B': // Block
        blockId = readInt(uniq, i, i + 1, end);
        break;
      case 'C': // Common block
        nk = NameKind::COMMON;
        name = readName(uniq, i, i + 1, end);
        break;
      case 'D': // Dispatch table
        nk = NameKind::DISPATCH_TABLE;
        assert(uniq[i + 1] == 'T');
        name = readName(uniq, i, i + 2, end);
        break;
      case 'E':
        if (uniq[i + 1] == 'C') { // Constant Entity
          nk = NameKind::CONSTANT;
          name = readName(uniq, i, i + 2, end);
        } else { // variable Entity
          nk = NameKind::VARIABLE;
          name = readName(uniq, i, i + 1, end);
        }
        break;
      case 'F': // procedure/Function ancestor component of a mangled prefix
        procs.push_back(readName(uniq, i, i + 1, end));
        break;
      case 'K':
        if (uniq[i + 1] == 'N') // Negative Kind
          kinds.push_back(-readInt(uniq, i, i + 2, end));
        else // [positive] Kind
          kinds.push_back(readInt(uniq, i, i + 1, end));
        break;
      case 'M': // Module
      case 'S': // Submodule
        modules.push_back(readName(uniq, i, i + 1, end));
        break;
      case 'N': // Namelist group
        nk = NameKind::NAMELIST_GROUP;
        name = readName(uniq, i, i + 1, end);
        break;
      case 'P': // Procedure/function (itself)
        nk = NameKind::PROCEDURE;
        name = readName(uniq, i, i + 1, end);
        break;
      case 'Q': // UniQue mangle name tag
        nk = NameKind::GENERATED;
        name = uniq;
        i = end;
        break;
      case 'T': // derived Type
        nk = NameKind::DERIVED_TYPE;
        name = readName(uniq, i, i + 1, end);
        break;
      case 'Y':
        if (uniq[i + 1] == 'I') { // tYpe descriptor for an Intrinsic type
          nk = NameKind::INTRINSIC_TYPE_DESC;
          name = readName(uniq, i, i + 1, end);
        } else { // tYpe descriptor
          nk = NameKind::TYPE_DESC;
          name = readName(uniq, i, i + 2, end);
        }
        break;
      default:
        assert(false && "unknown uniquing code");
        break;
      }
    }
    return {nk, DeconstructedName(modules, procs, blockId, name, kinds)};
  }
  return {NameKind::NOT_UNIQUED, DeconstructedName(uniq)};
}

bool fir::NameUniquer::isExternalFacingUniquedName(
    const std::pair<fir::NameUniquer::NameKind,
                    fir::NameUniquer::DeconstructedName> &deconstructResult) {
  return (deconstructResult.first == NameKind::PROCEDURE ||
          deconstructResult.first == NameKind::COMMON) &&
         deconstructResult.second.modules.empty() &&
         deconstructResult.second.procs.empty();
}

bool fir::NameUniquer::needExternalNameMangling(llvm::StringRef uniquedName) {
  auto result = fir::NameUniquer::deconstruct(uniquedName);
  return result.first != fir::NameUniquer::NameKind::NOT_UNIQUED &&
         fir::NameUniquer::isExternalFacingUniquedName(result);
}

bool fir::NameUniquer::belongsToModule(llvm::StringRef uniquedName,
                                       llvm::StringRef moduleName) {
  auto result = fir::NameUniquer::deconstruct(uniquedName);
  return !result.second.modules.empty() &&
         result.second.modules[0] == moduleName;
}

static std::string
mangleTypeDescriptorKinds(llvm::ArrayRef<std::int64_t> kinds) {
  if (kinds.empty())
    return "";
  std::string result;
  for (std::int64_t kind : kinds)
    result += (fir::kNameSeparator + std::to_string(kind)).str();
  return result;
}

static std::string getDerivedTypeObjectName(llvm::StringRef mangledTypeName,
                                            const llvm::StringRef separator) {
  mangledTypeName =
      fir::NameUniquer::dropTypeConversionMarkers(mangledTypeName);
  auto result = fir::NameUniquer::deconstruct(mangledTypeName);
  if (result.first != fir::NameUniquer::NameKind::DERIVED_TYPE)
    return "";
  std::string varName = separator.str() + result.second.name +
                        mangleTypeDescriptorKinds(result.second.kinds);
  llvm::SmallVector<llvm::StringRef> modules;
  for (const std::string &mod : result.second.modules)
    modules.push_back(mod);
  llvm::SmallVector<llvm::StringRef> procs;
  for (const std::string &proc : result.second.procs)
    procs.push_back(proc);
  return fir::NameUniquer::doVariable(modules, procs, result.second.blockId,
                                      varName);
}

std::string
fir::NameUniquer::getTypeDescriptorName(llvm::StringRef mangledTypeName) {
  return getDerivedTypeObjectName(mangledTypeName,
                                  fir::kTypeDescriptorSeparator);
}

std::string fir::NameUniquer::getTypeDescriptorAssemblyName(
    llvm::StringRef mangledTypeName) {
  return replaceSpecialSymbols(getTypeDescriptorName(mangledTypeName));
}

std::string fir::NameUniquer::getTypeDescriptorBindingTableName(
    llvm::StringRef mangledTypeName) {
  return getDerivedTypeObjectName(mangledTypeName, fir::kBindingTableSeparator);
}

std::string
fir::NameUniquer::getComponentInitName(llvm::StringRef mangledTypeName,
                                       llvm::StringRef componentName) {

  std::string prefix =
      getDerivedTypeObjectName(mangledTypeName, fir::kComponentInitSeparator);
  return (prefix + fir::kNameSeparator + componentName).str();
}

llvm::StringRef
fir::NameUniquer::dropTypeConversionMarkers(llvm::StringRef mangledTypeName) {
  if (mangledTypeName.ends_with(fir::boxprocSuffix))
    return mangledTypeName.drop_back(fir::boxprocSuffix.size());
  return mangledTypeName;
}

std::string fir::NameUniquer::replaceSpecialSymbols(const std::string &name) {
  return std::regex_replace(name, std::regex{"\\."}, "X");
}

bool fir::NameUniquer::isSpecialSymbol(llvm::StringRef name) {
  return !name.empty() && (name[0] == '.' || name[0] == 'X');
}
