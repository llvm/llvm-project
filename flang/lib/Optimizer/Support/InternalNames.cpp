//===-- InternalNames.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Support/InternalNames.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/Support/CommandLine.h"

static llvm::cl::opt<std::string> mainEntryName(
    "main-entry-name",
    llvm::cl::desc("override the name of the default PROGRAM entry (may be "
                   "helpful for using other runtimes)"));

constexpr std::int64_t BAD_VALUE = -1;

inline std::string prefix() { return "_Q"; }

static std::string doModules(llvm::ArrayRef<llvm::StringRef> mods) {
  std::string result;
  auto *token = "M";
  for (auto mod : mods) {
    result.append(token).append(mod.lower());
    token = "S";
  }
  return result;
}

static std::string doModulesHost(llvm::ArrayRef<llvm::StringRef> mods,
                                 llvm::Optional<llvm::StringRef> host) {
  std::string result = doModules(mods);
  if (host.hasValue())
    result.append("F").append(host->lower());
  return result;
}

inline llvm::SmallVector<llvm::StringRef, 2>
convertToStringRef(llvm::ArrayRef<std::string> from) {
  return {from.begin(), from.end()};
}

inline llvm::Optional<llvm::StringRef>
convertToStringRef(const llvm::Optional<std::string> &from) {
  llvm::Optional<llvm::StringRef> to;
  if (from.hasValue())
    to = from.getValue();
  return to;
}

static std::string readName(llvm::StringRef uniq, std::size_t &i,
                            std::size_t init, std::size_t end) {
  for (i = init; i < end && (uniq[i] < 'A' || uniq[i] > 'Z'); ++i) {
    // do nothing
  }
  return uniq.substr(init, i - init).str();
}

static std::int64_t readInt(llvm::StringRef uniq, std::size_t &i,
                            std::size_t init, std::size_t end) {
  for (i = init; i < end && uniq[i] >= '0' && uniq[i] <= '9'; ++i) {
    // do nothing
  }
  std::int64_t result = BAD_VALUE;
  if (uniq.substr(init, i - init).getAsInteger(10, result))
    return BAD_VALUE;
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
  std::string result = prefix();
  return result.append("B").append(toLower(name));
}

std::string
fir::NameUniquer::doConstant(llvm::ArrayRef<llvm::StringRef> modules,
                             llvm::Optional<llvm::StringRef> host,
                             llvm::StringRef name) {
  std::string result = prefix();
  result.append(doModulesHost(modules, host)).append("EC");
  return result.append(toLower(name));
}

std::string
fir::NameUniquer::doDispatchTable(llvm::ArrayRef<llvm::StringRef> modules,
                                  llvm::Optional<llvm::StringRef> host,
                                  llvm::StringRef name,
                                  llvm::ArrayRef<std::int64_t> kinds) {
  std::string result = prefix();
  result.append(doModulesHost(modules, host)).append("DT");
  return result.append(toLower(name)).append(doKinds(kinds));
}

std::string fir::NameUniquer::doGenerated(llvm::StringRef name) {
  std::string result = prefix();
  return result.append("Q").append(name);
}

std::string fir::NameUniquer::doIntrinsicTypeDescriptor(
    llvm::ArrayRef<llvm::StringRef> modules,
    llvm::Optional<llvm::StringRef> host, IntrinsicType type,
    std::int64_t kind) {
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
  std::string result = prefix();
  result.append(doModulesHost(modules, host)).append("C");
  return result.append(name).append(doKind(kind));
}

std::string
fir::NameUniquer::doProcedure(llvm::ArrayRef<llvm::StringRef> modules,
                              llvm::Optional<llvm::StringRef> host,
                              llvm::StringRef name) {
  std::string result = prefix();
  result.append(doModulesHost(modules, host)).append("P");
  return result.append(toLower(name));
}

std::string fir::NameUniquer::doType(llvm::ArrayRef<llvm::StringRef> modules,
                                     llvm::Optional<llvm::StringRef> host,
                                     llvm::StringRef name,
                                     llvm::ArrayRef<std::int64_t> kinds) {
  std::string result = prefix();
  result.append(doModulesHost(modules, host)).append("T");
  return result.append(toLower(name)).append(doKinds(kinds));
}

std::string
fir::NameUniquer::doTypeDescriptor(llvm::ArrayRef<llvm::StringRef> modules,
                                   llvm::Optional<llvm::StringRef> host,
                                   llvm::StringRef name,
                                   llvm::ArrayRef<std::int64_t> kinds) {
  std::string result = prefix();
  result.append(doModulesHost(modules, host)).append("CT");
  return result.append(toLower(name)).append(doKinds(kinds));
}

std::string fir::NameUniquer::doTypeDescriptor(
    llvm::ArrayRef<std::string> modules, llvm::Optional<std::string> host,
    llvm::StringRef name, llvm::ArrayRef<std::int64_t> kinds) {
  auto rmodules = convertToStringRef(modules);
  auto rhost = convertToStringRef(host);
  return doTypeDescriptor(rmodules, rhost, name, kinds);
}

std::string
fir::NameUniquer::doVariable(llvm::ArrayRef<llvm::StringRef> modules,
                             llvm::Optional<llvm::StringRef> host,
                             llvm::StringRef name) {
  std::string result = prefix();
  result.append(doModulesHost(modules, host)).append("E");
  return result.append(toLower(name));
}

llvm::StringRef fir::NameUniquer::doProgramEntry() {
  if (mainEntryName.size())
    return mainEntryName;
  return "_QQmain";
}

std::pair<fir::NameUniquer::NameKind, fir::NameUniquer::DeconstructedName>
fir::NameUniquer::deconstruct(llvm::StringRef uniq) {
  if (uniq.startswith("_Q")) {
    llvm::SmallVector<std::string, 4> modules;
    llvm::Optional<std::string> host;
    std::string name;
    llvm::SmallVector<std::int64_t, 8> kinds;
    NameKind nk = NameKind::NOT_UNIQUED;
    for (std::size_t i = 2, end{uniq.size()}; i != end;) {
      switch (uniq[i]) {
      case 'B':
        nk = NameKind::COMMON;
        name = readName(uniq, i, i + 1, end);
        break;
      case 'C':
        if (uniq[i + 1] == 'T') {
          nk = NameKind::TYPE_DESC;
          name = readName(uniq, i, i + 2, end);
        } else {
          nk = NameKind::INTRINSIC_TYPE_DESC;
          name = readName(uniq, i, i + 1, end);
        }
        break;
      case 'D':
        nk = NameKind::DISPATCH_TABLE;
        assert(uniq[i + 1] == 'T');
        name = readName(uniq, i, i + 2, end);
        break;
      case 'E':
        if (uniq[i + 1] == 'C') {
          nk = NameKind::CONSTANT;
          name = readName(uniq, i, i + 2, end);
        } else {
          nk = NameKind::VARIABLE;
          name = readName(uniq, i, i + 1, end);
        }
        break;
      case 'P':
        nk = NameKind::PROCEDURE;
        name = readName(uniq, i, i + 1, end);
        break;
      case 'Q':
        nk = NameKind::GENERATED;
        name = uniq;
        i = end;
        break;
      case 'T':
        nk = NameKind::DERIVED_TYPE;
        name = readName(uniq, i, i + 1, end);
        break;

      case 'M':
      case 'S':
        modules.push_back(readName(uniq, i, i + 1, end));
        break;
      case 'F':
        host = readName(uniq, i, i + 1, end);
        break;
      case 'K':
        if (uniq[i + 1] == 'N')
          kinds.push_back(-readInt(uniq, i, i + 2, end));
        else
          kinds.push_back(readInt(uniq, i, i + 1, end));
        break;

      default:
        assert(false && "unknown uniquing code");
        break;
      }
    }
    return {nk, DeconstructedName(modules, host, name, kinds)};
  }
  return {NameKind::NOT_UNIQUED, DeconstructedName(uniq)};
}
