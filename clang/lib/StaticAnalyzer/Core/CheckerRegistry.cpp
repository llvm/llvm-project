//===- CheckerRegistry.cpp - Maintains all available checkers -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/CheckerRegistry.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LLVM.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/CheckerOptInfo.h"
#include "clang/StaticAnalyzer/Core/AnalyzerOptions.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstddef>
#include <tuple>

using namespace clang;
using namespace ento;

static const char PackageSeparator = '.';

using CheckerInfoSet = llvm::SetVector<const CheckerRegistry::CheckerInfo *>;

static bool checkerNameLT(const CheckerRegistry::CheckerInfo &a,
                          const CheckerRegistry::CheckerInfo &b) {
  return a.FullName < b.FullName;
}

static bool isInPackage(const CheckerRegistry::CheckerInfo &checker,
                        StringRef packageName) {
  // Does the checker's full name have the package as a prefix?
  if (!checker.FullName.startswith(packageName))
    return false;

  // Is the package actually just the name of a specific checker?
  if (checker.FullName.size() == packageName.size())
    return true;

  // Is the checker in the package (or a subpackage)?
  if (checker.FullName[packageName.size()] == PackageSeparator)
    return true;

  return false;
}

static void collectCheckers(const CheckerRegistry::CheckerInfoList &checkers,
                            const llvm::StringMap<size_t> &packageSizes,
                            CheckerOptInfo &opt, CheckerInfoSet &collected) {
  // Use a binary search to find the possible start of the package.
  CheckerRegistry::CheckerInfo packageInfo(nullptr, opt.getName(), "");
  auto end = checkers.cend();
  auto i = std::lower_bound(checkers.cbegin(), end, packageInfo, checkerNameLT);

  // If we didn't even find a possible package, give up.
  if (i == end)
    return;

  // If what we found doesn't actually start the package, give up.
  if (!isInPackage(*i, opt.getName()))
    return;

  // There is at least one checker in the package; claim the option.
  opt.claim();

  // See how large the package is.
  // If the package doesn't exist, assume the option refers to a single checker.
  size_t size = 1;
  llvm::StringMap<size_t>::const_iterator packageSize =
    packageSizes.find(opt.getName());
  if (packageSize != packageSizes.end())
    size = packageSize->getValue();

  // Step through all the checkers in the package.
  for (auto checkEnd = i+size; i != checkEnd; ++i)
    if (opt.isEnabled())
      collected.insert(&*i);
    else
      collected.remove(&*i);
}

void CheckerRegistry::addChecker(InitializationFunction fn, StringRef name,
                                 StringRef desc) {
  Checkers.push_back(CheckerInfo(fn, name, desc));

  // Record the presence of the checker in its packages.
  StringRef packageName, leafName;
  std::tie(packageName, leafName) = name.rsplit(PackageSeparator);
  while (!leafName.empty()) {
    Packages[packageName] += 1;
    std::tie(packageName, leafName) = packageName.rsplit(PackageSeparator);
  }
}

void CheckerRegistry::initializeManager(CheckerManager &checkerMgr,
                                  SmallVectorImpl<CheckerOptInfo> &opts) const {
  // Sort checkers for efficient collection.
  llvm::sort(Checkers.begin(), Checkers.end(), checkerNameLT);

  // Collect checkers enabled by the options.
  CheckerInfoSet enabledCheckers;
  for (auto &i : opts)
    collectCheckers(Checkers, Packages, i, enabledCheckers);

  // Initialize the CheckerManager with all enabled checkers.
  for (const auto *i :enabledCheckers) {
    checkerMgr.setCurrentCheckName(CheckName(i->FullName));
    i->Initialize(checkerMgr);
  }
}

void CheckerRegistry::validateCheckerOptions(const AnalyzerOptions &opts,
                                             DiagnosticsEngine &diags) const {
  for (const auto &config : opts.Config) {
    size_t pos = config.getKey().find(':');
    if (pos == StringRef::npos)
      continue;

    bool hasChecker = false;
    StringRef checkerName = config.getKey().substr(0, pos);
    for (const auto &checker : Checkers) {
      if (checker.FullName.startswith(checkerName) &&
          (checker.FullName.size() == pos || checker.FullName[pos] == '.')) {
        hasChecker = true;
        break;
      }
    }
    if (!hasChecker)
      diags.Report(diag::err_unknown_analyzer_checker) << checkerName;
  }
}

void CheckerRegistry::printHelp(raw_ostream &out,
                                size_t maxNameChars) const {
  // FIXME: Alphabetical sort puts 'experimental' in the middle.
  // Would it be better to name it '~experimental' or something else
  // that's ASCIIbetically last?
  llvm::sort(Checkers.begin(), Checkers.end(), checkerNameLT);

  // FIXME: Print available packages.

  out << "CHECKERS:\n";

  // Find the maximum option length.
  size_t optionFieldWidth = 0;
  for (const auto &i : Checkers) {
    // Limit the amount of padding we are willing to give up for alignment.
    //   Package.Name     Description  [Hidden]
    size_t nameLength = i.FullName.size();
    if (nameLength <= maxNameChars)
      optionFieldWidth = std::max(optionFieldWidth, nameLength);
  }

  const size_t initialPad = 2;
  for (const auto &i : Checkers) {
    out.indent(initialPad) << i.FullName;

    int pad = optionFieldWidth - i.FullName.size();

    // Break on long option names.
    if (pad < 0) {
      out << '\n';
      pad = optionFieldWidth + initialPad;
    }
    out.indent(pad + 2) << i.Desc;

    out << '\n';
  }
}

void CheckerRegistry::printList(
    raw_ostream &out, SmallVectorImpl<CheckerOptInfo> &opts) const {
  llvm::sort(Checkers.begin(), Checkers.end(), checkerNameLT);

  // Collect checkers enabled by the options.
  CheckerInfoSet enabledCheckers;
  for (auto &i : opts)
    collectCheckers(Checkers, Packages, i, enabledCheckers);

  for (const auto *i : enabledCheckers)
    out << i->FullName << '\n';
}
