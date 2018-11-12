//=- ClangSACheckersEmitter.cpp - Generate Clang SA checkers tables -*- C++ -*-
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend emits Clang Static Analyzer checkers tables.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringMap.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <map>
#include <string>

using namespace llvm;

//===----------------------------------------------------------------------===//
// Static Analyzer Checkers Tables generation
//===----------------------------------------------------------------------===//

static std::string getPackageFullName(const Record *R);

static std::string getParentPackageFullName(const Record *R) {
  std::string name;
  if (DefInit *DI = dyn_cast<DefInit>(R->getValueInit("ParentPackage")))
    name = getPackageFullName(DI->getDef());
  return name;
}

static std::string getPackageFullName(const Record *R) {
  std::string name = getParentPackageFullName(R);
  if (!name.empty())
    name += ".";
  assert(!R->getValueAsString("PackageName").empty());
  name += R->getValueAsString("PackageName");
  return name;
}

static std::string getCheckerFullName(const Record *R) {
  std::string name = getParentPackageFullName(R);
  if (!name.empty())
    name += ".";
  assert(!R->getValueAsString("CheckerName").empty());
  name += R->getValueAsString("CheckerName");
  return name;
}

static std::string getStringValue(const Record &R, StringRef field) {
  if (StringInit *SI = dyn_cast<StringInit>(R.getValueInit(field)))
    return SI->getValue();
  return std::string();
}

namespace clang {
void EmitClangSACheckers(RecordKeeper &Records, raw_ostream &OS) {
  std::vector<Record*> checkers = Records.getAllDerivedDefinitions("Checker");
  std::vector<Record*> packages = Records.getAllDerivedDefinitions("Package");

  using SortedRecords = llvm::StringMap<const Record *>;

  OS << "\n#ifdef GET_PACKAGES\n";
  {
    SortedRecords sortedPackages;
    for (unsigned i = 0, e = packages.size(); i != e; ++i)
      sortedPackages[getPackageFullName(packages[i])] = packages[i];
  
    for (SortedRecords::iterator
           I = sortedPackages.begin(), E = sortedPackages.end(); I != E; ++I) {
      const Record &R = *I->second;
  
      OS << "PACKAGE(" << "\"";
      OS.write_escaped(getPackageFullName(&R)) << '\"';
      OS << ")\n";
    }
  }
  OS << "#endif // GET_PACKAGES\n\n";
  
  OS << "\n#ifdef GET_CHECKERS\n";
  for (unsigned i = 0, e = checkers.size(); i != e; ++i) {
    const Record &R = *checkers[i];

    OS << "CHECKER(" << "\"";
    OS.write_escaped(getCheckerFullName(&R)) << "\", ";
    OS << R.getName() << ", ";
    OS << "\"";
    OS.write_escaped(getStringValue(R, "HelpText")) << '\"';
    OS << ")\n";
  }
  OS << "#endif // GET_CHECKERS\n\n";
}
} // end namespace clang
