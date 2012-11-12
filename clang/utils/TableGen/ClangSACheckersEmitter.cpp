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

#include "llvm/ADT/DenseSet.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <map>
#include <string>
using namespace llvm;

//===----------------------------------------------------------------------===//
// Static Analyzer Checkers Tables generation
//===----------------------------------------------------------------------===//

/// \brief True if it is specified hidden or a parent package is specified
/// as hidden, otherwise false.
static bool isHidden(const Record &R) {
  if (R.getValueAsBit("Hidden"))
    return true;
  // Not declared as hidden, check the parent package if it is hidden.
  if (DefInit *DI = dyn_cast<DefInit>(R.getValueInit("ParentPackage")))
    return isHidden(*DI->getDef());

  return false;
}

static bool isCheckerNamed(const Record *R) {
  return !R->getValueAsString("CheckerName").empty();
}

static std::string getPackageFullName(const Record *R);

static std::string getParentPackageFullName(const Record *R) {
  std::string name;
  if (DefInit *DI = dyn_cast<DefInit>(R->getValueInit("ParentPackage")))
    name = getPackageFullName(DI->getDef());
  return name;
}

static std::string getPackageFullName(const Record *R) {
  std::string name = getParentPackageFullName(R);
  if (!name.empty()) name += ".";
  return name + R->getValueAsString("PackageName");
}

static std::string getCheckerFullName(const Record *R) {
  std::string name = getParentPackageFullName(R);
  if (isCheckerNamed(R)) {
    if (!name.empty()) name += ".";
    name += R->getValueAsString("CheckerName");
  }
  return name;
}

static std::string getStringValue(const Record &R, StringRef field) {
  if (StringInit *SI = dyn_cast<StringInit>(R.getValueInit(field)))
    return SI->getValue();
  return std::string();
}

namespace {
struct GroupInfo {
  llvm::DenseSet<const Record*> Checkers;
  llvm::DenseSet<const Record *> SubGroups;
  bool Hidden;
  unsigned Index;

  GroupInfo() : Hidden(false) { }
};
}

static void addPackageToCheckerGroup(const Record *package, const Record *group,
                  llvm::DenseMap<const Record *, GroupInfo *> &recordGroupMap) {
  llvm::DenseSet<const Record *> &checkers = recordGroupMap[package]->Checkers;
  for (llvm::DenseSet<const Record *>::iterator
         I = checkers.begin(), E = checkers.end(); I != E; ++I)
    recordGroupMap[group]->Checkers.insert(*I);

  llvm::DenseSet<const Record *> &subGroups = recordGroupMap[package]->SubGroups;
  for (llvm::DenseSet<const Record *>::iterator
         I = subGroups.begin(), E = subGroups.end(); I != E; ++I)
    addPackageToCheckerGroup(*I, group, recordGroupMap);
}

namespace clang {
void EmitClangSACheckers(RecordKeeper &Records, raw_ostream &OS) {
  std::vector<Record*> checkers = Records.getAllDerivedDefinitions("Checker");
  llvm::DenseMap<const Record *, unsigned> checkerRecIndexMap;
  for (unsigned i = 0, e = checkers.size(); i != e; ++i)
    checkerRecIndexMap[checkers[i]] = i;

  // Invert the mapping of checkers to package/group into a one to many
  // mapping of packages/groups to checkers.
  std::map<std::string, GroupInfo> groupInfoByName;
  llvm::DenseMap<const Record *, GroupInfo *> recordGroupMap;

  std::vector<Record*> packages = Records.getAllDerivedDefinitions("Package");
  for (unsigned i = 0, e = packages.size(); i != e; ++i) {
    Record *R = packages[i];
    std::string fullName = getPackageFullName(R);
    if (!fullName.empty()) {
      GroupInfo &info = groupInfoByName[fullName];
      info.Hidden = isHidden(*R);
      recordGroupMap[R] = &info;
    }
  }

  std::vector<Record*>
      checkerGroups = Records.getAllDerivedDefinitions("CheckerGroup");
  for (unsigned i = 0, e = checkerGroups.size(); i != e; ++i) {
    Record *R = checkerGroups[i];
    std::string name = R->getValueAsString("GroupName");
    if (!name.empty()) {
      GroupInfo &info = groupInfoByName[name];
      recordGroupMap[R] = &info;
    }
  }

  for (unsigned i = 0, e = checkers.size(); i != e; ++i) {
    Record *R = checkers[i];
    Record *package = 0;
    if (DefInit *
          DI = dyn_cast<DefInit>(R->getValueInit("ParentPackage")))
      package = DI->getDef();
    if (!isCheckerNamed(R) && !package)
      PrintFatalError(R->getLoc(), "Checker '" + R->getName() +
                      "' is neither named, nor in a package!");

    if (isCheckerNamed(R)) {
      // Create a pseudo-group to hold this checker.
      std::string fullName = getCheckerFullName(R);
      GroupInfo &info = groupInfoByName[fullName];
      info.Hidden = R->getValueAsBit("Hidden");
      recordGroupMap[R] = &info;
      info.Checkers.insert(R);
    } else {
      recordGroupMap[package]->Checkers.insert(R);
    }

    Record *currR = isCheckerNamed(R) ? R : package;
    // Insert the checker and its parent packages into the subgroups set of
    // the corresponding parent package.
    while (DefInit *DI
             = dyn_cast<DefInit>(currR->getValueInit("ParentPackage"))) {
      Record *parentPackage = DI->getDef();
      recordGroupMap[parentPackage]->SubGroups.insert(currR);
      currR = parentPackage;
    }
    // Insert the checker into the set of its group.
    if (DefInit *DI = dyn_cast<DefInit>(R->getValueInit("Group")))
      recordGroupMap[DI->getDef()]->Checkers.insert(R);
  }

  // If a package is in group, add all its checkers and its sub-packages
  // checkers into the group.
  for (unsigned i = 0, e = packages.size(); i != e; ++i)
    if (DefInit *DI = dyn_cast<DefInit>(packages[i]->getValueInit("Group")))
      addPackageToCheckerGroup(packages[i], DI->getDef(), recordGroupMap);

  typedef std::map<std::string, const Record *> SortedRecords;
  typedef llvm::DenseMap<const Record *, unsigned> RecToSortIndex;

  SortedRecords sortedGroups;
  RecToSortIndex groupToSortIndex;
  OS << "\n#ifdef GET_GROUPS\n";
  {
    for (unsigned i = 0, e = checkerGroups.size(); i != e; ++i)
      sortedGroups[checkerGroups[i]->getValueAsString("GroupName")]
                   = checkerGroups[i];

    unsigned sortIndex = 0;
    for (SortedRecords::iterator
           I = sortedGroups.begin(), E = sortedGroups.end(); I != E; ++I) {
      const Record *R = I->second;
  
      OS << "GROUP(" << "\"";
      OS.write_escaped(R->getValueAsString("GroupName")) << "\"";
      OS << ")\n";

      groupToSortIndex[R] = sortIndex++;
    }
  }
  OS << "#endif // GET_GROUPS\n\n";

  OS << "\n#ifdef GET_PACKAGES\n";
  {
    SortedRecords sortedPackages;
    for (unsigned i = 0, e = packages.size(); i != e; ++i)
      sortedPackages[getPackageFullName(packages[i])] = packages[i];
  
    for (SortedRecords::iterator
           I = sortedPackages.begin(), E = sortedPackages.end(); I != E; ++I) {
      const Record &R = *I->second;
  
      OS << "PACKAGE(" << "\"";
      OS.write_escaped(getPackageFullName(&R)) << "\", ";
      // Group index
      if (DefInit *DI = dyn_cast<DefInit>(R.getValueInit("Group")))
        OS << groupToSortIndex[DI->getDef()] << ", ";
      else
        OS << "-1, ";
      // Hidden bit
      if (isHidden(R))
        OS << "true";
      else
        OS << "false";
      OS << ")\n";
    }
  }
  OS << "#endif // GET_PACKAGES\n\n";
  
  OS << "\n#ifdef GET_CHECKERS\n";
  for (unsigned i = 0, e = checkers.size(); i != e; ++i) {
    const Record &R = *checkers[i];

    OS << "CHECKER(" << "\"";
    std::string name;
    if (isCheckerNamed(&R))
      name = getCheckerFullName(&R);
    OS.write_escaped(name) << "\", ";
    OS << R.getName() << ", ";
    OS << getStringValue(R, "DescFile") << ", ";
    OS << "\"";
    OS.write_escaped(getStringValue(R, "HelpText")) << "\", ";
    // Group index
    if (DefInit *DI = dyn_cast<DefInit>(R.getValueInit("Group")))
      OS << groupToSortIndex[DI->getDef()] << ", ";
    else
      OS << "-1, ";
    // Hidden bit
    if (isHidden(R))
      OS << "true";
    else
      OS << "false";
    OS << ")\n";
  }
  OS << "#endif // GET_CHECKERS\n\n";

  unsigned index = 0;
  for (std::map<std::string, GroupInfo>::iterator
         I = groupInfoByName.begin(), E = groupInfoByName.end(); I != E; ++I)
    I->second.Index = index++;

  // Walk through the packages/groups/checkers emitting an array for each
  // set of checkers and an array for each set of subpackages.

  OS << "\n#ifdef GET_MEMBER_ARRAYS\n";
  unsigned maxLen = 0;
  for (std::map<std::string, GroupInfo>::iterator
         I = groupInfoByName.begin(), E = groupInfoByName.end(); I != E; ++I) {
    maxLen = std::max(maxLen, (unsigned)I->first.size());

    llvm::DenseSet<const Record *> &checkers = I->second.Checkers;
    if (!checkers.empty()) {
      OS << "static const short CheckerArray" << I->second.Index << "[] = { ";
      // Make the output order deterministic.
      std::map<int, const Record *> sorted;
      for (llvm::DenseSet<const Record *>::iterator
             I = checkers.begin(), E = checkers.end(); I != E; ++I)
        sorted[(*I)->getID()] = *I;

      for (std::map<int, const Record *>::iterator
             I = sorted.begin(), E = sorted.end(); I != E; ++I)
        OS << checkerRecIndexMap[I->second] << ", ";
      OS << "-1 };\n";
    }
    
    llvm::DenseSet<const Record *> &subGroups = I->second.SubGroups;
    if (!subGroups.empty()) {
      OS << "static const short SubPackageArray" << I->second.Index << "[] = { ";
      // Make the output order deterministic.
      std::map<int, const Record *> sorted;
      for (llvm::DenseSet<const Record *>::iterator
             I = subGroups.begin(), E = subGroups.end(); I != E; ++I)
        sorted[(*I)->getID()] = *I;

      for (std::map<int, const Record *>::iterator
             I = sorted.begin(), E = sorted.end(); I != E; ++I) {
        OS << recordGroupMap[I->second]->Index << ", ";
      }
      OS << "-1 };\n";
    }
  }
  OS << "#endif // GET_MEMBER_ARRAYS\n\n";

  OS << "\n#ifdef GET_CHECKNAME_TABLE\n";
  for (std::map<std::string, GroupInfo>::iterator
         I = groupInfoByName.begin(), E = groupInfoByName.end(); I != E; ++I) {
    // Group option string.
    OS << "  { \"";
    OS.write_escaped(I->first) << "\","
                               << std::string(maxLen-I->first.size()+1, ' ');
    
    if (I->second.Checkers.empty())
      OS << "0, ";
    else
      OS << "CheckerArray" << I->second.Index << ", ";
    
    // Subgroups.
    if (I->second.SubGroups.empty())
      OS << "0, ";
    else
      OS << "SubPackageArray" << I->second.Index << ", ";

    OS << (I->second.Hidden ? "true" : "false");

    OS << " },\n";
  }
  OS << "#endif // GET_CHECKNAME_TABLE\n\n";
}
} // end namespace clang
