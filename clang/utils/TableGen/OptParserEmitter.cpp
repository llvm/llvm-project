//===- OptParserEmitter.cpp - Table Driven Command Line Parsing -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"

#include <map>

using namespace llvm;

static int StrCmpOptionName(const char *A, const char *B) {
  char a = *A, b = *B;
  while (a == b) {
    if (a == '\0')
      return 0;

    a = *++A;
    b = *++B;
  }

  if (a == '\0') // A is a prefix of B.
    return 1;
  if (b == '\0') // B is a prefix of A.
    return -1;

  // Otherwise lexicographic.
  return (a < b) ? -1 : 1;
}

static int CompareOptionRecords(const void *Av, const void *Bv) {
  const Record *A = *(const Record*const*) Av;
  const Record *B = *(const Record*const*) Bv;

  // Sentinel options precede all others and are only ordered by precedence.
  bool ASent = A->getValueAsDef("Kind")->getValueAsBit("Sentinel");
  bool BSent = B->getValueAsDef("Kind")->getValueAsBit("Sentinel");
  if (ASent != BSent)
    return ASent ? -1 : 1;

  // Compare options by name, unless they are sentinels.
  if (!ASent)
    if (int Cmp = StrCmpOptionName(A->getValueAsString("Name").c_str(),
                                   B->getValueAsString("Name").c_str()))
    return Cmp;

  if (!ASent) {
    std::vector<std::string> APrefixes = A->getValueAsListOfStrings("Prefixes");
    std::vector<std::string> BPrefixes = B->getValueAsListOfStrings("Prefixes");

    for (std::vector<std::string>::const_iterator APre = APrefixes.begin(),
                                                  AEPre = APrefixes.end(),
                                                  BPre = BPrefixes.begin(),
                                                  BEPre = BPrefixes.end();
                                                  APre != AEPre &&
                                                  BPre != BEPre;
                                                  ++APre, ++BPre) {
      if (int Cmp = StrCmpOptionName(APre->c_str(), BPre->c_str()))
        return Cmp;
    }
  }

  // Then by the kind precedence;
  int APrec = A->getValueAsDef("Kind")->getValueAsInt("Precedence");
  int BPrec = B->getValueAsDef("Kind")->getValueAsInt("Precedence");
  if (APrec == BPrec &&
      A->getValueAsListOfStrings("Prefixes") ==
      B->getValueAsListOfStrings("Prefixes")) {
    PrintError(A->getLoc(), Twine("Option is equivilent to"));
    PrintError(B->getLoc(), Twine("Other defined here"));
    PrintFatalError("Equivalent Options found.");
  }
  return APrec < BPrec ? -1 : 1;
}

static const std::string getOptionName(const Record &R) {
  // Use the record name unless EnumName is defined.
  if (isa<UnsetInit>(R.getValueInit("EnumName")))
    return R.getName();

  return R.getValueAsString("EnumName");
}

static raw_ostream &write_cstring(raw_ostream &OS, llvm::StringRef Str) {
  OS << '"';
  OS.write_escaped(Str);
  OS << '"';
  return OS;
}

/// OptParserEmitter - This tablegen backend takes an input .td file
/// describing a list of options and emits a data structure for parsing and
/// working with those options when given an input command line.
namespace clang {
void EmitOptParser(RecordKeeper &Records, raw_ostream &OS, bool GenDefs) {
  // Get the option groups and options.
  const std::vector<Record*> &Groups =
    Records.getAllDerivedDefinitions("OptionGroup");
  std::vector<Record*> Opts = Records.getAllDerivedDefinitions("Option");

  if (GenDefs)
    emitSourceFileHeader("Option Parsing Definitions", OS);
  else
    emitSourceFileHeader("Option Parsing Table", OS);

  array_pod_sort(Opts.begin(), Opts.end(), CompareOptionRecords);
  if (GenDefs) {
    // Generate prefix groups.
    typedef SmallVector<SmallString<2>, 2> PrefixKeyT;
    typedef std::map<PrefixKeyT, std::string> PrefixesT;
    PrefixesT Prefixes;
    Prefixes.insert(std::make_pair(PrefixKeyT(), "prefix_0"));
    unsigned CurPrefix = 0;
    for (unsigned i = 0, e = Opts.size(); i != e; ++i) {
      const Record &R = *Opts[i];
      std::vector<std::string> prf = R.getValueAsListOfStrings("Prefixes");
      PrefixKeyT prfkey(prf.begin(), prf.end());
      unsigned NewPrefix = CurPrefix + 1;
      if (Prefixes.insert(std::make_pair(prfkey, (Twine("prefix_") +
                                               Twine(NewPrefix)).str())).second)
        CurPrefix = NewPrefix;
    }

    OS << "#ifndef PREFIX\n";
    OS << "#error \"Define PREFIX prior to including this file!\"\n";
    OS << "#endif\n\n";

    // Dump prefixes.
    OS << "/////////\n";
    OS << "// Prefixes\n\n";
    OS << "#define COMMA ,\n";
    for (PrefixesT::const_iterator I = Prefixes.begin(), E = Prefixes.end();
                                   I != E; ++I) {
      OS << "PREFIX(";

      // Prefix name.
      OS << I->second;

      // Prefix values.
      OS << ", {";
      for (PrefixKeyT::const_iterator PI = I->first.begin(),
                                      PE = I->first.end(); PI != PE; ++PI) {
        OS << "\"" << *PI << "\" COMMA ";
      }
      OS << "0})\n";
    }
    OS << "#undef COMMA\n";
    OS << "\n";

    OS << "#ifndef OPTION\n";
    OS << "#error \"Define OPTION prior to including this file!\"\n";
    OS << "#endif\n\n";

    OS << "/////////\n";
    OS << "// Groups\n\n";
    for (unsigned i = 0, e = Groups.size(); i != e; ++i) {
      const Record &R = *Groups[i];

      // Start a single option entry.
      OS << "OPTION(";

      // The option prefix;
      OS << "0";

      // The option string.
      OS << ", \"" << R.getValueAsString("Name") << '"';

      // The option identifier name.
      OS  << ", "<< getOptionName(R);

      // The option kind.
      OS << ", Group";

      // The containing option group (if any).
      OS << ", ";
      if (const DefInit *DI = dyn_cast<DefInit>(R.getValueInit("Group")))
        OS << getOptionName(*DI->getDef());
      else
        OS << "INVALID";

      // The other option arguments (unused for groups).
      OS << ", INVALID, 0, 0";

      // The option help text.
      if (!isa<UnsetInit>(R.getValueInit("HelpText"))) {
        OS << ",\n";
        OS << "       ";
        write_cstring(OS, R.getValueAsString("HelpText"));
      } else
        OS << ", 0";

      // The option meta-variable name (unused).
      OS << ", 0)\n";
    }
    OS << "\n";

    OS << "//////////\n";
    OS << "// Options\n\n";
    for (unsigned i = 0, e = Opts.size(); i != e; ++i) {
      const Record &R = *Opts[i];

      // Start a single option entry.
      OS << "OPTION(";

      // The option prefix;
      std::vector<std::string> prf = R.getValueAsListOfStrings("Prefixes");
      OS << Prefixes[PrefixKeyT(prf.begin(), prf.end())] << ", ";

      // The option string.
      write_cstring(OS, R.getValueAsString("Name"));

      // The option identifier name.
      OS  << ", "<< getOptionName(R);

      // The option kind.
      OS << ", " << R.getValueAsDef("Kind")->getValueAsString("Name");

      // The containing option group (if any).
      OS << ", ";
      if (const DefInit *DI = dyn_cast<DefInit>(R.getValueInit("Group")))
        OS << getOptionName(*DI->getDef());
      else
        OS << "INVALID";

      // The option alias (if any).
      OS << ", ";
      if (const DefInit *DI = dyn_cast<DefInit>(R.getValueInit("Alias")))
        OS << getOptionName(*DI->getDef());
      else
        OS << "INVALID";

      // The option flags.
      const ListInit *LI = R.getValueAsListInit("Flags");
      if (LI->empty()) {
        OS << ", 0";
      } else {
        OS << ", ";
        for (unsigned i = 0, e = LI->size(); i != e; ++i) {
          if (i)
            OS << " | ";
          OS << cast<DefInit>(LI->getElement(i))->getDef()->getName();
        }
      }

      // The option parameter field.
      OS << ", " << R.getValueAsInt("NumArgs");

      // The option help text.
      if (!isa<UnsetInit>(R.getValueInit("HelpText"))) {
        OS << ",\n";
        OS << "       ";
        write_cstring(OS, R.getValueAsString("HelpText"));
      } else
        OS << ", 0";

      // The option meta-variable name.
      OS << ", ";
      if (!isa<UnsetInit>(R.getValueInit("MetaVarName")))
        write_cstring(OS, R.getValueAsString("MetaVarName"));
      else
        OS << "0";

      OS << ")\n";
    }
  }
}
} // end namespace clang
