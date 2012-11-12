//===- TreeView.cpp - diagtool tool for printing warning flags ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This diagnostic tool 
//
//===----------------------------------------------------------------------===//

#include "DiagTool.h"
#include "DiagnosticNames.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "llvm/Support/Format.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/DenseSet.h"
#include "clang/AST/ASTDiagnostic.h"
#include "clang/Basic/AllDiagnostics.h"
#include "llvm/Support/Process.h"

DEF_DIAGTOOL("tree",
             "Show warning flags in a tree view",
             TreeView)
  
using namespace clang;
using namespace diagtool;

static void printUsage() {
  llvm::errs() << "Usage: diagtool tree [--flags-only] [<diagnostic-group>]\n";
}

static bool showColors(llvm::raw_ostream &out) {
  if (&out != &llvm::errs() && &out != &llvm::outs())
    return false;
  return llvm::errs().is_displayed() && llvm::outs().is_displayed();
}

static void setColor(bool ShowColors, llvm::raw_ostream &out,
                     llvm::raw_ostream::Colors Color) {
  if (ShowColors)
    out << llvm::sys::Process::OutputColor(Color, false, false);
}

static void resetColor(bool ShowColors, llvm::raw_ostream &out) {
  if (ShowColors)
    out << llvm::sys::Process::ResetColor();
}

static clang::DiagnosticsEngine::Level getLevel(unsigned DiagID) {
  // FIXME: This feels like a hack.
  static clang::DiagnosticsEngine Diags(new DiagnosticIDs,
                                        new DiagnosticOptions);
  return Diags.getDiagnosticLevel(DiagID, SourceLocation());
}

static void printGroup(llvm::raw_ostream &out, const GroupRecord &Group,
                       bool FlagsOnly, unsigned Indent = 0) {
  out.indent(Indent * 2);
  
  bool ShowColors = showColors(out);
  setColor(ShowColors, out, llvm::raw_ostream::YELLOW);
  out << "-W" << Group.getName() << "\n";
  resetColor(ShowColors, out);
  
  ++Indent;
  for (GroupRecord::subgroup_iterator I = Group.subgroup_begin(),
                                      E = Group.subgroup_end();
       I != E; ++I) {
    printGroup(out, *I, FlagsOnly, Indent);
  }

  if (!FlagsOnly) {
    for (GroupRecord::diagnostics_iterator I = Group.diagnostics_begin(),
                                           E = Group.diagnostics_end();
         I != E; ++I) {
      if (ShowColors) {
        if (getLevel(I->DiagID) != DiagnosticsEngine::Ignored) {
          setColor(ShowColors, out, llvm::raw_ostream::GREEN);
        }
      }
      out.indent(Indent * 2);
      out << I->getName();
      resetColor(ShowColors, out);
      out << "\n";
    }
  }
}

static int showGroup(llvm::raw_ostream &out, StringRef RootGroup,
                     bool FlagsOnly) {
  ArrayRef<GroupRecord> AllGroups = getDiagnosticGroups();

  GroupRecord Key = { RootGroup.size(), RootGroup.data(), 0, 0 };
  const GroupRecord *Found =
    std::lower_bound(AllGroups.begin(), AllGroups.end(), Key);
  
  if (Found == AllGroups.end() || Found->getName() != RootGroup) {
    llvm::errs() << "No such diagnostic group exists\n";
    return 1;
  }
  
  printGroup(out, *Found, FlagsOnly);
  
  return 0;
}

static int showAll(llvm::raw_ostream &out, bool FlagsOnly) {
  ArrayRef<GroupRecord> AllGroups = getDiagnosticGroups();
  llvm::DenseSet<unsigned> NonRootGroupIDs;

  for (ArrayRef<GroupRecord>::iterator I = AllGroups.begin(),
                                       E = AllGroups.end();
       I != E; ++I) {
    for (GroupRecord::subgroup_iterator SI = I->subgroup_begin(),
                                        SE = I->subgroup_end();
         SI != SE; ++SI) {
      NonRootGroupIDs.insert((unsigned)SI.getID());
    }
  }

  assert(NonRootGroupIDs.size() < AllGroups.size());

  for (unsigned i = 0, e = AllGroups.size(); i != e; ++i) {
    if (!NonRootGroupIDs.count(i))
      printGroup(out, AllGroups[i], FlagsOnly);
  }

  return 0;
}

int TreeView::run(unsigned int argc, char **argv, llvm::raw_ostream &out) {
  // First check our one flag (--flags-only).
  bool FlagsOnly = false;
  if (argc > 0) {
    StringRef FirstArg(*argv);
    if (FirstArg.equals("--flags-only")) {
      FlagsOnly = true;
      --argc;
      ++argv;
    }
  }
  
  bool ShowAll = false;
  StringRef RootGroup;

  switch (argc) {
  case 0:
    ShowAll = true;
    break;
  case 1:
    RootGroup = argv[0];
    if (RootGroup.startswith("-W"))
      RootGroup = RootGroup.substr(2);
    if (RootGroup == "everything")
      ShowAll = true;
    // FIXME: Handle other special warning flags, like -pedantic.
    break;
  default:
    printUsage();
    return -1;
  }

  if (showColors(out)) {
    out << '\n';
    setColor(true, out, llvm::raw_ostream::GREEN);
    out << "GREEN";
    resetColor(true, out);
    out << " = enabled by default\n\n";
  }
  
  if (ShowAll)
    return showAll(out, FlagsOnly);

  return showGroup(out, RootGroup, FlagsOnly);
}

