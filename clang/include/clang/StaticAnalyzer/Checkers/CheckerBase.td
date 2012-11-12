//===--- CheckerBase.td - Checker TableGen classes ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the TableGen core definitions for checkers
//
//===----------------------------------------------------------------------===//

class CheckerGroup<string name> {
  string GroupName = name;
}
class InGroup<CheckerGroup G> { CheckerGroup Group = G; }

class Package<string name> {
  string       PackageName = name;
  bit          Hidden = 0;
  Package ParentPackage;
  CheckerGroup Group;
}
class InPackage<Package P> { Package ParentPackage = P; }

// All checkers are an indirect subclass of this.
class Checker<string name = ""> {
  string      CheckerName = name;
  string      DescFile;
  string      HelpText;
  bit         Hidden = 0;
  Package ParentPackage;
  CheckerGroup Group;
}

class DescFile<string filename> { string DescFile = filename; }
class HelpText<string text> { string HelpText = text; }
class Hidden { bit Hidden = 1; }
