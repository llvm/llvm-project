//===--- Diagnostic.td - C Language Family Diagnostic Handling ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the TableGen core definitions for the diagnostics
//  and diagnostic control.
//
//===----------------------------------------------------------------------===//

// Define the diagnostic mappings.
class DiagMapping;
def MAP_IGNORE  : DiagMapping;
def MAP_WARNING : DiagMapping;
def MAP_ERROR   : DiagMapping;
def MAP_FATAL   : DiagMapping;

// Define the diagnostic classes.
class DiagClass;
def CLASS_NOTE      : DiagClass;
def CLASS_WARNING   : DiagClass;
def CLASS_EXTENSION : DiagClass;
def CLASS_ERROR     : DiagClass;

// Diagnostic Categories.  These can be applied to groups or individual
// diagnostics to specify a category.
class DiagCategory<string Name> {
  string CategoryName = Name;
}

// Diagnostic Groups.
class DiagGroup<string Name, list<DiagGroup> subgroups = []> {
  string GroupName = Name;
  list<DiagGroup> SubGroups = subgroups;
  string CategoryName = "";
}
class InGroup<DiagGroup G> { DiagGroup Group = G; }
//class IsGroup<string Name> { DiagGroup Group = DiagGroup<Name>; }


// This defines all of the named diagnostic categories.
include "DiagnosticCategories.td"

// This defines all of the named diagnostic groups.
include "DiagnosticGroups.td"


// All diagnostics emitted by the compiler are an indirect subclass of this.
class Diagnostic<string text, DiagClass DC, DiagMapping defaultmapping> {
  /// Component is specified by the file with a big let directive.
  string      Component = ?;
  string      Text = text;
  DiagClass   Class = DC;
  bit         SFINAE = 1;
  bit         AccessControl = 0;
  bit         WarningNoWerror = 0;
  bit         WarningShowInSystemHeader = 0;
  DiagMapping DefaultMapping = defaultmapping;
  DiagGroup   Group;
  string      CategoryName = "";
}

class Error<string str>     : Diagnostic<str, CLASS_ERROR, MAP_ERROR>;
class Warning<string str>   : Diagnostic<str, CLASS_WARNING, MAP_WARNING>;
class Extension<string str> : Diagnostic<str, CLASS_EXTENSION, MAP_IGNORE>;
class ExtWarn<string str>   : Diagnostic<str, CLASS_EXTENSION, MAP_WARNING>;
class Note<string str>      : Diagnostic<str, CLASS_NOTE, MAP_FATAL/*ignored*/>;


class DefaultIgnore { DiagMapping DefaultMapping = MAP_IGNORE; }
class DefaultWarn   { DiagMapping DefaultMapping = MAP_WARNING; }
class DefaultError  { DiagMapping DefaultMapping = MAP_ERROR; }
class DefaultFatal  { DiagMapping DefaultMapping = MAP_FATAL; }
class DefaultWarnNoWerror {
  bit WarningNoWerror = 1;
}
class DefaultWarnShowInSystemHeader {
  bit WarningShowInSystemHeader = 1;
}

class NoSFINAE { bit SFINAE = 0; }
class AccessControl { bit AccessControl = 1; }

// Definitions for Diagnostics.
include "DiagnosticASTKinds.td"
include "DiagnosticAnalysisKinds.td"
include "DiagnosticCommentKinds.td"
include "DiagnosticCommonKinds.td"
include "DiagnosticDriverKinds.td"
include "DiagnosticFrontendKinds.td"
include "DiagnosticLexKinds.td"
include "DiagnosticParseKinds.td"
include "DiagnosticSemaKinds.td"
include "DiagnosticSerializationKinds.td"

