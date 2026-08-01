//===--- Diagnostic.td - C Language Family Diagnostic Handling ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the TableGen core definitions for the diagnostics
//  and diagnostic control.
//
//===----------------------------------------------------------------------===//

// See the Internals Manual, section The Diagnostics Subsystem for an overview.

// Define the diagnostic severities.
class Severity<string N> {
  string Name = N;
}
def SEV_Ignored : Severity<"Ignored">;
def SEV_Remark  : Severity<"Remark">;
def SEV_Warning : Severity<"Warning">;
def SEV_Error   : Severity<"Error">;
def SEV_Fatal   : Severity<"Fatal">;

// Define the diagnostic classes.
class DiagClass;
def CLASS_NOTE      : DiagClass;
def CLASS_REMARK    : DiagClass;
def CLASS_WARNING   : DiagClass;
def CLASS_EXTENSION : DiagClass;
def CLASS_ERROR     : DiagClass;
def CLASS_TRAP      : DiagClass;

// Responses to a diagnostic in a SFINAE context.
class SFINAEResponse;
def SFINAE_SubstitutionFailure : SFINAEResponse;
def SFINAE_Suppress            : SFINAEResponse;
def SFINAE_Report              : SFINAEResponse;
def SFINAE_AccessControl       : SFINAEResponse;

// Textual substitutions which may be performed on the text of diagnostics
class TextSubstitution<string Text> {
  string Substitution = Text;
  // TODO: These are only here to allow substitutions to be declared inline with
  // diagnostics
  string Component = "";
  string CategoryName = "";
  bit Deferrable = 0;
}

// Diagnostic Categories.  These can be applied to groups or individual
// diagnostics to specify a category.
class DiagCategory<string Name> {
  string CategoryName = Name;
}

// Diagnostic Groups.
class DiagGroup<string Name, list<DiagGroup> subgroups = [], code docs = [{}]> {
  string GroupName = Name;
  list<DiagGroup> SubGroups = subgroups;
  string CategoryName = "";
  code Documentation = docs;
}
class InGroup<DiagGroup G> { DiagGroup Group = G; }
//class IsGroup<string Name> { DiagGroup Group = DiagGroup<Name>; }

// Specifies an explicit Stable ID for a diagnostic, rather than the default one
// generated from the diagnostic's name.
class StableId<string Id> { string StableId = Id; }

// Specifies a list of Stable IDs by which the diagnostic was previously known.
// If the Stable ID of a diagnostic is renamed, the previous Stable ID should be
// added to the LegacyStableIds list.
//
// If a diagnostic is split into two or more diagnostics for specific cases, the
// Stable ID of the original diagnostic should be added to the LegacyStableIds
// list of each of the new diagnostics created from it.
//
// If two or more diagnostics are merged into a single diagnostic, the Stable ID
// of each of the original diagnostics should be added to the LegacyStableIds
// list of the new (merged) diagnostic.
class LegacyStableIds<list<string> Ids> { list<string> LegacyStableIds = Ids; }

// This defines documentation for diagnostic groups.
include "DiagnosticDocs.td"

// This defines all of the named diagnostic categories.
include "DiagnosticCategories.td"

// This defines all of the named diagnostic groups.
include "DiagnosticGroups.td"


// All diagnostics emitted by the compiler are an indirect subclass of this.
class Diagnostic<string summary, DiagClass DC, Severity defaultmapping> {
  /// Component is specified by the file with a big let directive.
  string         Component = ?;
  string         Summary = summary;
  DiagClass      Class = DC;
  SFINAEResponse SFINAE = SFINAE_Suppress;
  bit            AccessControl = 0;
  bit            WarningNoWerror = 0;
  bit            ShowInSystemHeader = 0;
  bit            ShowInSystemMacro = 1;
  bit            Deferrable = 0;
  Severity       DefaultSeverity = defaultmapping;
  DiagGroup      Group;
  string         CategoryName = "";
  /// The default of "" means that the actual Stable ID should be generated from
  /// the name of the diagnostic.
  string         StableId = "";
  list<string>   LegacyStableIds = [];
}

class SFINAEFailure {
  SFINAEResponse SFINAE = SFINAE_SubstitutionFailure;
}
class NoSFINAE {
  SFINAEResponse SFINAE = SFINAE_Report;
}
class AccessControl {
  SFINAEResponse SFINAE = SFINAE_AccessControl;
}

class ShowInSystemHeader {
  bit ShowInSystemHeader = 1;
}

class SuppressInSystemHeader {
  bit ShowInSystemHeader = 0;
}

class ShowInSystemMacro {
  bit ShowInSystemMacro = 1;
}

class SuppressInSystemMacro {
  bit ShowInSystemMacro = 0;
}

class Deferrable {
  bit Deferrable = 1;
}

class NonDeferrable {
  bit Deferrable = 0;
}

// FIXME: ExtWarn and Extension should also be SFINAEFailure by default.
class Error<string str>     : Diagnostic<str, CLASS_ERROR, SEV_Error>, SFINAEFailure {
  bit ShowInSystemHeader = 1;
}
// Warnings default to on (but can be default-off'd with DefaultIgnore).
// This is used for warnings about questionable code; warnings about
// accepted language extensions should use Extension or ExtWarn below instead.
class Warning<string str>   : Diagnostic<str, CLASS_WARNING, SEV_Warning>;
// Remarks can be turned on with -R flags and provide commentary, e.g. on
// optimizer decisions.
class Remark<string str>    : Diagnostic<str, CLASS_REMARK, SEV_Ignored>;
// Extensions are warnings about accepted language extensions.
// Extension warnings are default-off but enabled by -pedantic.
class Extension<string str> : Diagnostic<str, CLASS_EXTENSION, SEV_Ignored>;
// ExtWarns are warnings about accepted language extensions.
// ExtWarn warnings are default-on.
class ExtWarn<string str>   : Diagnostic<str, CLASS_EXTENSION, SEV_Warning>;
// Notes can provide supplementary information on errors, warnings, and remarks.
class Note<string str>      : Diagnostic<str, CLASS_NOTE, SEV_Fatal/*ignored*/>;
// Trap messages attached to traps in debug info.
class Trap<string str>      : Diagnostic<str, CLASS_TRAP, SEV_Fatal/*ignored*/>;

class DefaultIgnore { Severity DefaultSeverity = SEV_Ignored; }
class DefaultWarn   { Severity DefaultSeverity = SEV_Warning; }
class DefaultError  { Severity DefaultSeverity = SEV_Error; }
class DefaultFatal  { Severity DefaultSeverity = SEV_Fatal; }
class DefaultWarnNoWerror {
  bit WarningNoWerror = 1;
}
class DefaultRemark { Severity DefaultSeverity = SEV_Remark; }

class CompatWarningId<string name, int std, string diag, string diag_pre> {
  string Component = ?;
  string Name = name;
  string Diag = diag;
  string DiagPre = diag_pre;
  int Std = std;

  // This is unused, but Tablegen will complain if it's missing because we define
  // the compatibility ids in the same place as the other diagnostics (which means
  // that we'll be inside a 'let CategoryName = "" in { ... }' block).
  string CategoryName = ?;
}

// Compatibility warnings.
multiclass CompatWarning<
    // Diagnostic message.
    string message,
    // Version number.
    int std_ver,
    // Is this C++?
    bit cxx,
    // ExtWarn if true, Extension if false.
    bit ext_warn = true,
    // Language mode name to use in the diagnostic text.
    string std_ver_name = ""#std_ver,
    // Version string used in diagnostic group names.
    string diag_group_ver_str = ""#std_ver> {
    defvar lang = !if(cxx, "C++", "C");
    defvar prefix = !if(cxx, "CXX", "C");
    defvar prefix_lower = !tolower(prefix#diag_group_ver_str);
    defvar is_cxx_11 = !and(cxx, !eq(std_ver, 11));

    // 'X is a C(++)YZ extension'.
    def compat_pre_#prefix_lower#_#NAME :
        Diagnostic<!strconcat(message, " a ", lang, std_ver_name,  " extension"),
                   CLASS_EXTENSION,
                   !if(ext_warn, SEV_Warning, SEV_Ignored)>,
        InGroup<!cast<DiagGroup>(prefix#diag_group_ver_str)>;

    // 'X is incompatible with C++98' (if is_cxx_11 is true).
    // 'X is incompatible with C(++) standards before C(++)YZ' (otherwise).
    def compat_#prefix_lower#_#NAME :
        Warning<!if(is_cxx_11,
                    !strconcat(message, " incompatible with C++98"),
                    !strconcat(message, " incompatible with ", lang, " standards before ", lang, std_ver_name))>,
        InGroup<!cast<DiagGroup>(!if(is_cxx_11,
                                     prefix#"98Compat",
                                     prefix#"Pre"#diag_group_ver_str#"Compat"))>,
        DefaultIgnore;

    def : CompatWarningId<
        NAME, std_ver,
        "compat_"#prefix_lower#"_"#NAME,
        "compat_pre_"#prefix_lower#"_"#NAME>;
}

// These generate pairs of C(++) compatibility warnings of the form:
//
//    - compat_c(xx)<std>_<name>
//    - compat_pre_c(xx)<std>_<name>
//
// The 'compat_c(xx)...' warning is intended to be issued in C(++)<std> mode,
// and the 'compat_pre_c(xx)...' warning in C(++) modes before C(++)<std>.
//
// Example:
//
//   defm inline_variable : C(XX)11Compat<"inline variables are">;
//
// This generates two warnings:
//
//   - compat_c(xx)11_inline_variable:     'inline variables are incompatible with C(++) standards before C(++)11'
//   - compat_pre_c(xx)11_inline_variable: 'inline variables are a C(++)11 extension'
//
multiclass CXX11Compat<string message, bit ext_warn = true> : CompatWarning<message, 11, /*cxx=*/true, ext_warn>;
multiclass CXX14Compat<string message, bit ext_warn = true> : CompatWarning<message, 14, /*cxx=*/true, ext_warn>;
multiclass CXX17Compat<string message, bit ext_warn = true> : CompatWarning<message, 17, /*cxx=*/true, ext_warn>;
multiclass CXX20Compat<string message, bit ext_warn = true> : CompatWarning<message, 20, /*cxx=*/true, ext_warn>;
multiclass CXX23Compat<string message, bit ext_warn = true> : CompatWarning<message, 23, /*cxx=*/true, ext_warn>;
multiclass CXX26Compat<string message, bit ext_warn = true> : CompatWarning<message, 26, /*cxx=*/true, ext_warn, "2c">;
multiclass CXX29Compat<string message, bit ext_warn = true> : CompatWarning<message, 29, /*cxx=*/true, ext_warn, "2d">;

// C compatibility warnings generally use Extension rather than ExtWarn.
multiclass C99Compat<string message, bit ext_warn = false> : CompatWarning<message, 99, /*cxx=*/false, ext_warn>;
multiclass C11Compat<string message, bit ext_warn = false> : CompatWarning<message, 11, /*cxx=*/false, ext_warn>;
multiclass C17Compat<string message, bit ext_warn = false> : CompatWarning<message, 17, /*cxx=*/false, ext_warn>;
multiclass C23Compat<string message, bit ext_warn = false> : CompatWarning<message, 23, /*cxx=*/false, ext_warn>;
multiclass C2yCompat<string message, bit ext_warn = false> : CompatWarning<message, 29, /*cxx=*/false, ext_warn, "2y", "2y">;

// Definitions for Diagnostics.
include "DiagnosticASTKinds.td"
include "DiagnosticCommentKinds.td"
include "DiagnosticCommonKinds.td"
include "DiagnosticCrossTUKinds.td"
include "DiagnosticDriverKinds.td"
include "DiagnosticFrontendKinds.td"
include "DiagnosticInstallAPIKinds.td"
include "DiagnosticLexKinds.td"
include "DiagnosticParseKinds.td"
include "DiagnosticRefactoringKinds.td"
include "DiagnosticSemaKinds.td"
include "DiagnosticSerializationKinds.td"
include "DiagnosticTrapKinds.td"
