//===--- OptParser.td - Common Option Parsing Interfaces ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the common interfaces used by the option parsing TableGen
//  backend.
//
//===----------------------------------------------------------------------===//

// Define the kinds of options.

class OptionKind<string name, int predecence = 0, bit sentinel = 0> {
  string Name = name;
  // The kind precedence, kinds with lower precedence are matched first.
  int Precedence = predecence;
  // Indicate a sentinel option.
  bit Sentinel = sentinel;
}

// An option group.
def KIND_GROUP : OptionKind<"Group">;
// The input option kind.
def KIND_INPUT : OptionKind<"Input", 1, 1>;
// The unknown option kind.
def KIND_UNKNOWN : OptionKind<"Unknown", 2, 1>;
// A flag with no values.
def KIND_FLAG : OptionKind<"Flag">;
// An option which prefixes its (single) value.
def KIND_JOINED : OptionKind<"Joined", 1>;
// An option which is followed by its value.
def KIND_SEPARATE : OptionKind<"Separate">;
// An option followed by its values, which are separated by commas.
def KIND_COMMAJOINED : OptionKind<"CommaJoined">;
// An option which is which takes multiple (separate) arguments.
def KIND_MULTIARG : OptionKind<"MultiArg">;
// An option which is either joined to its (non-empty) value, or followed by its
// value.
def KIND_JOINED_OR_SEPARATE : OptionKind<"JoinedOrSeparate">;
// An option which is both joined to its (first) value, and followed by its
// (second) value.
def KIND_JOINED_AND_SEPARATE : OptionKind<"JoinedAndSeparate">;

// Define the option flags.

class OptionFlag {}

// DriverOption - The option is a "driver" option, and should not be forwarded
// to gcc.
def DriverOption : OptionFlag;

// LinkerInput - The option is a linker input.
def LinkerInput : OptionFlag;

// NoArgumentUnused - Don't report argument unused warnings for this option; this
// is useful for options like -static or -dynamic which a user may always end up
// passing, even if the platform defaults to (or only supports) that option.
def NoArgumentUnused : OptionFlag;

// RenderAsInput - The option should not render the name when rendered as an
// input (i.e., the option is rendered as values).
def RenderAsInput : OptionFlag;

// RenderJoined - The option should be rendered joined, even if separate (only
// sensible on single value separate options).
def RenderJoined : OptionFlag;

// RenderSeparate - The option should be rendered separately, even if joined
// (only sensible on joined options).
def RenderSeparate : OptionFlag;

// Unsupported - The option is unsupported, and the driver will reject command
// lines that use it.
def Unsupported : OptionFlag;

// HelpHidden - The option should not be displayed in --help, even if it has
// help text. Clients *can* use this in conjunction with the OptTable::PrintHelp
// arguments to implement hidden help groups.
def HelpHidden : OptionFlag;

// NoForward - The option should not be implicitly forwarded to other tools.
def NoForward : OptionFlag;

// CC1Option - This option should be accepted by clang -cc1.
def CC1Option : OptionFlag;

// NoDriverOption - This option should not be accepted by the driver.
def NoDriverOption : OptionFlag;

// Define the option group class.

class OptionGroup<string name> {
  string EnumName = ?; // Uses the def name if undefined.
  string Name = name;
  string HelpText = ?;
  OptionGroup Group = ?;
}

// Define the option class.

class Option<list<string> prefixes, string name, OptionKind kind> {
  string EnumName = ?; // Uses the def name if undefined.
  list<string> Prefixes = prefixes;
  string Name = name;
  OptionKind Kind = kind;
  // Used by MultiArg option kind.
  int NumArgs = 0;
  string HelpText = ?;
  string MetaVarName = ?;
  list<OptionFlag> Flags = [];
  OptionGroup Group = ?;
  Option Alias = ?;
}

// Helpers for defining options.

class Flag<list<string> prefixes, string name>
  : Option<prefixes, name, KIND_FLAG>;
class Joined<list<string> prefixes, string name>
  : Option<prefixes, name, KIND_JOINED>;
class Separate<list<string> prefixes, string name>
  : Option<prefixes, name, KIND_SEPARATE>;
class CommaJoined<list<string> prefixes, string name>
  : Option<prefixes, name, KIND_COMMAJOINED>;
class MultiArg<list<string> prefixes, string name, int numargs>
  : Option<prefixes, name, KIND_MULTIARG> {
  int NumArgs = numargs;
}
class JoinedOrSeparate<list<string> prefixes, string name>
  : Option<prefixes, name, KIND_JOINED_OR_SEPARATE>;
class JoinedAndSeparate<list<string> prefixes, string name>
  : Option<prefixes, name, KIND_JOINED_AND_SEPARATE>;

// Mix-ins for adding optional attributes.

class Alias<Option alias> { Option Alias = alias; }
class EnumName<string name> { string EnumName = name; }
class Flags<list<OptionFlag> flags> { list<OptionFlag> Flags = flags; }
class Group<OptionGroup group> { OptionGroup Group = group; }
class HelpText<string text> { string HelpText = text; }
class MetaVarName<string name> { string MetaVarName = name; }

// Predefined options.

// FIXME: Have generator validate that these appear in correct position (and
// aren't duplicated).
def INPUT : Option<[], "<input>", KIND_INPUT>, Flags<[DriverOption,CC1Option]>;
def UNKNOWN : Option<[], "<unknown>", KIND_UNKNOWN>;
