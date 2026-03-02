//===--- BuiltinsBase.td - common structured used by builtins -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Attributes
// ==========

class Attribute<string mangling> {
  string Mangling = mangling;
}

class IndexedAttribute<string baseMangling, int I> : Attribute<baseMangling> {
  int Index = I;
}

class MultiIndexAttribute<string baseMangling, list<int> Is>
    : Attribute<baseMangling> {
  list<int> Indices = Is;
}

// Standard Attributes
// -------------------
def NoReturn : Attribute<"r">;

// Attributes from the gnu:: namespace
// -----------------------------------
def Const : Attribute<"c">;
def NoThrow : Attribute<"n">;
def Pure : Attribute<"U">;
def ReturnsTwice : Attribute<"j">;

class NonNullOptMode<int mode> {
  int value = mode;
}

def NonOptimizing : NonNullOptMode<0>;
def Optimizing    : NonNullOptMode<1>;

class NonNull<NonNullOptMode Mode, list<int> Is> : MultiIndexAttribute<"N:" # Mode.value # ":", Is> {
  int optMode = Mode.value;
}

// builtin-specific attributes
// ---------------------------

// Signature is meaningless, use custom typechecking.
def CustomTypeChecking : Attribute<"t">;

// Type is not important to semantic analysis and codegen; recognize as builtin
// even if type doesn't match signature, and don't warn if we can't be sure the
// type is right.
def IgnoreSignature : Attribute<"T">;

// Arguments are not evaluated for their side-effects.
def UnevaluatedArguments : Attribute<"u">;

// FIXME: This is misused in a lot of the places it is used currently.
// This function is equivalent to a library function without the __builtin_
// prefix. This is relevant for CodeGen; it should not be used if custom CodeGen
// is required for a builtin.
def FunctionWithBuiltinPrefix : Attribute<"F">;

def FunctionWithoutBuiltinPrefix : Attribute<"f">;

// const, but only when -fno-math-errno and FP exceptions are ignored.
def ConstIgnoringErrnoAndExceptions : Attribute<"e">;

// const when FP exceptions are ignored.
def ConstIgnoringExceptions : Attribute<"g">;

// This function requires a specific header or an explicit declaration.
def RequireDeclaration : Attribute<"h">;

// FIXME: Why is this not simply the min_vector_width attribute?
// Vector has to be at least N bits wide.
class RequiredVectorWidth<int N> : IndexedAttribute<"V", N>;

class PrintfFormat<int I> : IndexedAttribute<"p", I>;
class VPrintfFormat<int I> : IndexedAttribute<"P", I>;
class ScanfFormat<int I> : IndexedAttribute<"s", I>;
class VScanfFormat<int I> : IndexedAttribute<"S", I>;

// Other Attributes
// ----------------

// Builtin can be constant evaluated
def Constexpr : Attribute<"E">;
// Builtin is immediate and must be constant evaluated. Implies Constexpr, and will only be supported in C++20 mode.
def Consteval : Attribute<"EG">;

// Callback behavior: the first index argument is called with the arguments
// indicated by the remaining indices.
class Callback<list<int> ArgIndices> : MultiIndexAttribute<"C", ArgIndices>;

// Prefixes
// ========

class NamePrefix<string spelling> {
  string Spelling = spelling;
}

// Documentation
// =============

class DocumentationCategory<string name> {
  string Name = name;
  code Content = [{}];
}

// This category is for builtins which have not yet been properly documented,
// but should be.
def DocCatUndocumented : DocumentationCategory<"Undocumented"> {
  let Content = [{
This section lists builtins which are recognized by Clang, but which are
currently missing documentation.
}];
}

// Builtins listed under the InternalOnly category do not generate any entry
// in the documentation. This category should be used only when we _want_
// to not document the builtin, e.g. if the builtin is for internal use only.
def DocCatInternalOnly : DocumentationCategory<"InternalOnly">;

// Specifies the documentation to be associated with the given category.
class Documentation {
  DocumentationCategory Category;
  code Content = [{}];

  // If the heading is empty, one may be picked automatically. If the builtin
  // only has one spelling, no heading is required as the builtin's sole
  // spelling is sufficient.
  string Heading = "";
}

// Specifies that the builtin is explicitly omitted from the documentation,
// because it is not intended to be user-facing.
def InternalOnly : Documentation { let Category = DocCatInternalOnly; }

// Specifies that the builtin is undocumented, but that it _should_ have
// documentation.
def Undocumented : Documentation {
  let Category = DocCatUndocumented;
  let Content = [{No documentation.}];
}

// Builtin kinds
// =============

class Builtin {
  list<string> Spellings;
  list<Attribute> Attributes = [];
  string Prototype;
  string Namespace;
  // On some platforms, some functions are actually macros. In that case we need
  // to #undef them.
  bit RequiresUndef = 0;
  // Enables builtins to generate `long long` outside of OpenCL and `long` inside.
  bit EnableOpenCLLong = 0;
  // Requires a common prefix to be prepended. Each generated set of builtins
  // can optionally extract one common prefix that is handled separately.
  NamePrefix RequiredNamePrefix;
  // Documentation associated with this builtin.
  list<Documentation> Documentation = [Undocumented];
  // Optional argument names for documentation. When provided, these are
  // interleaved with the parameter types from the Prototype field to produce
  // a human-readable prototype in the generated documentation.
  list<string> ArgNames = [];
}

class AtomicBuiltin : Builtin;

class LibBuiltin<string header, string languages = "ALL_LANGUAGES"> : Builtin {
  string Header = header;
  string Languages = languages;
  bit AddBuiltinPrefixedAlias = 0;
  bit OnlyBuiltinPrefixedAliasIsConstexpr = 0;
}

class MSLibBuiltin<string header> : LibBuiltin<header, "ALL_MS_LANGUAGES">;
class GNULibBuiltin<string header> : LibBuiltin<header, "ALL_GNU_LANGUAGES">;
class ObjCLibBuiltin<string header> : LibBuiltin<header, "OBJC_LANG">;
class CxxLibBuiltin<string header> : LibBuiltin<header, "CXX_LANG">;

class LangBuiltin<string languages> : Builtin {
  string Languages = languages;
}

class MSLangBuiltin : LangBuiltin<"ALL_MS_LANGUAGES">;
class CoroLangBuiltin : LangBuiltin<"COR_LANG">;
class OCLPipeLangBuiltin : LangBuiltin<"OCL_PIPE">;
class OCL_DSELangBuiltin : LangBuiltin<"OCL_DSE">;
class OCL_GASLangBuiltin : LangBuiltin<"OCL_GAS">;
class OCLLangBuiltin : LangBuiltin<"ALL_OCL_LANGUAGES">;

class TargetBuiltin : Builtin {
  string Features = "";
}
class TargetLibBuiltin : TargetBuiltin {
  string Header;
  string Languages = "ALL_LANGUAGES";
}

class Template<list<string> substitutions,
               list<string> affixes,
               bit as_prefix = 0> {
  list<string> Substitutions = substitutions;
  list<string> Affixes = affixes;
  bit AsPrefix = as_prefix;
}
