//===--- RefactoringOptions.cpp - A set of all the refactoring options ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Refactor/RefactoringOptions.h"
#include "llvm/Support/YAMLTraits.h"

using namespace clang;
using namespace clang::tooling;
using namespace clang::tooling::option;
using namespace llvm::yaml;

void RefactoringOptionSet::print(llvm::raw_ostream &OS) const {
  Output YamlOut(OS);
  if (YamlOut.preflightDocument(0)) {
    YamlOut.beginFlowMapping();
    for (const auto &Option : Options)
      Option.getValue()->serialize(YamlOut);
    YamlOut.endFlowMapping();
    YamlOut.postflightDocument();
  }
}

template <> struct CustomMappingTraits<RefactoringOptionSet> {
  static void inputOne(IO &YamlIn, StringRef Key,
                       RefactoringOptionSet &Result) {
#define HANDLE(Type)                                                           \
  if (Key == Type::Name) {                                                     \
    Type Value;                                                                \
    Value.serialize(YamlIn);                                                   \
    Result.add(Value);                                                         \
    return;                                                                    \
  }
    HANDLE(AvoidTextualMatches)
#undef HANDLE
    YamlIn.setError(Twine("Unknown refactoring option ") + Key);
  }
  static void output(IO &, RefactoringOptionSet &) {
    llvm_unreachable("Output is done without mapping traits");
  }
};

llvm::Expected<RefactoringOptionSet>
RefactoringOptionSet::parse(StringRef Source) {
  Input YamlIn(Source);
  // FIXME: Don't dump errors to stderr.
  RefactoringOptionSet Result;
  YamlIn >> Result;
  if (YamlIn.error())
    return llvm::make_error<llvm::StringError>("Failed to parse the option set",
                                               YamlIn.error());
  return std::move(Result);
}

void RefactoringOption::serialize(const SerializationContext &) {}

void clang::tooling::option::detail::BoolOptionBase::serializeImpl(
    const SerializationContext &Context, const char *Name) {
  Context.IO.mapRequired(Name, Value);
}
