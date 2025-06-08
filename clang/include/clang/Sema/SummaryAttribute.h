#ifndef LLVM_CLANG_SEMA_SEMASUMMARYATTRIBUTE_H
#define LLVM_CLANG_SEMA_SEMASUMMARYATTRIBUTE_H

#include "clang/AST/Decl.h"
#include <string>

namespace clang {
enum SummaryAttribute {
  NO_WRITE_GLOBAL,
};

class FunctionSummary;

class SummaryAttributeManager {
  inline static std::unordered_map<SummaryAttribute, std::string> AttrToStr;

protected:
  const SummaryAttribute Attr;
  const char *Str;

public:
  SummaryAttributeManager(SummaryAttribute Attr, const char *Str)
      : Attr(Attr), Str(Str) {
    assert(AttrToStr.count(Attr) == 0 && "attribute already registered");
    for (auto &&[attr, str] : AttrToStr)
      assert(str != Str && "attribute representation is already used");

    AttrToStr[Attr] = Str;
  }
  virtual ~SummaryAttributeManager() = default;

  virtual bool predicate(const FunctionDecl *FD) = 0;

  // FIXME: This should receive all the parsed summaries as well.
  virtual bool merge(FunctionSummary &Summary) = 0;

  virtual std::string serialize() const { return Str; };
  virtual std::optional<SummaryAttribute> parse(std::string_view Input) const {
    if (Str == Input)
      return Attr;

    return std::nullopt;
  };

  std::optional<SummaryAttribute> infer(const FunctionDecl *FD) {
    if (predicate(FD))
      return Attr;

    return std::nullopt;
  };
};
} // namespace clang

#endif // LLVM_CLANG_SEMA_SEMASUMMARYATTRIBUTEH
