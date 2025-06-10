#ifndef LLVM_CLANG_SEMA_SEMASUMMARYATTRIBUTE_H
#define LLVM_CLANG_SEMA_SEMASUMMARYATTRIBUTE_H

#include "clang/AST/Decl.h"
#include <string>

namespace clang {
enum SummaryAttribute {
  NO_WRITE_GLOBAL,
};

class FunctionSummary;

class SummaryAttributeDescription {
protected:
  const SummaryAttribute Attr;
  std::string_view Serialzed;

public:
  SummaryAttributeDescription(SummaryAttribute Attr, const char *Str) : Attr(Attr), Serialzed(Str) {}
  virtual ~SummaryAttributeDescription() = default;

  SummaryAttribute getAttribute() { return Attr; };

  virtual bool predicate(const FunctionDecl *FD) = 0;
  std::optional<SummaryAttribute> infer(const FunctionDecl *FD);

  virtual bool merge(const FunctionSummary &Summary) const = 0;

  virtual std::string serialize();
  virtual std::optional<SummaryAttribute> parse(std::string_view input);
};
} // namespace clang

#endif // LLVM_CLANG_SEMA_SEMASUMMARYATTRIBUTEH
