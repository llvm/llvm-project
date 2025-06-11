#ifndef LLVM_CLANG_SEMA_SEMASUMMARYATTRIBUTE_H
#define LLVM_CLANG_SEMA_SEMASUMMARYATTRIBUTE_H

#include "clang/AST/Decl.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include <string>

namespace clang {
enum SummaryAttributeKind {
  NO_WRITE_GLOBAL,
};

class FunctionSummary;

class SummaryAttribute {
  const SummaryAttributeKind Kind;
  std::string_view Serialzed;

public:
  SummaryAttribute(SummaryAttributeKind Attr, const char *Str)
      : Kind(Attr), Serialzed(Str) {}
  virtual ~SummaryAttribute() = default;

  SummaryAttributeKind getKind() { return Kind; }

  virtual bool infer(const FunctionDecl *FD) const = 0;
  virtual bool merge(const FunctionSummary &Summary) const = 0;

  virtual std::string serialize() const { return std::string(Serialzed); };
  virtual bool parse(std::string_view input) const {
    return input == Serialzed;
  };
};

class NoWriteGlobalAttr : public SummaryAttribute {
  class Callback : public ast_matchers::MatchFinder::MatchCallback {
  public:
    bool WriteGlobal = false;

    void
    run(const ast_matchers::MatchFinder::MatchResult &Result) override final;
  };

public:
  NoWriteGlobalAttr() : SummaryAttribute(NO_WRITE_GLOBAL, "no_write_global") {}

  bool infer(const FunctionDecl *FD) const override final;
  bool merge(const FunctionSummary &Summary) const override final;
};
} // namespace clang

#endif // LLVM_CLANG_SEMA_SEMASUMMARYATTRIBUTEH
