#ifndef LLVM_CLANG_SUMMARY_SUMMARYATTRIBUTE_H
#define LLVM_CLANG_SUMMARY_SUMMARYATTRIBUTE_H

#include "clang/AST/Decl.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

namespace clang {
enum SummaryAttrKind { NO_WRITE_GLOBAL, NO_WRITE_PTR_PARAMETER };

class FunctionSummary;
class SummaryContext;

class SummaryAttr {
  const SummaryAttrKind Kind;
  const char *Spelling;

protected:
  SummaryAttr(SummaryAttrKind Kind, const char *Spelling)
      : Kind(Kind), Spelling(Spelling) {};

public:
  virtual ~SummaryAttr() = default;

  SummaryAttrKind getKind() const { return Kind; }
  const char *getSpelling() const { return Spelling; }

  virtual bool infer(const FunctionDecl *FD) const = 0;
  virtual bool merge(const FunctionSummary &Caller,
                     const FunctionSummary *Callee) const = 0;

  virtual std::string serialize() const { return std::string(Spelling); };
  virtual bool parse(std::string_view input) const {
    return input == Spelling;
  };
};

class NoWriteGlobalAttr : public SummaryAttr {
  NoWriteGlobalAttr() : SummaryAttr(NO_WRITE_GLOBAL, "no_write_global") {}

public:
  bool infer(const FunctionDecl *FD) const override final;
  bool merge(const FunctionSummary &Caller,
             const FunctionSummary *Callee) const override final;

  static bool classof(const SummaryAttr *A) {
    return A->getKind() == NO_WRITE_GLOBAL;
  }
  friend class SummaryContext;
};
} // namespace clang

#endif // LLVM_CLANG_SUMMARY_SUMMARYATTRIBUTEH
