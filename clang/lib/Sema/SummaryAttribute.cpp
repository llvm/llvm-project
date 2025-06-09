#include "clang/Sema/SummaryAttribute.h"

namespace clang {
std::string SummaryAttributeDescription::serialize() { return std::string(Serialzed); }

std::optional<SummaryAttribute> SummaryAttributeDescription::parse(std::string_view input) {
  if(input == Serialzed)
    return Attr;

  return std::nullopt;
}

std::optional<SummaryAttribute> SummaryAttributeDescription::infer(const FunctionDecl *FD) {
  if (predicate(FD))
    return Attr;

  return std::nullopt;
}
} // namespace clang