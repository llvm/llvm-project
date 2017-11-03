//===- ASTMatchersInternal.cpp - Structural query framework ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Implements the base layer of the matcher framework.
//
//===----------------------------------------------------------------------===//

#include "clang/ASTMatchers/ASTMatchersInternal.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace clang {
namespace ast_matchers {
namespace internal {

bool NotUnaryOperator(const ast_type_traits::DynTypedNode &DynNode,
                      ASTMatchFinder *Finder, BoundNodesTreeBuilder *Builder,
                      ArrayRef<DynTypedMatcher> InnerMatchers);

bool AllOfVariadicOperator(const ast_type_traits::DynTypedNode &DynNode,
                           ASTMatchFinder *Finder,
                           BoundNodesTreeBuilder *Builder,
                           ArrayRef<DynTypedMatcher> InnerMatchers);

bool EachOfVariadicOperator(const ast_type_traits::DynTypedNode &DynNode,
                            ASTMatchFinder *Finder,
                            BoundNodesTreeBuilder *Builder,
                            ArrayRef<DynTypedMatcher> InnerMatchers);

bool AnyOfVariadicOperator(const ast_type_traits::DynTypedNode &DynNode,
                           ASTMatchFinder *Finder,
                           BoundNodesTreeBuilder *Builder,
                           ArrayRef<DynTypedMatcher> InnerMatchers);

void BoundNodesTreeBuilder::visitMatches(Visitor *ResultVisitor) {
  if (Bindings.empty())
    Bindings.push_back(BoundNodesMap());
  for (BoundNodesMap &Binding : Bindings) {
    ResultVisitor->visitMatch(BoundNodes(Binding));
  }
}

namespace {

using VariadicOperatorFunction = bool (*)(
    const ast_type_traits::DynTypedNode &DynNode, ASTMatchFinder *Finder,
    BoundNodesTreeBuilder *Builder, ArrayRef<DynTypedMatcher> InnerMatchers);

template <VariadicOperatorFunction Func>
class VariadicMatcher : public DynMatcherInterface {
public:
  VariadicMatcher(std::vector<DynTypedMatcher> InnerMatchers)
      : InnerMatchers(std::move(InnerMatchers)) {}

  bool dynMatches(const ast_type_traits::DynTypedNode &DynNode,
                  ASTMatchFinder *Finder,
                  BoundNodesTreeBuilder *Builder) const override {
    return Func(DynNode, Finder, Builder, InnerMatchers);
  }

private:
  std::vector<DynTypedMatcher> InnerMatchers;
};

class IdDynMatcher : public DynMatcherInterface {
public:
  IdDynMatcher(StringRef ID,
               IntrusiveRefCntPtr<DynMatcherInterface> InnerMatcher)
      : ID(ID), InnerMatcher(std::move(InnerMatcher)) {}

  bool dynMatches(const ast_type_traits::DynTypedNode &DynNode,
                  ASTMatchFinder *Finder,
                  BoundNodesTreeBuilder *Builder) const override {
    bool Result = InnerMatcher->dynMatches(DynNode, Finder, Builder);
    if (Result) Builder->setBinding(ID, DynNode);
    return Result;
  }

private:
  const std::string ID;
  const IntrusiveRefCntPtr<DynMatcherInterface> InnerMatcher;
};

/// \brief A matcher that always returns true.
///
/// We only ever need one instance of this matcher, so we create a global one
/// and reuse it to reduce the overhead of the matcher and increase the chance
/// of cache hits.
class TrueMatcherImpl : public DynMatcherInterface {
public:
  TrueMatcherImpl() {
    Retain(); // Reference count will never become zero.
  }

  bool dynMatches(const ast_type_traits::DynTypedNode &, ASTMatchFinder *,
                  BoundNodesTreeBuilder *) const override {
    return true;
  }
};

} // namespace

static llvm::ManagedStatic<TrueMatcherImpl> TrueMatcherInstance;

DynTypedMatcher DynTypedMatcher::constructVariadic(
    DynTypedMatcher::VariadicOperator Op,
    ast_type_traits::ASTNodeKind SupportedKind,
    std::vector<DynTypedMatcher> InnerMatchers) {
  assert(!InnerMatchers.empty() && "Array must not be empty.");
  assert(std::all_of(InnerMatchers.begin(), InnerMatchers.end(),
                     [SupportedKind](const DynTypedMatcher &M) {
                       return M.canConvertTo(SupportedKind);
                     }) &&
         "InnerMatchers must be convertible to SupportedKind!");

  // We must relax the restrict kind here.
  // The different operators might deal differently with a mismatch.
  // Make it the same as SupportedKind, since that is the broadest type we are
  // allowed to accept.
  auto RestrictKind = SupportedKind;

  switch (Op) {
  case VO_AllOf:
    // In the case of allOf() we must pass all the checks, so making
    // RestrictKind the most restrictive can save us time. This way we reject
    // invalid types earlier and we can elide the kind checks inside the
    // matcher.
    for (auto &IM : InnerMatchers) {
      RestrictKind = ast_type_traits::ASTNodeKind::getMostDerivedType(
          RestrictKind, IM.RestrictKind);
    }
    return DynTypedMatcher(
        SupportedKind, RestrictKind,
        new VariadicMatcher<AllOfVariadicOperator>(std::move(InnerMatchers)));

  case VO_AnyOf:
    return DynTypedMatcher(
        SupportedKind, RestrictKind,
        new VariadicMatcher<AnyOfVariadicOperator>(std::move(InnerMatchers)));

  case VO_EachOf:
    return DynTypedMatcher(
        SupportedKind, RestrictKind,
        new VariadicMatcher<EachOfVariadicOperator>(std::move(InnerMatchers)));

  case VO_UnaryNot:
    // FIXME: Implement the Not operator to take a single matcher instead of a
    // vector.
    return DynTypedMatcher(
        SupportedKind, RestrictKind,
        new VariadicMatcher<NotUnaryOperator>(std::move(InnerMatchers)));
  }
  llvm_unreachable("Invalid Op value.");
}

DynTypedMatcher DynTypedMatcher::trueMatcher(
    ast_type_traits::ASTNodeKind NodeKind) {
  return DynTypedMatcher(NodeKind, NodeKind, &*TrueMatcherInstance);
}

bool DynTypedMatcher::canMatchNodesOfKind(
    ast_type_traits::ASTNodeKind Kind) const {
  return RestrictKind.isBaseOf(Kind);
}

DynTypedMatcher DynTypedMatcher::dynCastTo(
    const ast_type_traits::ASTNodeKind Kind) const {
  auto Copy = *this;
  Copy.SupportedKind = Kind;
  Copy.RestrictKind =
      ast_type_traits::ASTNodeKind::getMostDerivedType(Kind, RestrictKind);
  return Copy;
}

bool DynTypedMatcher::matches(const ast_type_traits::DynTypedNode &DynNode,
                              ASTMatchFinder *Finder,
                              BoundNodesTreeBuilder *Builder) const {
  if (RestrictKind.isBaseOf(DynNode.getNodeKind()) &&
      Implementation->dynMatches(DynNode, Finder, Builder)) {
    return true;
  }
  // Delete all bindings when a matcher does not match.
  // This prevents unexpected exposure of bound nodes in unmatches
  // branches of the match tree.
  Builder->removeBindings([](const BoundNodesMap &) { return true; });
  return false;
}

bool DynTypedMatcher::matchesNoKindCheck(
    const ast_type_traits::DynTypedNode &DynNode, ASTMatchFinder *Finder,
    BoundNodesTreeBuilder *Builder) const {
  assert(RestrictKind.isBaseOf(DynNode.getNodeKind()));
  if (Implementation->dynMatches(DynNode, Finder, Builder)) {
    return true;
  }
  // Delete all bindings when a matcher does not match.
  // This prevents unexpected exposure of bound nodes in unmatches
  // branches of the match tree.
  Builder->removeBindings([](const BoundNodesMap &) { return true; });
  return false;
}

llvm::Optional<DynTypedMatcher> DynTypedMatcher::tryBind(StringRef ID) const {
  if (!AllowBind) return llvm::None;
  auto Result = *this;
  Result.Implementation =
      new IdDynMatcher(ID, std::move(Result.Implementation));
  return std::move(Result);
}

bool DynTypedMatcher::canConvertTo(ast_type_traits::ASTNodeKind To) const {
  const auto From = getSupportedKind();
  auto QualKind = ast_type_traits::ASTNodeKind::getFromNodeKind<QualType>();
  auto TypeKind = ast_type_traits::ASTNodeKind::getFromNodeKind<Type>();
  /// Mimic the implicit conversions of Matcher<>.
  /// - From Matcher<Type> to Matcher<QualType>
  if (From.isSame(TypeKind) && To.isSame(QualKind)) return true;
  /// - From Matcher<Base> to Matcher<Derived>
  return From.isBaseOf(To);
}

void BoundNodesTreeBuilder::addMatch(const BoundNodesTreeBuilder &Other) {
  Bindings.append(Other.Bindings.begin(), Other.Bindings.end());
}

bool NotUnaryOperator(const ast_type_traits::DynTypedNode &DynNode,
                      ASTMatchFinder *Finder, BoundNodesTreeBuilder *Builder,
                      ArrayRef<DynTypedMatcher> InnerMatchers) {
  if (InnerMatchers.size() != 1)
    return false;

  // The 'unless' matcher will always discard the result:
  // If the inner matcher doesn't match, unless returns true,
  // but the inner matcher cannot have bound anything.
  // If the inner matcher matches, the result is false, and
  // any possible binding will be discarded.
  // We still need to hand in all the bound nodes up to this
  // point so the inner matcher can depend on bound nodes,
  // and we need to actively discard the bound nodes, otherwise
  // the inner matcher will reset the bound nodes if it doesn't
  // match, but this would be inversed by 'unless'.
  BoundNodesTreeBuilder Discard(*Builder);
  return !InnerMatchers[0].matches(DynNode, Finder, &Discard);
}

bool AllOfVariadicOperator(const ast_type_traits::DynTypedNode &DynNode,
                           ASTMatchFinder *Finder,
                           BoundNodesTreeBuilder *Builder,
                           ArrayRef<DynTypedMatcher> InnerMatchers) {
  // allOf leads to one matcher for each alternative in the first
  // matcher combined with each alternative in the second matcher.
  // Thus, we can reuse the same Builder.
  for (const DynTypedMatcher &InnerMatcher : InnerMatchers) {
    if (!InnerMatcher.matchesNoKindCheck(DynNode, Finder, Builder))
      return false;
  }
  return true;
}

bool EachOfVariadicOperator(const ast_type_traits::DynTypedNode &DynNode,
                            ASTMatchFinder *Finder,
                            BoundNodesTreeBuilder *Builder,
                            ArrayRef<DynTypedMatcher> InnerMatchers) {
  BoundNodesTreeBuilder Result;
  bool Matched = false;
  for (const DynTypedMatcher &InnerMatcher : InnerMatchers) {
    BoundNodesTreeBuilder BuilderInner(*Builder);
    if (InnerMatcher.matches(DynNode, Finder, &BuilderInner)) {
      Matched = true;
      Result.addMatch(BuilderInner);
    }
  }
  *Builder = std::move(Result);
  return Matched;
}

bool AnyOfVariadicOperator(const ast_type_traits::DynTypedNode &DynNode,
                           ASTMatchFinder *Finder,
                           BoundNodesTreeBuilder *Builder,
                           ArrayRef<DynTypedMatcher> InnerMatchers) {
  for (const DynTypedMatcher &InnerMatcher : InnerMatchers) {
    BoundNodesTreeBuilder Result = *Builder;
    if (InnerMatcher.matches(DynNode, Finder, &Result)) {
      *Builder = std::move(Result);
      return true;
    }
  }
  return false;
}

Matcher<NamedDecl> hasAnyNameFunc(ArrayRef<const StringRef *> NameRefs) {
  std::vector<std::string> Names;
  for (auto *Name : NameRefs)
    Names.emplace_back(*Name);
  return internal::Matcher<NamedDecl>(
      new internal::HasNameMatcher(std::move(Names)));
}

HasNameMatcher::HasNameMatcher(std::vector<std::string> N)
    : UseUnqualifiedMatch(std::all_of(
          N.begin(), N.end(),
          [](StringRef Name) { return Name.find("::") == Name.npos; })),
      Names(std::move(N)) {
#ifndef NDEBUG
  for (StringRef Name : Names)
    assert(!Name.empty());
#endif
}

static bool consumeNameSuffix(StringRef &FullName, StringRef Suffix) {
  StringRef Name = FullName;
  if (!Name.endswith(Suffix))
    return false;
  Name = Name.drop_back(Suffix.size());
  if (!Name.empty()) {
    if (!Name.endswith("::"))
      return false;
    Name = Name.drop_back(2);
  }
  FullName = Name;
  return true;
}

static StringRef getNodeName(const NamedDecl &Node,
                             llvm::SmallString<128> &Scratch) {
  // Simple name.
  if (Node.getIdentifier())
    return Node.getName();

  if (Node.getDeclName()) {
    // Name needs to be constructed.
    Scratch.clear();
    llvm::raw_svector_ostream OS(Scratch);
    Node.printName(OS);
    return OS.str();
  }

  return "(anonymous)";
}

static StringRef getNodeName(const RecordDecl &Node,
                             llvm::SmallString<128> &Scratch) {
  if (Node.getIdentifier()) {
    return Node.getName();
  }
  Scratch.clear();
  return ("(anonymous " + Node.getKindName() + ")").toStringRef(Scratch);
}

static StringRef getNodeName(const NamespaceDecl &Node,
                             llvm::SmallString<128> &Scratch) {
  return Node.isAnonymousNamespace() ? "(anonymous namespace)" : Node.getName();
}

namespace {

class PatternSet {
public:
  PatternSet(ArrayRef<std::string> Names) {
    for (StringRef Name : Names)
      Patterns.push_back({Name, Name.startswith("::")});
  }

  /// Consumes the name suffix from each pattern in the set and removes the ones
  /// that didn't match.
  /// Return true if there are still any patterns left.
  bool consumeNameSuffix(StringRef NodeName, bool CanSkip) {
    for (size_t I = 0; I < Patterns.size();) {
      if (internal::consumeNameSuffix(Patterns[I].P, NodeName) ||
          CanSkip) {
        ++I;
      } else {
        Patterns.erase(Patterns.begin() + I);
      }
    }
    return !Patterns.empty();
  }

  /// Check if any of the patterns are a match.
  /// A match will be a pattern that was fully consumed, that also matches the
  /// 'fully qualified' requirement.
  bool foundMatch(bool AllowFullyQualified) const {
    for (auto& P: Patterns)
      if (P.P.empty() && (AllowFullyQualified || !P.IsFullyQualified))
        return true;
    return false;
  }

private:
  struct Pattern {
    StringRef P;
    bool IsFullyQualified;
  };

  llvm::SmallVector<Pattern, 8> Patterns;
};

} // namespace

bool HasNameMatcher::matchesNodeUnqualified(const NamedDecl &Node) const {
  assert(UseUnqualifiedMatch);
  llvm::SmallString<128> Scratch;
  StringRef NodeName = getNodeName(Node, Scratch);
  return std::any_of(Names.begin(), Names.end(), [&](StringRef Name) {
    return consumeNameSuffix(Name, NodeName) && Name.empty();
  });
}

bool HasNameMatcher::matchesNodeFullFast(const NamedDecl &Node) const {
  PatternSet Patterns(Names);
  llvm::SmallString<128> Scratch;

  // This function is copied and adapted from NamedDecl::printQualifiedName()
  // By matching each part individually we optimize in a couple of ways:
  //  - We can exit early on the first failure.
  //  - We can skip inline/anonymous namespaces without another pass.
  //  - We print one name at a time, reducing the chance of overflowing the
  //    inlined space of the SmallString.

  // First, match the name.
  if (!Patterns.consumeNameSuffix(getNodeName(Node, Scratch),
                                  /*CanSkip=*/false))
    return false;

  // Try to match each declaration context.
  // We are allowed to skip anonymous and inline namespaces if they don't match.
  const DeclContext *Ctx = Node.getDeclContext();

  if (Ctx->isFunctionOrMethod())
    return Patterns.foundMatch(/*AllowFullyQualified=*/false);

  for (; Ctx && isa<NamedDecl>(Ctx); Ctx = Ctx->getParent()) {
    if (Patterns.foundMatch(/*AllowFullyQualified=*/false))
      return true;

    if (const auto *ND = dyn_cast<NamespaceDecl>(Ctx)) {
      // If it matches (or we can skip it), continue.
      if (Patterns.consumeNameSuffix(getNodeName(*ND, Scratch),
                                     /*CanSkip=*/ND->isAnonymousNamespace() ||
                                         ND->isInline()))
        continue;
      return false;
    }
    if (const auto *RD = dyn_cast<RecordDecl>(Ctx)) {
      if (!isa<ClassTemplateSpecializationDecl>(Ctx)) {
        if (Patterns.consumeNameSuffix(getNodeName(*RD, Scratch),
                                       /*CanSkip=*/false))
          continue;

        return false;
      }
    }

    // We don't know how to deal with this DeclContext.
    // Fallback to the slow version of the code.
    return matchesNodeFullSlow(Node);
  }

  return Patterns.foundMatch(/*AllowFullyQualified=*/true);
}

bool HasNameMatcher::matchesNodeFullSlow(const NamedDecl &Node) const {
  const bool SkipUnwrittenCases[] = {false, true};
  for (bool SkipUnwritten : SkipUnwrittenCases) {
    llvm::SmallString<128> NodeName = StringRef("::");
    llvm::raw_svector_ostream OS(NodeName);

    if (SkipUnwritten) {
      PrintingPolicy Policy = Node.getASTContext().getPrintingPolicy();
      Policy.SuppressUnwrittenScope = true;
      Node.printQualifiedName(OS, Policy);
    } else {
      Node.printQualifiedName(OS);
    }

    const StringRef FullName = OS.str();

    for (const StringRef Pattern : Names) {
      if (Pattern.startswith("::")) {
        if (FullName == Pattern)
          return true;
      } else if (FullName.endswith(Pattern) &&
                 FullName.drop_back(Pattern.size()).endswith("::")) {
        return true;
      }
    }
  }

  return false;
}

bool HasNameMatcher::matchesNode(const NamedDecl &Node) const {
  assert(matchesNodeFullFast(Node) == matchesNodeFullSlow(Node));
  if (UseUnqualifiedMatch) {
    assert(matchesNodeUnqualified(Node) == matchesNodeFullFast(Node));
    return matchesNodeUnqualified(Node);
  }
  return matchesNodeFullFast(Node);
}

} // end namespace internal
} // end namespace ast_matchers
} // end namespace clang
