//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TodoCommentCheck.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Preprocessor.h"
#include <optional>

namespace clang::tidy {

namespace google::readability {

enum class StyleKind { Parentheses, Hyphen };

} // namespace google::readability

template <> struct OptionEnumMapping<google::readability::StyleKind> {
  static ArrayRef<std::pair<google::readability::StyleKind, StringRef>>
  getEnumMapping() {
    static constexpr std::pair<google::readability::StyleKind, StringRef>
        Mapping[] = {
            {google::readability::StyleKind::Hyphen, "Hyphen"},
            {google::readability::StyleKind::Parentheses, "Parentheses"},
        };
    return {Mapping};
  }
};

} // namespace clang::tidy

namespace clang::tidy::google::readability {
class TodoCommentCheck::TodoCommentHandler : public CommentHandler {
public:
  TodoCommentHandler(TodoCommentCheck &Check, std::optional<std::string> User)
      : Check(Check), User(User ? *User : "unknown"),
        TodoMatch(R"(^// *TODO *((\((.*)\))?:?( )?|: *(.*) *- *)?(.*)$)") {
    const llvm::StringRef TodoStyleString =
        Check.Options.get("Style", "Hyphen");
    for (const auto &[Value, Name] :
         OptionEnumMapping<StyleKind>::getEnumMapping()) {
      if (Name == TodoStyleString) {
        TodoStyle = Value;
        return;
      }
    }
    Check.configurationDiag(
        "invalid value '%0' for "
        "google-readability-todo.Style; valid values are "
        "'Parentheses' and 'Hyphen'. Defaulting to 'Hyphen'")
        << TodoStyleString;
  }

  bool HandleComment(Preprocessor &PP, SourceRange Range) override {
    const StringRef Text =
        Lexer::getSourceText(CharSourceRange::getCharRange(Range),
                             PP.getSourceManager(), PP.getLangOpts());

    SmallVector<StringRef, 7> Matches;
    if (!TodoMatch.match(Text, &Matches))
      return false;

    const StyleKind ParsedStyle =
        !Matches[3].empty() ? StyleKind::Parentheses : StyleKind::Hyphen;
    const StringRef Username =
        ParsedStyle == StyleKind::Parentheses ? Matches[3] : Matches[5];
    const StringRef Comment = Matches[6];

    if (!Username.empty() &&
        (ParsedStyle == StyleKind::Parentheses || !Comment.empty())) {
      return false;
    }

    if (Username.empty()) {
      Check.diag(Range.getBegin(), "missing username/bug in TODO")
          << FixItHint::CreateReplacement(
                 CharSourceRange::getCharRange(Range),
                 createReplacementString(Username, Comment));
    }

    if (Comment.empty())
      Check.diag(Range.getBegin(), "missing details in TODO");

    return false;
  }

  std::string createReplacementString(const StringRef Username,
                                      const StringRef Comment) const {
    if (TodoStyle == StyleKind::Parentheses) {
      return ("// TODO(" + Twine(User) +
              "): " + (Comment.empty() ? "some details" : Comment))
          .str();
    }
    return ("// TODO: " + Twine(User) + " - " +
            (Comment.empty() ? "some details" : Comment))
        .str();
  }

  StyleKind getTodoStyle() const { return TodoStyle; }

private:
  TodoCommentCheck &Check;
  std::string User;
  llvm::Regex TodoMatch;
  StyleKind TodoStyle = StyleKind::Hyphen;
};

TodoCommentCheck::TodoCommentCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      Handler(std::make_unique<TodoCommentHandler>(
          *this, Context->getOptions().User)) {}

TodoCommentCheck::~TodoCommentCheck() = default;

void TodoCommentCheck::registerPPCallbacks(const SourceManager &SM,
                                           Preprocessor *PP,
                                           Preprocessor *ModuleExpanderPP) {
  PP->addCommentHandler(Handler.get());
}

void TodoCommentCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "Style", Handler->getTodoStyle());
}

} // namespace clang::tidy::google::readability
