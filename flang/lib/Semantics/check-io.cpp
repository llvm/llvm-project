//===-- lib/Semantics/check-io.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "check-io.h"
#include "flang/Common/format.h"
#include "flang/Parser/tools.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/tools.h"
#include <unordered_map>

namespace Fortran::semantics {

// TODO: C1234, C1235 -- defined I/O constraints

class FormatErrorReporter {
public:
  FormatErrorReporter(SemanticsContext &context,
      const parser::CharBlock &formatCharBlock, int errorAllowance = 3)
      : context_{context}, formatCharBlock_{formatCharBlock},
        errorAllowance_{errorAllowance} {}

  bool Say(const common::FormatMessage &);

private:
  SemanticsContext &context_;
  const parser::CharBlock &formatCharBlock_;
  int errorAllowance_; // initialized to maximum number of errors to report
};

bool FormatErrorReporter::Say(const common::FormatMessage &msg) {
  if (!msg.isError && !context_.warnOnNonstandardUsage()) {
    return false;
  }
  parser::MessageFormattedText text{
      parser::MessageFixedText(msg.text, strlen(msg.text), msg.isError),
      msg.arg};
  if (formatCharBlock_.size()) {
    // The input format is a folded expression.  Error markers span the full
    // original unfolded expression in formatCharBlock_.
    context_.Say(formatCharBlock_, text);
  } else {
    // The input format is a source expression.  Error markers have an offset
    // and length relative to the beginning of formatCharBlock_.
    parser::CharBlock messageCharBlock{
        parser::CharBlock(formatCharBlock_.begin() + msg.offset, msg.length)};
    context_.Say(messageCharBlock, text);
  }
  return msg.isError && --errorAllowance_ <= 0;
}

void IoChecker::Enter(
    const parser::Statement<common::Indirection<parser::FormatStmt>> &stmt) {
  if (!stmt.label) {
    context_.Say("Format statement must be labeled"_err_en_US); // C1301
  }
  const char *formatStart{static_cast<const char *>(
      std::memchr(stmt.source.begin(), '(', stmt.source.size()))};
  parser::CharBlock reporterCharBlock{formatStart, static_cast<std::size_t>(0)};
  FormatErrorReporter reporter{context_, reporterCharBlock};
  auto reporterWrapper{[&](const auto &msg) { return reporter.Say(msg); }};
  switch (context_.GetDefaultKind(TypeCategory::Character)) {
  case 1: {
    common::FormatValidator<char> validator{formatStart,
        stmt.source.size() - (formatStart - stmt.source.begin()),
        reporterWrapper};
    validator.Check();
    break;
  }
  case 2: { // TODO: Get this to work.
    common::FormatValidator<char16_t> validator{
        /*???*/ nullptr, /*???*/ 0, reporterWrapper};
    validator.Check();
    break;
  }
  case 4: { // TODO: Get this to work.
    common::FormatValidator<char32_t> validator{
        /*???*/ nullptr, /*???*/ 0, reporterWrapper};
    validator.Check();
    break;
  }
  default:
    CRASH_NO_CASE;
  }
}

void IoChecker::Enter(const parser::ConnectSpec &spec) {
  // ConnectSpec context FileNameExpr
  if (std::get_if<parser::FileNameExpr>(&spec.u)) {
    SetSpecifier(IoSpecKind::File);
  }
}

void IoChecker::Enter(const parser::ConnectSpec::CharExpr &spec) {
  IoSpecKind specKind{};
  using ParseKind = parser::ConnectSpec::CharExpr::Kind;
  switch (std::get<ParseKind>(spec.t)) {
  case ParseKind::Access:
    specKind = IoSpecKind::Access;
    break;
  case ParseKind::Action:
    specKind = IoSpecKind::Action;
    break;
  case ParseKind::Asynchronous:
    specKind = IoSpecKind::Asynchronous;
    break;
  case ParseKind::Blank:
    specKind = IoSpecKind::Blank;
    break;
  case ParseKind::Decimal:
    specKind = IoSpecKind::Decimal;
    break;
  case ParseKind::Delim:
    specKind = IoSpecKind::Delim;
    break;
  case ParseKind::Encoding:
    specKind = IoSpecKind::Encoding;
    break;
  case ParseKind::Form:
    specKind = IoSpecKind::Form;
    break;
  case ParseKind::Pad:
    specKind = IoSpecKind::Pad;
    break;
  case ParseKind::Position:
    specKind = IoSpecKind::Position;
    break;
  case ParseKind::Round:
    specKind = IoSpecKind::Round;
    break;
  case ParseKind::Sign:
    specKind = IoSpecKind::Sign;
    break;
  case ParseKind::Carriagecontrol:
    specKind = IoSpecKind::Carriagecontrol;
    break;
  case ParseKind::Convert:
    specKind = IoSpecKind::Convert;
    break;
  case ParseKind::Dispose:
    specKind = IoSpecKind::Dispose;
    break;
  }
  SetSpecifier(specKind);
  if (const std::optional<std::string> charConst{GetConstExpr<std::string>(
          std::get<parser::ScalarDefaultCharExpr>(spec.t))}) {
    std::string s{parser::ToUpperCaseLetters(*charConst)};
    if (specKind == IoSpecKind::Access) {
      flags_.set(Flag::KnownAccess);
      flags_.set(Flag::AccessDirect, s == "DIRECT");
      flags_.set(Flag::AccessStream, s == "STREAM");
    }
    CheckStringValue(specKind, *charConst, parser::FindSourceLocation(spec));
    if (specKind == IoSpecKind::Carriagecontrol &&
        (s == "FORTRAN" || s == "NONE")) {
      context_.Say(parser::FindSourceLocation(spec),
          "Unimplemented %s value '%s'"_err_en_US,
          parser::ToUpperCaseLetters(common::EnumToString(specKind)),
          *charConst);
    }
  }
}

void IoChecker::Enter(const parser::ConnectSpec::Newunit &var) {
  CheckForDefinableVariable(var, "NEWUNIT");
  SetSpecifier(IoSpecKind::Newunit);
}

void IoChecker::Enter(const parser::ConnectSpec::Recl &spec) {
  SetSpecifier(IoSpecKind::Recl);
  if (const std::optional<std::int64_t> recl{
          GetConstExpr<std::int64_t>(spec)}) {
    if (*recl <= 0) {
      context_.Say(parser::FindSourceLocation(spec),
          "RECL value (%jd) must be positive"_err_en_US,
          *recl); // 12.5.6.15
    }
  }
}

void IoChecker::Enter(const parser::EndLabel &) {
  SetSpecifier(IoSpecKind::End);
}

void IoChecker::Enter(const parser::EorLabel &) {
  SetSpecifier(IoSpecKind::Eor);
}

void IoChecker::Enter(const parser::ErrLabel &) {
  SetSpecifier(IoSpecKind::Err);
}

void IoChecker::Enter(const parser::FileUnitNumber &) {
  SetSpecifier(IoSpecKind::Unit);
  flags_.set(Flag::NumberUnit);
}

void IoChecker::Enter(const parser::Format &spec) {
  SetSpecifier(IoSpecKind::Fmt);
  flags_.set(Flag::FmtOrNml);
  std::visit(
      common::visitors{
          [&](const parser::Label &) { flags_.set(Flag::LabelFmt); },
          [&](const parser::Star &) { flags_.set(Flag::StarFmt); },
          [&](const parser::Expr &format) {
            const SomeExpr *expr{GetExpr(format)};
            if (!expr) {
              return;
            }
            auto type{expr->GetType()};
            if (!type ||
                (type->category() != TypeCategory::Integer &&
                    type->category() != TypeCategory::Character) ||
                type->kind() !=
                    context_.defaultKinds().GetDefaultKind(type->category())) {
              context_.Say(format.source,
                  "Format expression must be default character or integer"_err_en_US);
              return;
            }
            if (type->category() == TypeCategory::Integer) {
              flags_.set(Flag::AssignFmt);
              if (expr->Rank() != 0 || !IsVariable(*expr)) {
                context_.Say(format.source,
                    "Assigned format label must be a scalar variable"_err_en_US);
              }
              return;
            }
            flags_.set(Flag::CharFmt);
            const std::optional<std::string> constantFormat{
                GetConstExpr<std::string>(format)};
            if (!constantFormat) {
              return;
            }
            // validate constant format -- 12.6.2.2
            bool isFolded{constantFormat->size() != format.source.size() - 2};
            parser::CharBlock reporterCharBlock{isFolded
                    ? parser::CharBlock{format.source}
                    : parser::CharBlock{format.source.begin() + 1,
                          static_cast<std::size_t>(0)}};
            FormatErrorReporter reporter{context_, reporterCharBlock};
            auto reporterWrapper{
                [&](const auto &msg) { return reporter.Say(msg); }};
            switch (context_.GetDefaultKind(TypeCategory::Character)) {
            case 1: {
              common::FormatValidator<char> validator{constantFormat->c_str(),
                  constantFormat->length(), reporterWrapper, stmt_};
              validator.Check();
              break;
            }
            case 2: {
              // TODO: Get this to work.  (Maybe combine with earlier instance?)
              common::FormatValidator<char16_t> validator{
                  /*???*/ nullptr, /*???*/ 0, reporterWrapper, stmt_};
              validator.Check();
              break;
            }
            case 4: {
              // TODO: Get this to work.  (Maybe combine with earlier instance?)
              common::FormatValidator<char32_t> validator{
                  /*???*/ nullptr, /*???*/ 0, reporterWrapper, stmt_};
              validator.Check();
              break;
            }
            default:
              CRASH_NO_CASE;
            }
          },
      },
      spec.u);
}

void IoChecker::Enter(const parser::IdExpr &) { SetSpecifier(IoSpecKind::Id); }

void IoChecker::Enter(const parser::IdVariable &spec) {
  SetSpecifier(IoSpecKind::Id);
  const auto *expr{GetExpr(spec)};
  if (!expr || !expr->GetType()) {
    return;
  }
  CheckForDefinableVariable(spec, "ID");
  int kind{expr->GetType()->kind()};
  int defaultKind{context_.GetDefaultKind(TypeCategory::Integer)};
  if (kind < defaultKind) {
    context_.Say(
        "ID kind (%d) is smaller than default INTEGER kind (%d)"_err_en_US,
        std::move(kind), std::move(defaultKind)); // C1229
  }
}

void IoChecker::Enter(const parser::InputItem &spec) {
  flags_.set(Flag::DataList);
  const parser::Variable *var{std::get_if<parser::Variable>(&spec.u)};
  if (!var) {
    return;
  }
  CheckForDefinableVariable(*var, "Input");
}

void IoChecker::Enter(const parser::InquireSpec &spec) {
  // InquireSpec context FileNameExpr
  if (std::get_if<parser::FileNameExpr>(&spec.u)) {
    SetSpecifier(IoSpecKind::File);
  }
}

void IoChecker::Enter(const parser::InquireSpec::CharVar &spec) {
  IoSpecKind specKind{};
  using ParseKind = parser::InquireSpec::CharVar::Kind;
  switch (std::get<ParseKind>(spec.t)) {
  case ParseKind::Access:
    specKind = IoSpecKind::Access;
    break;
  case ParseKind::Action:
    specKind = IoSpecKind::Action;
    break;
  case ParseKind::Asynchronous:
    specKind = IoSpecKind::Asynchronous;
    break;
  case ParseKind::Blank:
    specKind = IoSpecKind::Blank;
    break;
  case ParseKind::Decimal:
    specKind = IoSpecKind::Decimal;
    break;
  case ParseKind::Delim:
    specKind = IoSpecKind::Delim;
    break;
  case ParseKind::Direct:
    specKind = IoSpecKind::Direct;
    break;
  case ParseKind::Encoding:
    specKind = IoSpecKind::Encoding;
    break;
  case ParseKind::Form:
    specKind = IoSpecKind::Form;
    break;
  case ParseKind::Formatted:
    specKind = IoSpecKind::Formatted;
    break;
  case ParseKind::Iomsg:
    specKind = IoSpecKind::Iomsg;
    break;
  case ParseKind::Name:
    specKind = IoSpecKind::Name;
    break;
  case ParseKind::Pad:
    specKind = IoSpecKind::Pad;
    break;
  case ParseKind::Position:
    specKind = IoSpecKind::Position;
    break;
  case ParseKind::Read:
    specKind = IoSpecKind::Read;
    break;
  case ParseKind::Readwrite:
    specKind = IoSpecKind::Readwrite;
    break;
  case ParseKind::Round:
    specKind = IoSpecKind::Round;
    break;
  case ParseKind::Sequential:
    specKind = IoSpecKind::Sequential;
    break;
  case ParseKind::Sign:
    specKind = IoSpecKind::Sign;
    break;
  case ParseKind::Status:
    specKind = IoSpecKind::Status;
    break;
  case ParseKind::Stream:
    specKind = IoSpecKind::Stream;
    break;
  case ParseKind::Unformatted:
    specKind = IoSpecKind::Unformatted;
    break;
  case ParseKind::Write:
    specKind = IoSpecKind::Write;
    break;
  case ParseKind::Carriagecontrol:
    specKind = IoSpecKind::Carriagecontrol;
    break;
  case ParseKind::Convert:
    specKind = IoSpecKind::Convert;
    break;
  case ParseKind::Dispose:
    specKind = IoSpecKind::Dispose;
    break;
  }
  CheckForDefinableVariable(std::get<parser::ScalarDefaultCharVariable>(spec.t),
      parser::ToUpperCaseLetters(common::EnumToString(specKind)));
  SetSpecifier(specKind);
}

void IoChecker::Enter(const parser::InquireSpec::IntVar &spec) {
  IoSpecKind specKind{};
  using ParseKind = parser::InquireSpec::IntVar::Kind;
  switch (std::get<parser::InquireSpec::IntVar::Kind>(spec.t)) {
  case ParseKind::Iostat:
    specKind = IoSpecKind::Iostat;
    break;
  case ParseKind::Nextrec:
    specKind = IoSpecKind::Nextrec;
    break;
  case ParseKind::Number:
    specKind = IoSpecKind::Number;
    break;
  case ParseKind::Pos:
    specKind = IoSpecKind::Pos;
    break;
  case ParseKind::Recl:
    specKind = IoSpecKind::Recl;
    break;
  case ParseKind::Size:
    specKind = IoSpecKind::Size;
    break;
  }
  CheckForDefinableVariable(std::get<parser::ScalarIntVariable>(spec.t),
      parser::ToUpperCaseLetters(common::EnumToString(specKind)));
  SetSpecifier(specKind);
}

void IoChecker::Enter(const parser::InquireSpec::LogVar &spec) {
  IoSpecKind specKind{};
  using ParseKind = parser::InquireSpec::LogVar::Kind;
  switch (std::get<parser::InquireSpec::LogVar::Kind>(spec.t)) {
  case ParseKind::Exist:
    specKind = IoSpecKind::Exist;
    break;
  case ParseKind::Named:
    specKind = IoSpecKind::Named;
    break;
  case ParseKind::Opened:
    specKind = IoSpecKind::Opened;
    break;
  case ParseKind::Pending:
    specKind = IoSpecKind::Pending;
    break;
  }
  SetSpecifier(specKind);
}

void IoChecker::Enter(const parser::IoControlSpec &spec) {
  // IoControlSpec context Name
  flags_.set(Flag::IoControlList);
  if (std::holds_alternative<parser::Name>(spec.u)) {
    SetSpecifier(IoSpecKind::Nml);
    flags_.set(Flag::FmtOrNml);
  }
}

void IoChecker::Enter(const parser::IoControlSpec::Asynchronous &spec) {
  SetSpecifier(IoSpecKind::Asynchronous);
  if (const std::optional<std::string> charConst{
          GetConstExpr<std::string>(spec)}) {
    flags_.set(
        Flag::AsynchronousYes, parser::ToUpperCaseLetters(*charConst) == "YES");
    CheckStringValue(IoSpecKind::Asynchronous, *charConst,
        parser::FindSourceLocation(spec)); // C1223
  }
}

void IoChecker::Enter(const parser::IoControlSpec::CharExpr &spec) {
  IoSpecKind specKind{};
  using ParseKind = parser::IoControlSpec::CharExpr::Kind;
  switch (std::get<ParseKind>(spec.t)) {
  case ParseKind::Advance:
    specKind = IoSpecKind::Advance;
    break;
  case ParseKind::Blank:
    specKind = IoSpecKind::Blank;
    break;
  case ParseKind::Decimal:
    specKind = IoSpecKind::Decimal;
    break;
  case ParseKind::Delim:
    specKind = IoSpecKind::Delim;
    break;
  case ParseKind::Pad:
    specKind = IoSpecKind::Pad;
    break;
  case ParseKind::Round:
    specKind = IoSpecKind::Round;
    break;
  case ParseKind::Sign:
    specKind = IoSpecKind::Sign;
    break;
  }
  SetSpecifier(specKind);
  if (const std::optional<std::string> charConst{GetConstExpr<std::string>(
          std::get<parser::ScalarDefaultCharExpr>(spec.t))}) {
    if (specKind == IoSpecKind::Advance) {
      flags_.set(
          Flag::AdvanceYes, parser::ToUpperCaseLetters(*charConst) == "YES");
    }
    CheckStringValue(specKind, *charConst, parser::FindSourceLocation(spec));
  }
}

void IoChecker::Enter(const parser::IoControlSpec::Pos &) {
  SetSpecifier(IoSpecKind::Pos);
}

void IoChecker::Enter(const parser::IoControlSpec::Rec &) {
  SetSpecifier(IoSpecKind::Rec);
}

void IoChecker::Enter(const parser::IoControlSpec::Size &var) {
  CheckForDefinableVariable(var, "SIZE");
  SetSpecifier(IoSpecKind::Size);
}

void IoChecker::Enter(const parser::IoUnit &spec) {
  if (const parser::Variable * var{std::get_if<parser::Variable>(&spec.u)}) {
    if (stmt_ == IoStmtKind::Write) {
      CheckForDefinableVariable(*var, "Internal file");
    }
    if (const auto *expr{GetExpr(*var)}) {
      if (HasVectorSubscript(*expr)) {
        context_.Say(parser::FindSourceLocation(*var), // C1201
            "Internal file must not have a vector subscript"_err_en_US);
      } else if (!ExprTypeKindIsDefault(*expr, context_)) {
        // This may be too restrictive; other kinds may be valid.
        context_.Say(parser::FindSourceLocation(*var), // C1202
            "Invalid character kind for an internal file variable"_err_en_US);
      }
    }
    SetSpecifier(IoSpecKind::Unit);
    flags_.set(Flag::InternalUnit);
  } else if (std::get_if<parser::Star>(&spec.u)) {
    SetSpecifier(IoSpecKind::Unit);
    flags_.set(Flag::StarUnit);
  }
}

void IoChecker::Enter(const parser::MsgVariable &var) {
  if (stmt_ == IoStmtKind::None) {
    // allocate, deallocate, image control
    CheckForDefinableVariable(var, "ERRMSG");
    return;
  }
  CheckForDefinableVariable(var, "IOMSG");
  SetSpecifier(IoSpecKind::Iomsg);
}

void IoChecker::Enter(const parser::OutputItem &item) {
  flags_.set(Flag::DataList);
  if (const auto *x{std::get_if<parser::Expr>(&item.u)}) {
    if (const auto *expr{GetExpr(*x)}) {
      const Symbol *last{GetLastSymbol(*expr)};
      if (last && IsProcedurePointer(*last)) {
        context_.Say(parser::FindSourceLocation(*x),
            "Output item must not be a procedure pointer"_err_en_US); // C1233
      }
    }
  }
}

void IoChecker::Enter(const parser::StatusExpr &spec) {
  SetSpecifier(IoSpecKind::Status);
  if (const std::optional<std::string> charConst{
          GetConstExpr<std::string>(spec)}) {
    // Status values for Open and Close are different.
    std::string s{parser::ToUpperCaseLetters(*charConst)};
    if (stmt_ == IoStmtKind::Open) {
      flags_.set(Flag::KnownStatus);
      flags_.set(Flag::StatusNew, s == "NEW");
      flags_.set(Flag::StatusReplace, s == "REPLACE");
      flags_.set(Flag::StatusScratch, s == "SCRATCH");
      // CheckStringValue compares for OPEN Status string values.
      CheckStringValue(
          IoSpecKind::Status, *charConst, parser::FindSourceLocation(spec));
      return;
    }
    CHECK(stmt_ == IoStmtKind::Close);
    if (s != "DELETE" && s != "KEEP") {
      context_.Say(parser::FindSourceLocation(spec),
          "Invalid STATUS value '%s'"_err_en_US, *charConst);
    }
  }
}

void IoChecker::Enter(const parser::StatVariable &var) {
  if (stmt_ == IoStmtKind::None) {
    // allocate, deallocate, image control
    CheckForDefinableVariable(var, "STAT");
    return;
  }
  CheckForDefinableVariable(var, "IOSTAT");
  SetSpecifier(IoSpecKind::Iostat);
}

void IoChecker::Leave(const parser::BackspaceStmt &) {
  CheckForPureSubprogram();
  CheckForRequiredSpecifier(
      flags_.test(Flag::NumberUnit), "UNIT number"); // C1240
  Done();
}

void IoChecker::Leave(const parser::CloseStmt &) {
  CheckForPureSubprogram();
  CheckForRequiredSpecifier(
      flags_.test(Flag::NumberUnit), "UNIT number"); // C1208
  Done();
}

void IoChecker::Leave(const parser::EndfileStmt &) {
  CheckForPureSubprogram();
  CheckForRequiredSpecifier(
      flags_.test(Flag::NumberUnit), "UNIT number"); // C1240
  Done();
}

void IoChecker::Leave(const parser::FlushStmt &) {
  CheckForPureSubprogram();
  CheckForRequiredSpecifier(
      flags_.test(Flag::NumberUnit), "UNIT number"); // C1243
  Done();
}

void IoChecker::Leave(const parser::InquireStmt &stmt) {
  if (std::get_if<std::list<parser::InquireSpec>>(&stmt.u)) {
    CheckForPureSubprogram();
    // Inquire by unit or by file (vs. by output list).
    CheckForRequiredSpecifier(
        flags_.test(Flag::NumberUnit) || specifierSet_.test(IoSpecKind::File),
        "UNIT number or FILE"); // C1246
    CheckForProhibitedSpecifier(IoSpecKind::File, IoSpecKind::Unit); // C1246
    CheckForRequiredSpecifier(IoSpecKind::Id, IoSpecKind::Pending); // C1248
  }
  Done();
}

void IoChecker::Leave(const parser::OpenStmt &) {
  CheckForPureSubprogram();
  CheckForRequiredSpecifier(specifierSet_.test(IoSpecKind::Unit) ||
          specifierSet_.test(IoSpecKind::Newunit),
      "UNIT or NEWUNIT"); // C1204, C1205
  CheckForProhibitedSpecifier(
      IoSpecKind::Newunit, IoSpecKind::Unit); // C1204, C1205
  CheckForRequiredSpecifier(flags_.test(Flag::StatusNew), "STATUS='NEW'",
      IoSpecKind::File); // 12.5.6.10
  CheckForRequiredSpecifier(flags_.test(Flag::StatusReplace),
      "STATUS='REPLACE'", IoSpecKind::File); // 12.5.6.10
  CheckForProhibitedSpecifier(flags_.test(Flag::StatusScratch),
      "STATUS='SCRATCH'", IoSpecKind::File); // 12.5.6.10
  if (flags_.test(Flag::KnownStatus)) {
    CheckForRequiredSpecifier(IoSpecKind::Newunit,
        specifierSet_.test(IoSpecKind::File) ||
            flags_.test(Flag::StatusScratch),
        "FILE or STATUS='SCRATCH'"); // 12.5.6.12
  } else {
    CheckForRequiredSpecifier(IoSpecKind::Newunit,
        specifierSet_.test(IoSpecKind::File) ||
            specifierSet_.test(IoSpecKind::Status),
        "FILE or STATUS"); // 12.5.6.12
  }
  if (flags_.test(Flag::KnownAccess)) {
    CheckForRequiredSpecifier(flags_.test(Flag::AccessDirect),
        "ACCESS='DIRECT'", IoSpecKind::Recl); // 12.5.6.15
    CheckForProhibitedSpecifier(flags_.test(Flag::AccessStream),
        "STATUS='STREAM'", IoSpecKind::Recl); // 12.5.6.15
  }
  Done();
}

void IoChecker::Leave(const parser::PrintStmt &) {
  CheckForPureSubprogram();
  Done();
}

static void CheckForDoVariableInNamelist(const Symbol &namelist,
    SemanticsContext &context, parser::CharBlock namelistLocation) {
  const auto &details{namelist.GetUltimate().get<NamelistDetails>()};
  for (const Symbol &object : details.objects()) {
    context.CheckIndexVarRedefine(namelistLocation, object);
  }
}

static void CheckForDoVariableInNamelistSpec(
    const parser::ReadStmt &readStmt, SemanticsContext &context) {
  const std::list<parser::IoControlSpec> &controls{readStmt.controls};
  for (const auto &control : controls) {
    if (const auto *namelist{std::get_if<parser::Name>(&control.u)}) {
      if (const Symbol * symbol{namelist->symbol}) {
        CheckForDoVariableInNamelist(*symbol, context, namelist->source);
      }
    }
  }
}

static void CheckForDoVariable(
    const parser::ReadStmt &readStmt, SemanticsContext &context) {
  CheckForDoVariableInNamelistSpec(readStmt, context);
  const std::list<parser::InputItem> &items{readStmt.items};
  for (const auto &item : items) {
    if (const parser::Variable *
        variable{std::get_if<parser::Variable>(&item.u)}) {
      context.CheckIndexVarRedefine(*variable);
    }
  }
}

void IoChecker::Leave(const parser::ReadStmt &readStmt) {
  if (!flags_.test(Flag::InternalUnit)) {
    CheckForPureSubprogram();
  }
  CheckForDoVariable(readStmt, context_);
  if (!flags_.test(Flag::IoControlList)) {
    Done();
    return;
  }
  LeaveReadWrite();
  CheckForProhibitedSpecifier(IoSpecKind::Delim); // C1212
  CheckForProhibitedSpecifier(IoSpecKind::Sign); // C1212
  CheckForProhibitedSpecifier(IoSpecKind::Rec, IoSpecKind::End); // C1220
  CheckForRequiredSpecifier(IoSpecKind::Eor,
      specifierSet_.test(IoSpecKind::Advance) && !flags_.test(Flag::AdvanceYes),
      "ADVANCE with value 'NO'"); // C1222 + 12.6.2.1p2
  CheckForRequiredSpecifier(IoSpecKind::Blank, flags_.test(Flag::FmtOrNml),
      "FMT or NML"); // C1227
  CheckForRequiredSpecifier(
      IoSpecKind::Pad, flags_.test(Flag::FmtOrNml), "FMT or NML"); // C1227
  Done();
}

void IoChecker::Leave(const parser::RewindStmt &) {
  CheckForRequiredSpecifier(
      flags_.test(Flag::NumberUnit), "UNIT number"); // C1240
  CheckForPureSubprogram();
  Done();
}

void IoChecker::Leave(const parser::WaitStmt &) {
  CheckForRequiredSpecifier(
      flags_.test(Flag::NumberUnit), "UNIT number"); // C1237
  CheckForPureSubprogram();
  Done();
}

void IoChecker::Leave(const parser::WriteStmt &) {
  if (!flags_.test(Flag::InternalUnit)) {
    CheckForPureSubprogram();
  }
  LeaveReadWrite();
  CheckForProhibitedSpecifier(IoSpecKind::Blank); // C1213
  CheckForProhibitedSpecifier(IoSpecKind::End); // C1213
  CheckForProhibitedSpecifier(IoSpecKind::Eor); // C1213
  CheckForProhibitedSpecifier(IoSpecKind::Pad); // C1213
  CheckForProhibitedSpecifier(IoSpecKind::Size); // C1213
  CheckForRequiredSpecifier(
      IoSpecKind::Sign, flags_.test(Flag::FmtOrNml), "FMT or NML"); // C1227
  CheckForRequiredSpecifier(IoSpecKind::Delim,
      flags_.test(Flag::StarFmt) || specifierSet_.test(IoSpecKind::Nml),
      "FMT=* or NML"); // C1228
  Done();
}

void IoChecker::LeaveReadWrite() const {
  CheckForRequiredSpecifier(IoSpecKind::Unit); // C1211
  CheckForProhibitedSpecifier(IoSpecKind::Nml, IoSpecKind::Rec); // C1216
  CheckForProhibitedSpecifier(IoSpecKind::Nml, IoSpecKind::Fmt); // C1216
  CheckForProhibitedSpecifier(
      IoSpecKind::Nml, flags_.test(Flag::DataList), "a data list"); // C1216
  CheckForProhibitedSpecifier(flags_.test(Flag::InternalUnit),
      "UNIT=internal-file", IoSpecKind::Pos); // C1219
  CheckForProhibitedSpecifier(flags_.test(Flag::InternalUnit),
      "UNIT=internal-file", IoSpecKind::Rec); // C1219
  CheckForProhibitedSpecifier(
      flags_.test(Flag::StarUnit), "UNIT=*", IoSpecKind::Pos); // C1219
  CheckForProhibitedSpecifier(
      flags_.test(Flag::StarUnit), "UNIT=*", IoSpecKind::Rec); // C1219
  CheckForProhibitedSpecifier(
      IoSpecKind::Rec, flags_.test(Flag::StarFmt), "FMT=*"); // C1220
  CheckForRequiredSpecifier(IoSpecKind::Advance,
      flags_.test(Flag::CharFmt) || flags_.test(Flag::LabelFmt) ||
          flags_.test(Flag::AssignFmt),
      "an explicit format"); // C1221
  CheckForProhibitedSpecifier(IoSpecKind::Advance,
      flags_.test(Flag::InternalUnit), "UNIT=internal-file"); // C1221
  CheckForRequiredSpecifier(flags_.test(Flag::AsynchronousYes),
      "ASYNCHRONOUS='YES'", flags_.test(Flag::NumberUnit),
      "UNIT=number"); // C1224
  CheckForRequiredSpecifier(IoSpecKind::Id, flags_.test(Flag::AsynchronousYes),
      "ASYNCHRONOUS='YES'"); // C1225
  CheckForProhibitedSpecifier(IoSpecKind::Pos, IoSpecKind::Rec); // C1226
  CheckForRequiredSpecifier(IoSpecKind::Decimal, flags_.test(Flag::FmtOrNml),
      "FMT or NML"); // C1227
  CheckForRequiredSpecifier(IoSpecKind::Round, flags_.test(Flag::FmtOrNml),
      "FMT or NML"); // C1227
}

void IoChecker::SetSpecifier(IoSpecKind specKind) {
  if (stmt_ == IoStmtKind::None) {
    // FMT may appear on PRINT statements, which don't have any checks.
    // [IO]MSG and [IO]STAT parse symbols are shared with non-I/O statements.
    return;
  }
  // C1203, C1207, C1210, C1236, C1239, C1242, C1245
  if (specifierSet_.test(specKind)) {
    context_.Say("Duplicate %s specifier"_err_en_US,
        parser::ToUpperCaseLetters(common::EnumToString(specKind)));
  }
  specifierSet_.set(specKind);
}

void IoChecker::CheckStringValue(IoSpecKind specKind, const std::string &value,
    const parser::CharBlock &source) const {
  static std::unordered_map<IoSpecKind, const std::set<std::string>> specValues{
      {IoSpecKind::Access, {"DIRECT", "SEQUENTIAL", "STREAM"}},
      {IoSpecKind::Action, {"READ", "READWRITE", "WRITE"}},
      {IoSpecKind::Advance, {"NO", "YES"}},
      {IoSpecKind::Asynchronous, {"NO", "YES"}},
      {IoSpecKind::Blank, {"NULL", "ZERO"}},
      {IoSpecKind::Decimal, {"COMMA", "POINT"}},
      {IoSpecKind::Delim, {"APOSTROPHE", "NONE", "QUOTE"}},
      {IoSpecKind::Encoding, {"DEFAULT", "UTF-8"}},
      {IoSpecKind::Form, {"FORMATTED", "UNFORMATTED"}},
      {IoSpecKind::Pad, {"NO", "YES"}},
      {IoSpecKind::Position, {"APPEND", "ASIS", "REWIND"}},
      {IoSpecKind::Round,
          {"COMPATIBLE", "DOWN", "NEAREST", "PROCESSOR_DEFINED", "UP", "ZERO"}},
      {IoSpecKind::Sign, {"PLUS", "PROCESSOR_DEFINED", "SUPPRESS"}},
      {IoSpecKind::Status,
          // Open values; Close values are {"DELETE", "KEEP"}.
          {"NEW", "OLD", "REPLACE", "SCRATCH", "UNKNOWN"}},
      {IoSpecKind::Carriagecontrol, {"LIST", "FORTRAN", "NONE"}},
      {IoSpecKind::Convert, {"BIG_ENDIAN", "LITTLE_ENDIAN", "NATIVE"}},
      {IoSpecKind::Dispose, {"DELETE", "KEEP"}},
  };
  if (!specValues.at(specKind).count(parser::ToUpperCaseLetters(value))) {
    context_.Say(source, "Invalid %s value '%s'"_err_en_US,
        parser::ToUpperCaseLetters(common::EnumToString(specKind)), value);
  }
}

// CheckForRequiredSpecifier and CheckForProhibitedSpecifier functions
// need conditions to check, and string arguments to insert into a message.
// An IoSpecKind provides both an absence/presence condition and a string
// argument (its name).  A (condition, string) pair provides an arbitrary
// condition and an arbitrary string.

void IoChecker::CheckForRequiredSpecifier(IoSpecKind specKind) const {
  if (!specifierSet_.test(specKind)) {
    context_.Say("%s statement must have a %s specifier"_err_en_US,
        parser::ToUpperCaseLetters(common::EnumToString(stmt_)),
        parser::ToUpperCaseLetters(common::EnumToString(specKind)));
  }
}

void IoChecker::CheckForRequiredSpecifier(
    bool condition, const std::string &s) const {
  if (!condition) {
    context_.Say("%s statement must have a %s specifier"_err_en_US,
        parser::ToUpperCaseLetters(common::EnumToString(stmt_)), s);
  }
}

void IoChecker::CheckForRequiredSpecifier(
    IoSpecKind specKind1, IoSpecKind specKind2) const {
  if (specifierSet_.test(specKind1) && !specifierSet_.test(specKind2)) {
    context_.Say("If %s appears, %s must also appear"_err_en_US,
        parser::ToUpperCaseLetters(common::EnumToString(specKind1)),
        parser::ToUpperCaseLetters(common::EnumToString(specKind2)));
  }
}

void IoChecker::CheckForRequiredSpecifier(
    IoSpecKind specKind, bool condition, const std::string &s) const {
  if (specifierSet_.test(specKind) && !condition) {
    context_.Say("If %s appears, %s must also appear"_err_en_US,
        parser::ToUpperCaseLetters(common::EnumToString(specKind)), s);
  }
}

void IoChecker::CheckForRequiredSpecifier(
    bool condition, const std::string &s, IoSpecKind specKind) const {
  if (condition && !specifierSet_.test(specKind)) {
    context_.Say("If %s appears, %s must also appear"_err_en_US, s,
        parser::ToUpperCaseLetters(common::EnumToString(specKind)));
  }
}

void IoChecker::CheckForRequiredSpecifier(bool condition1,
    const std::string &s1, bool condition2, const std::string &s2) const {
  if (condition1 && !condition2) {
    context_.Say("If %s appears, %s must also appear"_err_en_US, s1, s2);
  }
}

void IoChecker::CheckForProhibitedSpecifier(IoSpecKind specKind) const {
  if (specifierSet_.test(specKind)) {
    context_.Say("%s statement must not have a %s specifier"_err_en_US,
        parser::ToUpperCaseLetters(common::EnumToString(stmt_)),
        parser::ToUpperCaseLetters(common::EnumToString(specKind)));
  }
}

void IoChecker::CheckForProhibitedSpecifier(
    IoSpecKind specKind1, IoSpecKind specKind2) const {
  if (specifierSet_.test(specKind1) && specifierSet_.test(specKind2)) {
    context_.Say("If %s appears, %s must not appear"_err_en_US,
        parser::ToUpperCaseLetters(common::EnumToString(specKind1)),
        parser::ToUpperCaseLetters(common::EnumToString(specKind2)));
  }
}

void IoChecker::CheckForProhibitedSpecifier(
    IoSpecKind specKind, bool condition, const std::string &s) const {
  if (specifierSet_.test(specKind) && condition) {
    context_.Say("If %s appears, %s must not appear"_err_en_US,
        parser::ToUpperCaseLetters(common::EnumToString(specKind)), s);
  }
}

void IoChecker::CheckForProhibitedSpecifier(
    bool condition, const std::string &s, IoSpecKind specKind) const {
  if (condition && specifierSet_.test(specKind)) {
    context_.Say("If %s appears, %s must not appear"_err_en_US, s,
        parser::ToUpperCaseLetters(common::EnumToString(specKind)));
  }
}

template <typename A>
void IoChecker::CheckForDefinableVariable(
    const A &variable, const std::string &s) const {
  if (const auto *var{parser::Unwrap<parser::Variable>(variable)}) {
    if (auto expr{AnalyzeExpr(context_, *var)}) {
      auto at{var->GetSource()};
      if (auto whyNot{WhyNotModifiable(at, *expr, context_.FindScope(at),
              true /*vectorSubscriptIsOk*/)}) {
        const Symbol *base{GetFirstSymbol(*expr)};
        context_
            .Say(at, "%s variable '%s' must be definable"_err_en_US, s,
                (base ? base->name() : at).ToString())
            .Attach(std::move(*whyNot));
      }
    }
  }
}

void IoChecker::CheckForPureSubprogram() const { // C1597
  CHECK(context_.location());
  if (FindPureProcedureContaining(context_.FindScope(*context_.location()))) {
    context_.Say("External I/O is not allowed in a pure subprogram"_err_en_US);
  }
}

} // namespace Fortran::semantics
