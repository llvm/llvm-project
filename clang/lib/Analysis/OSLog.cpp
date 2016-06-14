//===--- OSLog.cpp - Analysis of calls to os_log builtins -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines APIs for determining the layout of the data buffer for
// os_log() and os_trace().
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Analyses/OSLog.h"
#include "clang/Analysis/Analyses/FormatString.h"
#include "clang/Basic/Builtins.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprObjC.h"
#include "llvm/ADT/SmallBitVector.h"

using namespace clang;
using llvm::APInt;

using clang::analyze_os_log::OSLogBufferItem;
using clang::analyze_os_log::OSLogBufferLayout;

class OSLogFormatStringHandler
  : public analyze_format_string::FormatStringHandler {
private:
  ArrayRef<const Expr *> Args;
  SmallVector<Optional<OSLogBufferItem::Kind>, 4> ArgKind;
  SmallVector<Optional<unsigned>, 4> ArgSize;
  SmallVector<unsigned char, 4> ArgFlags;

public:
  OSLogFormatStringHandler(ArrayRef<const Expr *> args)
    : FormatStringHandler(), Args(args), ArgKind(args.size(), None),
      ArgSize(args.size(), None), ArgFlags(args.size(), 0)
  {}

  virtual bool HandlePrintfSpecifier(const analyze_printf::PrintfSpecifier &FS,
                                     const char *startSpecifier,
                                     unsigned specifierLen) {

    // Cases to handle:
    //  * "%f", "%d"... scalar (assumed for anything that doesn't fit the below
    //    cases)
    //  * "%s" pointer to null-terminated string
    //  * "%.*s" strlen (arg), pointer to string
    //  * "%.16s" strlen (non-arg), pointer to string
    //  * "%.*P" len (arg), pointer to data
    //  * "%.16P" len (non-arg), pointer to data
    //  * "%@" pointer to objc object

    unsigned argIndex = FS.getArgIndex();
    if (argIndex >= Args.size()) {
      return false;
    }
    switch (FS.getConversionSpecifier().getKind()) {
    case clang::analyze_format_string::ConversionSpecifier::sArg: { // "%s"
      ArgKind[argIndex] = OSLogBufferItem::StringKind;
      auto &precision = FS.getPrecision();
      switch (precision.getHowSpecified()) {
      case clang::analyze_format_string::OptionalAmount::NotSpecified: // "%s"
        break;
      case clang::analyze_format_string::OptionalAmount::Constant: // "%.16s"
        ArgSize[argIndex] = precision.getConstantAmount();
        break;
      case clang::analyze_format_string::OptionalAmount::Arg: // "%.*s"
        ArgKind[precision.getArgIndex()] = OSLogBufferItem::CountKind;
        break;
      case clang::analyze_format_string::OptionalAmount::Invalid:
        return false;
      }
      break;
    }
    case clang::analyze_format_string::ConversionSpecifier::PArg: { // "%P"
      ArgKind[argIndex] = OSLogBufferItem::PointerKind;
      auto &precision = FS.getPrecision();
      switch (precision.getHowSpecified()) {
      case clang::analyze_format_string::OptionalAmount::NotSpecified: // "%P"
        return false; // length must be supplied with pointer format specifier
      case clang::analyze_format_string::OptionalAmount::Constant: // "%.16P"
        ArgSize[argIndex] = precision.getConstantAmount();
        break;
      case clang::analyze_format_string::OptionalAmount::Arg: // "%.*P"
        ArgKind[precision.getArgIndex()] = OSLogBufferItem::CountKind;
        break;
      case clang::analyze_format_string::OptionalAmount::Invalid:
        return false;
      }
      break;
    }
    case clang::analyze_format_string::ConversionSpecifier::ObjCObjArg: // "%@"
      ArgKind[argIndex] = OSLogBufferItem::ObjCObjKind;
      break;
    default:
      ArgKind[argIndex] = OSLogBufferItem::ScalarKind;
      break;
    }

    if (FS.isPrivate()) {
      ArgFlags[argIndex] |= OSLogBufferItem::IsPrivate;
    }
    if (FS.isPublic()) {
      ArgFlags[argIndex] |= OSLogBufferItem::IsPublic;
    }
    return true;
  }

  void computeLayout(ASTContext &Ctx, OSLogBufferLayout &layout) const {
    layout.Items.clear();
    for (unsigned i = 0; i < Args.size(); i++) {
      const Expr *arg = Args[i];
      if (ArgSize[i]) {
        layout.Items.emplace_back(Ctx, CharUnits::fromQuantity(*ArgSize[i]),
                                  ArgFlags[i]);
      }
      CharUnits size = Ctx.getTypeSizeInChars(arg->getType());
      if (ArgKind[i]) {
        layout.Items.emplace_back(*ArgKind[i], arg, size, ArgFlags[i]);
      } else {
        layout.Items.emplace_back(OSLogBufferItem::ScalarKind, arg, size,
                                  ArgFlags[i]);
      }
    }
  }
};

bool clang::analyze_os_log::computeOSLogBufferLayout(ASTContext &Ctx,
                                                     const CallExpr *E,
                                                     OSLogBufferLayout &layout)
{
  ArrayRef<const Expr *> Args(E->getArgs(), E->getArgs() + E->getNumArgs());

  const Expr *StringArg;
  ArrayRef<const Expr *> VarArgs;
  switch (E->getBuiltinCallee()) {
  case Builtin::BI__builtin_os_log_format_buffer_size:
    assert(E->getNumArgs() >= 1 &&
           "__builtin_os_log_format_buffer_size takes at least 1 argument");
    StringArg = E->getArg(0);
    VarArgs = Args.slice(1);
    break;
  case Builtin::BI__builtin_os_log_format:
    assert(E->getNumArgs() >= 2 &&
           "__builtin_os_log_format takes at least 2 arguments");
    StringArg = E->getArg(1);
    VarArgs = Args.slice(2);
    break;
  default:
    llvm_unreachable("non-os_log builtin passed to computeOSLogBufferLayout");
  }

  const StringLiteral *Lit = cast<StringLiteral>(StringArg->IgnoreParenCasts());
  assert(Lit && (Lit->isAscii() || Lit->isUTF8()));
  StringRef data = Lit->getString();
  OSLogFormatStringHandler H(VarArgs);
  ParsePrintfString(H, data.begin(), data.end(), Ctx.getLangOpts(),
                    Ctx.getTargetInfo(), /*isFreeBSDKPrintf*/false);

  H.computeLayout(Ctx, layout);
  return true;
}
