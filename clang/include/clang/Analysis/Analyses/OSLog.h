//= OSLog.h - Analysis of calls to os_log builtins --*- C++ -*-===============//
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

#ifndef LLVM_CLANG_ANALYSIS_ANALYSES_OSLOG_H
#define LLVM_CLANG_ANALYSIS_ANALYSES_OSLOG_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"

namespace clang {
namespace analyze_os_log {

/// An OSLogBufferItem represents a single item in the data written by a call
/// to os_log() or os_trace().
class OSLogBufferItem {
public:
  enum Kind {
    // The item is a scalar (int, float, raw pointer, etc.). No further copying
    // is required. This is the only kind allowed by os_trace().
    ScalarKind = 0,

    // The item is a count, which describes the length of the following item to
    // be copied. A count may only be followed by an item of kind StringKind or
    // PointerKind.
    CountKind,

    // The item is a pointer to a C string. If preceded by a count 'n',
    // os_log() will copy at most 'n' bytes from the pointer.
    StringKind,

    // The item is a pointer to a block of raw data. This item must be preceded
    // by a count 'n'. os_log() will copy exactly 'n' bytes from the pointer.
    PointerKind,

    // The item is a pointer to an Objective-C object. os_log() may retain the
    // object for later processing.
    ObjCObjKind
  };

  enum {
    // The item is marked "private" in the format string.
    IsPrivate = 0x1,

    // The item is marked "public" in the format string.
    IsPublic = 0x2
  };

private:
  Kind TheKind = ScalarKind;
  const Expr *TheExpr = nullptr;
  CharUnits ConstValue;
  CharUnits Size; // size of the data, not including the header bytes
  unsigned Flags = 0;

public:
  OSLogBufferItem(Kind kind, const Expr *expr, CharUnits size, unsigned flags)
    : TheKind(kind), TheExpr(expr), Size(size), Flags(flags) {}

  OSLogBufferItem(ASTContext &Ctx, CharUnits value, unsigned flags)
    : TheKind(CountKind), ConstValue(value),
      Size(Ctx.getTypeSizeInChars(Ctx.IntTy)), Flags(flags) {}

  unsigned char getDescriptorByte() const {
    unsigned char result = 0;
    if (getIsPrivate()) result |= 0x01;
    if (getIsPublic()) result |= 0x02;
    result |= ((unsigned)getKind()) << 4;
    return result;
  }

  unsigned char getSizeByte() const {
    return getSize().getQuantity();
  }

  Kind getKind() const { return TheKind; }
  bool getIsPrivate() const { return (Flags & IsPrivate) != 0; }
  bool getIsPublic() const { return (Flags & IsPublic) != 0; }

  const Expr *getExpr() const { return TheExpr; }
  CharUnits getConstValue() const { return ConstValue; }
  CharUnits getSize() const { return Size; }
};

class OSLogBufferLayout {
public:
  SmallVector<OSLogBufferItem, 4> Items;

  CharUnits getSize() const {
    CharUnits result;
    result += CharUnits::fromQuantity(2); // summary byte, num-args byte
    for (auto &item : Items) {
      // descriptor byte, size byte
      result += item.getSize() + CharUnits::fromQuantity(2);
    }
    return result;
  }

  bool getHasPrivateItems() const {
    return std::any_of(Items.begin(), Items.end(),
      [](const OSLogBufferItem &item) { return item.getIsPrivate(); });
  }

  bool getHasPublicItems() const {
    return std::any_of(Items.begin(), Items.end(),
      [](const OSLogBufferItem &item) { return item.getIsPublic(); });
  }

  bool getHasNonScalar() const {
    return std::any_of(Items.begin(), Items.end(),
      [](const OSLogBufferItem &item) {
        return item.getKind() != OSLogBufferItem::ScalarKind;
      });
  }

  unsigned char getSummaryByte() const {
    unsigned char result = 0;
    if (getHasPrivateItems()) result |= 0x01;
    if (getHasNonScalar()) result |= 0x02;
    return result;
  }

  unsigned char getNumArgsByte() const {
    return Items.size();
  }
};

// Given a call 'E' to one of the builtins __builtin_os_log_format() or
// __builtin_os_log_format_buffer_size(), compute the layout of the buffer that
// the call will write into and store it in 'layout'. Returns 'false' if there
// was some error encountered while computing the layout, and 'true' otherwise.
bool computeOSLogBufferLayout(clang::ASTContext &Ctx, const clang::CallExpr *E,
                              OSLogBufferLayout &layout);

} // namespace analyze_os_log
} // namespace clang
#endif
