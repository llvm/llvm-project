//===----- FileOffset.h - Offset in a file ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_EDIT_FILEOFFSET_H
#define LLVM_CLANG_EDIT_FILEOFFSET_H

#include "clang/Basic/SourceLocation.h"

namespace clang {

namespace edit {

class FileOffset {
  FileID FID;
  unsigned Offs;
public:
  FileOffset() : Offs(0) { }
  FileOffset(FileID fid, unsigned offs) : FID(fid), Offs(offs) { }

  bool isInvalid() const { return FID.isInvalid(); }

  FileID getFID() const { return FID; }
  unsigned getOffset() const { return Offs; }

  FileOffset getWithOffset(unsigned offset) const {
    FileOffset NewOffs = *this;
    NewOffs.Offs += offset;
    return NewOffs;
  }

  friend bool operator==(FileOffset LHS, FileOffset RHS) {
    return LHS.FID == RHS.FID && LHS.Offs == RHS.Offs;
  }
  friend bool operator!=(FileOffset LHS, FileOffset RHS) {
    return !(LHS == RHS);
  }
  friend bool operator<(FileOffset LHS, FileOffset RHS) {
    if (LHS.FID != RHS.FID)
      return LHS.FID < RHS.FID;
    return LHS.Offs < RHS.Offs;
  }
  friend bool operator>(FileOffset LHS, FileOffset RHS) {
    if (LHS.FID != RHS.FID)
      return LHS.FID > RHS.FID;
    return LHS.Offs > RHS.Offs;
  }
  friend bool operator>=(FileOffset LHS, FileOffset RHS) {
    return LHS > RHS || LHS == RHS;
  }
  friend bool operator<=(FileOffset LHS, FileOffset RHS) {
    return LHS < RHS || LHS == RHS;
  }
};

}

} // end namespace clang

#endif
