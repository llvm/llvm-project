//===--- OpenCLOptions.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Defines the clang::OpenCLOptions class.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_OPENCLOPTIONS_H
#define LLVM_CLANG_BASIC_OPENCLOPTIONS_H

#include "llvm/ADT/StringRef.h"

namespace clang {

/// \brief OpenCL supported extensions and optional core features
class OpenCLOptions {
public:
#define OPENCLEXT(nm) unsigned nm : 1;
#include "clang/Basic/OpenCLExtensions.def"

  OpenCLOptions() {
#define OPENCLEXT(nm)   nm = 0;
#include "clang/Basic/OpenCLExtensions.def"
  }

  // Enable or disable all options.
  void setAll(bool Enable = true) {
#define OPENCLEXT(nm)   nm = Enable;
#include "clang/Basic/OpenCLExtensions.def"
  }

  /// \brief Enable or disable support for OpenCL extensions
  /// \param Ext name of the extension optionally prefixed with
  ///        '+' or '-'
  /// \param Enable used when \p Ext is not prefixed by '+' or '-'
  void set(llvm::StringRef Ext, bool Enable = true) {
    assert(!Ext.empty() && "Extension is empty.");

    switch (Ext[0]) {
    case '+':
      Enable = true;
      Ext = Ext.drop_front();
      break;
    case '-':
      Enable = false;
      Ext = Ext.drop_front();
      break;
    }

    if (Ext.equals("all")) {
      setAll(Enable);
      return;
    }

#define OPENCLEXT(nm)       \
    if (Ext.equals(#nm)) {  \
      nm = Enable;          \
    }
#include "clang/Basic/OpenCLExtensions.def"
  }

  // Is supported with OpenCL version \p OCLVer.
#define OPENCLEXT_INTERNAL(Ext, Avail, ...) \
  bool is_##Ext##_supported(unsigned OCLVer) const { \
    return Ext && OCLVer >= Avail; \
  }
#include "clang/Basic/OpenCLExtensions.def"


  // Is supported OpenCL extension with OpenCL version \p OCLVer.
  // For supported optional core feature, return false.
#define OPENCLEXT_INTERNAL(Ext, Avail, Core) \
  bool is_##Ext##_supported_extension(unsigned CLVer) const { \
    return is_##Ext##_supported(CLVer) && (Core == ~0U || CLVer < Core); \
  }
#include "clang/Basic/OpenCLExtensions.def"

  // Is supported OpenCL core features with OpenCL version \p OCLVer.
  // For supported extension, return false.
#define OPENCLEXT_INTERNAL(Ext, Avail, Core) \
  bool is_##Ext##_supported_core(unsigned CLVer) const { \
    return is_##Ext##_supported(CLVer) && Core != ~0U && CLVer >= Core; \
  }
#include "clang/Basic/OpenCLExtensions.def"

};

}  // end namespace clang

#endif
