//===-- DeclVendor.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_DeclVendor_h_
#define liblldb_DeclVendor_h_

#include "lldb/Core/ClangForward.h"
#include "lldb/lldb-defines.h"

#include "clang/AST/ExternalASTMerger.h"

#include <vector>

namespace lldb_private {

// The Decl vendor class is intended as a generic interface to search for named
// declarations that are not necessarily backed by a specific symbol file.
class DeclVendor {
public:
  // Constructors and Destructors
  DeclVendor() {}

  virtual ~DeclVendor() {}

  /// Look up the set of Decls that the DeclVendor currently knows about
  /// matching a given name.
  ///
  /// \param[in] name
  ///     The name to look for.
  ///
  /// \param[in] append
  ///     If true, FindDecls will clear "decls" when it starts.
  ///
  /// \param[in] max_matches
  ///     The maximum number of Decls to return.  UINT32_MAX means "as
  ///     many as possible."
  ///
  /// \return
  ///     The number of Decls added to decls; will not exceed
  ///     max_matches.
  virtual uint32_t FindDecls(ConstString name, bool append,
                             uint32_t max_matches,
                             std::vector<clang::NamedDecl *> &decls) = 0;

  /// Look up the types that the DeclVendor currently knows about matching a
  /// given name.
  ///
  /// \param[in] name
  ///     The name to look for.
  ///
  /// \param[in] max_matches
  //      The maximum number of matches. UINT32_MAX means "as many as possible".
  ///
  /// \return
  ///     The vector of CompilerTypes that was found.
  std::vector<CompilerType> FindTypes(ConstString name, uint32_t max_matches);

  /// Interface for ExternalASTMerger.  Returns an ImporterSource 
  /// allowing type completion.
  ///
  /// \return
  ///     An ImporterSource for this DeclVendor.
  virtual clang::ExternalASTMerger::ImporterSource GetImporterSource() = 0;

private:
  // For DeclVendor only
  DISALLOW_COPY_AND_ASSIGN(DeclVendor);
};

} // namespace lldb_private

#endif
