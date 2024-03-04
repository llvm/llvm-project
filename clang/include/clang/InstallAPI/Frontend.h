//===- InstallAPI/Frontend.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Top level wrappers for InstallAPI frontend operations.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INSTALLAPI_FRONTEND_H
#define LLVM_CLANG_INSTALLAPI_FRONTEND_H

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Availability.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/InstallAPI/Context.h"
#include "clang/InstallAPI/Visitor.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/MemoryBuffer.h"

namespace clang {
namespace installapi {

using SymbolFlags = llvm::MachO::SymbolFlags;
using RecordLinkage = llvm::MachO::RecordLinkage;
using GlobalRecord = llvm::MachO::GlobalRecord;
using ObjCInterfaceRecord = llvm::MachO::ObjCInterfaceRecord;

// Represents a collection of frontend records for a library that are tied to a
// darwin target triple.
class FrontendRecordsSlice : public llvm::MachO::RecordsSlice {
public:
  FrontendRecordsSlice(const llvm::Triple &T)
      : llvm::MachO::RecordsSlice({T}) {}

  /// Add non-ObjC global record with attributes from AST.
  ///
  /// \param Name The name of symbol.
  /// \param Linkage The linkage of symbol.
  /// \param GV The kind of global.
  /// \param Avail The availability information tied to the active target
  /// triple.
  /// \param D The pointer to the declaration from traversing AST.
  /// \param Access The intended access level of symbol.
  /// \param Flags The flags that describe attributes of the symbol.
  /// \return The non-owning pointer to added record in slice.
  GlobalRecord *addGlobal(StringRef Name, RecordLinkage Linkage,
                          GlobalRecord::Kind GV,
                          const clang::AvailabilityInfo Avail, const Decl *D,
                          const HeaderType Access,
                          SymbolFlags Flags = SymbolFlags::None);

  /// Add ObjC Class record with attributes from AST.
  ///
  /// \param Name The name of class, not symbol.
  /// \param Linkage The linkage of symbol.
  /// \param Avail The availability information tied to the active target
  /// triple.
  /// \param D The pointer to the declaration from traversing AST.
  /// \param Access The intended access level of symbol.
  /// \param IsEHType Whether declaration has an exception attribute.
  /// \return The non-owning pointer to added record in slice.
  ObjCInterfaceRecord *addObjCInterface(StringRef Name, RecordLinkage Linkage,
                                        const clang::AvailabilityInfo Avail,
                                        const Decl *D, HeaderType Access,
                                        bool IsEHType);

private:
  /// Frontend information captured about records.
  struct FrontendAttrs {
    const AvailabilityInfo Avail;
    const Decl *D;
    const HeaderType Access;
  };

  /// Mapping of records stored in slice to their frontend attributes.
  llvm::DenseMap<llvm::MachO::Record *, FrontendAttrs> FrontendRecords;
};

/// Create a buffer that contains all headers to scan
/// for global symbols with.
std::unique_ptr<llvm::MemoryBuffer> createInputBuffer(InstallAPIContext &Ctx);

class InstallAPIAction : public ASTFrontendAction {
public:
  explicit InstallAPIAction(InstallAPIContext &Ctx) : Ctx(Ctx) {}

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override {
    return std::make_unique<InstallAPIVisitor>(
        CI.getASTContext(), Ctx, CI.getSourceManager(), CI.getPreprocessor());
  }

private:
  InstallAPIContext &Ctx;
};
} // namespace installapi
} // namespace clang

#endif // LLVM_CLANG_INSTALLAPI_FRONTEND_H
