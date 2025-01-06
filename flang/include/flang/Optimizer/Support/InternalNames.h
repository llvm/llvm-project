//===-- Optimizer/Support/InternalNames.h -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_SUPPORT_INTERNALNAMES_H
#define FORTRAN_OPTIMIZER_SUPPORT_INTERNALNAMES_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <optional>

namespace fir {

static constexpr llvm::StringRef kNameSeparator = ".";
static constexpr llvm::StringRef kBoundsSeparator = ".b.";
static constexpr llvm::StringRef kComponentSeparator = ".c.";
static constexpr llvm::StringRef kComponentInitSeparator = ".di.";
static constexpr llvm::StringRef kDataPtrInitSeparator = ".dp.";
static constexpr llvm::StringRef kTypeDescriptorSeparator = ".dt.";
static constexpr llvm::StringRef kKindParameterSeparator = ".kp.";
static constexpr llvm::StringRef kLenKindSeparator = ".lpk.";
static constexpr llvm::StringRef kLenParameterSeparator = ".lv.";
static constexpr llvm::StringRef kNameStringSeparator = ".n.";
static constexpr llvm::StringRef kProcPtrSeparator = ".p.";
static constexpr llvm::StringRef kSpecialBindingSeparator = ".s.";
static constexpr llvm::StringRef kBindingTableSeparator = ".v.";
static constexpr llvm::StringRef boxprocSuffix = "UnboxProc";

/// Internal name mangling of identifiers
///
/// In order to generate symbolically referencable artifacts in a ModuleOp,
/// it is required that those symbols be uniqued.  This is a simple interface
/// for converting Fortran symbols into unique names.
///
/// This is intentionally bijective. Given a symbol's parse name, type, and
/// scope-like information, we can generate a uniqued (mangled) name.  Given a
/// uniqued name, we can return the symbol parse name, type of the symbol, and
/// any scope-like information for that symbol.
struct NameUniquer {
  enum class IntrinsicType { CHARACTER, COMPLEX, INTEGER, LOGICAL, REAL };

  /// The sort of the unique name
  enum class NameKind {
    NOT_UNIQUED,
    BLOCK_DATA_NAME,
    COMMON,
    CONSTANT,
    DERIVED_TYPE,
    DISPATCH_TABLE,
    GENERATED,
    INTRINSIC_TYPE_DESC,
    NAMELIST_GROUP,
    PROCEDURE,
    TYPE_DESC,
    VARIABLE
  };

  /// Components of an unparsed unique name
  struct DeconstructedName {
    DeconstructedName(llvm::StringRef name) : name{name} {}
    DeconstructedName(llvm::ArrayRef<std::string> modules,
                      llvm::ArrayRef<std::string> procs, std::int64_t blockId,
                      llvm::StringRef name, llvm::ArrayRef<std::int64_t> kinds)
        : modules{modules}, procs{procs}, blockId{blockId}, name{name},
          kinds{kinds} {}

    llvm::SmallVector<std::string> modules;
    llvm::SmallVector<std::string> procs;
    std::int64_t blockId;
    std::string name;
    llvm::SmallVector<std::int64_t> kinds;
  };

  /// Unique a common block name
  static std::string doCommonBlock(llvm::StringRef name);

  /// Unique a (global) constant name
  static std::string doConstant(llvm::ArrayRef<llvm::StringRef> modules,
                                llvm::ArrayRef<llvm::StringRef> procs,
                                std::int64_t block, llvm::StringRef name);

  /// Unique a dispatch table name
  static std::string doDispatchTable(llvm::ArrayRef<llvm::StringRef> modules,
                                     llvm::ArrayRef<llvm::StringRef> procs,
                                     std::int64_t block, llvm::StringRef name,
                                     llvm::ArrayRef<std::int64_t> kinds);

  /// Unique a compiler generated name without scope context.
  static std::string doGenerated(llvm::StringRef name);
  /// Unique a compiler generated name with scope context.
  static std::string doGenerated(llvm::ArrayRef<llvm::StringRef> modules,
                                 llvm::ArrayRef<llvm::StringRef> procs,
                                 std::int64_t blockId, llvm::StringRef name);

  /// Unique an intrinsic type descriptor
  static std::string
  doIntrinsicTypeDescriptor(llvm::ArrayRef<llvm::StringRef> modules,
                            llvm::ArrayRef<llvm::StringRef> procs,
                            std::int64_t block, IntrinsicType type,
                            std::int64_t kind);

  /// Unique a procedure name
  static std::string doProcedure(llvm::ArrayRef<llvm::StringRef> modules,
                                 llvm::ArrayRef<llvm::StringRef> procs,
                                 llvm::StringRef name);

  /// Unique a derived type name
  static std::string doType(llvm::ArrayRef<llvm::StringRef> modules,
                            llvm::ArrayRef<llvm::StringRef> procs,
                            std::int64_t block, llvm::StringRef name,
                            llvm::ArrayRef<std::int64_t> kinds);

  /// Unique a (derived) type descriptor name
  static std::string doTypeDescriptor(llvm::ArrayRef<llvm::StringRef> modules,
                                      llvm::ArrayRef<llvm::StringRef> procs,
                                      std::int64_t block, llvm::StringRef name,
                                      llvm::ArrayRef<std::int64_t> kinds);
  static std::string doTypeDescriptor(llvm::ArrayRef<std::string> modules,
                                      llvm::ArrayRef<std::string> procs,
                                      std::int64_t block, llvm::StringRef name,
                                      llvm::ArrayRef<std::int64_t> kinds);

  /// Unique a (global) variable name. A variable with save attribute
  /// defined inside a subprogram also needs to be handled here
  static std::string doVariable(llvm::ArrayRef<llvm::StringRef> modules,
                                llvm::ArrayRef<llvm::StringRef> procs,
                                std::int64_t block, llvm::StringRef name);

  /// Unique a namelist group name
  static std::string doNamelistGroup(llvm::ArrayRef<llvm::StringRef> modules,
                                     llvm::ArrayRef<llvm::StringRef> procs,
                                     llvm::StringRef name);

  /// Entry point for the PROGRAM (called by the runtime)
  /// Can be overridden with the `--main-entry-name=<name>` option.
  static llvm::StringRef doProgramEntry();

  /// Decompose `uniquedName` into the parse name, symbol type, and scope info
  static std::pair<NameKind, DeconstructedName>
  deconstruct(llvm::StringRef uniquedName);

  /// Check if the name is an external facing name.
  static bool isExternalFacingUniquedName(
      const std::pair<NameKind, DeconstructedName> &deconstructResult);

  /// Check whether the name should be re-mangle with external ABI convention.
  static bool needExternalNameMangling(llvm::StringRef uniquedName);

  /// Does \p uniquedName belong to module \p moduleName?
  static bool belongsToModule(llvm::StringRef uniquedName,
                              llvm::StringRef moduleName);

  /// Given a mangled derived type name, get the name of the related derived
  /// type descriptor object. Returns an empty string if \p mangledTypeName is
  /// not a valid mangled derived type name.
  static std::string getTypeDescriptorName(llvm::StringRef mangledTypeName);

  static std::string
  getTypeDescriptorAssemblyName(llvm::StringRef mangledTypeName);

  /// Given a mangled derived type name, get the name of the related binding
  /// table object. Returns an empty string if \p mangledTypeName is not a valid
  /// mangled derived type name.
  static std::string
  getTypeDescriptorBindingTableName(llvm::StringRef mangledTypeName);

  /// Given a mangled derived type name and a component name, get the name of
  /// the global object containing the component default initialization.
  static std::string getComponentInitName(llvm::StringRef mangledTypeName,
                                          llvm::StringRef componentName);

  /// Remove markers that have been added when doing partial type
  /// conversions. mlir::Type cannot be mutated in a pass, so new
  /// fir::RecordType must be created when lowering member types.
  /// Suffixes added to these new types are meaningless and are
  /// dropped in the names passed to LLVM.
  static llvm::StringRef
  dropTypeConversionMarkers(llvm::StringRef mangledTypeName);

  static std::string replaceSpecialSymbols(const std::string &name);

  /// Returns true if the passed name denotes a special symbol (e.g. global
  /// symbol generated for derived type description).
  static bool isSpecialSymbol(llvm::StringRef name);

private:
  static std::string intAsString(std::int64_t i);
  static std::string doKind(std::int64_t kind);
  static std::string doKinds(llvm::ArrayRef<std::int64_t> kinds);
  static std::string toLower(llvm::StringRef name);

  NameUniquer() = delete;
  NameUniquer(const NameUniquer &) = delete;
  NameUniquer(NameUniquer &&) = delete;
  NameUniquer &operator=(const NameUniquer &) = delete;
};

} // namespace fir

#endif // FORTRAN_OPTIMIZER_SUPPORT_INTERNALNAMES_H
