//===- JSONFormatImpl.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Internal implementation header shared by all translation units in the
// JSONFormat subdirectory. Not part of the public API.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_LIB_ANALYSIS_SCALABLE_SERIALIZATION_JSONFORMAT_JSONFORMATIMPL_H
#define CLANG_LIB_ANALYSIS_SCALABLE_SERIALIZATION_JSONFORMAT_JSONFORMATIMPL_H

#include "../../ModelStringConversions.h"
#include "clang/Analysis/Scalable/Model/EntityLinkage.h"
#include "clang/Analysis/Scalable/Serialization/JSONFormat.h"
#include "clang/Analysis/Scalable/Support/ErrorBuilder.h"
#include "clang/Analysis/Scalable/Support/FormatProviders.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Registry.h"

namespace clang::ssaf {

using Array = llvm::json::Array;
using Object = llvm::json::Object;
using Value = llvm::json::Value;

//----------------------------------------------------------------------------
// File Format Constant
//----------------------------------------------------------------------------

inline constexpr const char *JSONFormatFileExtension = ".json";

//----------------------------------------------------------------------------
// Error Message Constants
//----------------------------------------------------------------------------

namespace ErrorMessages {

inline constexpr const char *FailedToReadFile =
    "failed to read file '{0}': {1}";
inline constexpr const char *FailedToWriteFile =
    "failed to write file '{0}': {1}";
inline constexpr const char *FileNotFound = "file does not exist";
inline constexpr const char *FileIsDirectory =
    "path is a directory, not a file";
inline constexpr const char *FileIsNotJSON =
    "file does not end with '{0}' extension";
inline constexpr const char *FileExists = "file already exists";
inline constexpr const char *ParentDirectoryNotFound =
    "parent directory does not exist";

inline constexpr const char *ReadingFromField = "reading {0} from field '{1}'";
inline constexpr const char *WritingToField = "writing {0} to field '{1}'";
inline constexpr const char *ReadingFromIndex = "reading {0} from index '{1}'";
inline constexpr const char *WritingToIndex = "writing {0} to index '{1}'";
inline constexpr const char *ReadingFromFile = "reading {0} from file '{1}'";
inline constexpr const char *WritingToFile = "writing {0} to file '{1}'";

inline constexpr const char *FailedInsertionOnDuplication =
    "failed to insert {0} at index '{1}': encountered duplicate '{2}'";

inline constexpr const char *FailedToReadObject =
    "failed to read {0}: expected JSON {1}";
inline constexpr const char *FailedToReadObjectAtField =
    "failed to read {0} from field '{1}': expected JSON {2}";
inline constexpr const char *FailedToReadObjectAtIndex =
    "failed to read {0} from index '{1}': expected JSON {2}";

inline constexpr const char *FailedToDeserializeEntitySummaryNoFormatInfo =
    "failed to deserialize EntitySummary: no FormatInfo registered for '{0}'";
inline constexpr const char *FailedToSerializeEntitySummaryNoFormatInfo =
    "failed to serialize EntitySummary: no FormatInfo registered for '{0}'";

inline constexpr const char *FailedToDeserializeEntitySummaryMissingData =
    "failed to deserialize EntitySummary: null EntitySummary data for '{0}'";
inline constexpr const char *FailedToSerializeEntitySummaryMissingData =
    "JSONFormat - null EntitySummary data for '{0}'";

inline constexpr const char
    *FailedToDeserializeEntitySummaryMismatchedSummaryName =
        "failed to deserialize EntitySummary: EntitySummary data for '{0}' "
        "reports mismatched '{1}'";
inline constexpr const char
    *FailedToSerializeEntitySummaryMismatchedSummaryName =
        "JSONFormat - EntitySummary data for '{0}' reports mismatched '{1}'";

inline constexpr const char *InvalidBuildNamespaceKind =
    "invalid BuildNamespaceKind value '{0}' for field 'kind'";

inline constexpr const char *InvalidEntityLinkageType =
    "invalid EntityLinkageType value '{0}' for field 'type'";

inline constexpr const char *FailedToDeserializeLinkageTableExtraId =
    "failed to deserialize LinkageTable: extra '{0}' not present in IdTable";

inline constexpr const char *FailedToDeserializeLinkageTableMissingId =
    "failed to deserialize LinkageTable: missing '{0}' present in IdTable";

} // namespace ErrorMessages

//----------------------------------------------------------------------------
// JSON Reader and Writer
//----------------------------------------------------------------------------

llvm::Expected<Value> readJSON(llvm::StringRef Path);
llvm::Error writeJSON(Value &&V, llvm::StringRef Path);

//----------------------------------------------------------------------------
// SummaryName helpers (free functions, anonymous-namespace in .cpp)
//----------------------------------------------------------------------------

SummaryName summaryNameFromJSON(llvm::StringRef SummaryNameStr);
llvm::StringRef summaryNameToJSON(const SummaryName &SN);

//----------------------------------------------------------------------------
// BuildNamespaceKind helpers
//----------------------------------------------------------------------------

llvm::Expected<BuildNamespaceKind>
buildNamespaceKindFromJSON(llvm::StringRef BuildNamespaceKindStr);

// Provided for consistency with respect to rest of the codebase.
llvm::StringRef buildNamespaceKindToJSON(BuildNamespaceKind BNK);

//----------------------------------------------------------------------------
// EntityLinkageType helpers
//----------------------------------------------------------------------------

llvm::Expected<EntityLinkageType>
entityLinkageTypeFromJSON(llvm::StringRef EntityLinkageTypeStr);

// Provided for consistency with respect to rest of the codebase.
llvm::StringRef entityLinkageTypeToJSON(EntityLinkageType LT);

} // namespace clang::ssaf

#endif // CLANG_LIB_ANALYSIS_SCALABLE_SERIALIZATION_JSONFORMAT_JSONFORMATIMPL_H
