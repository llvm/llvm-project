//===- CompileJobCacheResult.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_COMPILEJOBCACHERESULT_H
#define LLVM_CLANG_FRONTEND_COMPILEJOBCACHERESULT_H

#include "clang/Basic/LLVM.h"
#include "llvm/CAS/CASNodeSchema.h"
#include "llvm/CAS/ObjectStore.h"

namespace clang {
namespace cas {
class CompileJobResultSchema;

class CompileJobCacheResult : public ObjectProxy {
public:
  /// Categorization for the output kinds that is used to decouple the
  /// compilation cache key from the specific output paths.
  enum class OutputKind : char {
    MainOutput, ///< Main output file, e.g. object file, pcm file, etc.
    SerializedDiagnostics,
    Dependencies,
    Stderr, ///< Contents of stderr.
  };

  /// Returns all \c OutputKind values.
  static ArrayRef<OutputKind> getAllOutputKinds();

  /// A single output file or stream.
  struct Output {
    /// The CAS object for this output.
    ObjectRef Object;
    /// The output kind.
    OutputKind Kind;

    bool operator==(const Output &Other) const {
      return Object == Other.Object && Kind == Other.Kind;
    }
  };

  /// Retrieves each \c Output from this result.
  llvm::Error
  forEachOutput(llvm::function_ref<llvm::Error(Output)> Callback) const;

  size_t getNumOutputs() const;

  /// Retrieves a specific output specified by \p Kind, if it exists.
  Optional<Output> getOutput(OutputKind Kind) const;

  /// Print this result to \p OS.
  llvm::Error print(llvm::raw_ostream &OS);

  /// Helper to build a \c CompileJobCacheResult from individual outputs.
  class Builder {
  public:
    Builder();
    ~Builder();
    /// Treat outputs added for \p Path as having the given \p Kind. Otherwise
    /// they will have kind \c Unknown.
    void addKindMap(OutputKind Kind, StringRef Path);
    /// Add an output with an explicit \p Kind.
    void addOutput(OutputKind Kind, ObjectRef Object);
    /// Add an output for the given \p Path. There must be a a kind map for it.
    llvm::Error addOutput(StringRef Path, ObjectRef Object);
    /// Build a single \c ObjectRef representing the provided outputs. The
    /// result can be used with \c CompileJobResultSchema to retrieve the
    /// original outputs.
    Expected<ObjectRef> build(ObjectStore &CAS);

  private:
    struct PrivateImpl;
    PrivateImpl &Impl;
  };

private:
  ObjectRef getOutputObject(size_t I) const;
  ObjectRef getPathsListRef() const;
  OutputKind getOutputKind(size_t I) const;
  Expected<ObjectRef> getOutputPath(size_t I) const;

private:
  friend class CompileJobResultSchema;
  CompileJobCacheResult(const ObjectProxy &);
};

class CompileJobResultSchema
    : public llvm::RTTIExtends<CompileJobResultSchema, llvm::cas::NodeSchema> {
public:
  static char ID;

  CompileJobResultSchema(ObjectStore &CAS);

  /// Attempt to load \p Ref as a \c CompileJobCacheResult if it matches the
  /// schema.
  Expected<CompileJobCacheResult> load(ObjectRef Ref) const;

  bool isRootNode(const ObjectProxy &Node) const final;
  bool isNode(const ObjectProxy &Node) const final;

  /// Get this schema's marker node.
  ObjectRef getKindRef() const { return KindRef; }

private:
  ObjectRef KindRef;
};

} // namespace cas
} // namespace clang

#endif // LLVM_CLANG_FRONTEND_COMPILEJOBCACHERESULT_H
