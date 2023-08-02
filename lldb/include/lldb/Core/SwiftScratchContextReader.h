//===-- SwiftASTContextReader.h ---------------------------------*- C++ -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2018 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_SwiftASTContextReader_h_
#define liblldb_SwiftASTContextReader_h_

#include <shared_mutex>

namespace lldb_private {

class TypeSystemSwiftTypeRefForExpressions;
class ExecutionContext;
class ExecutionContextRef;

/// A scratch Swift context pointer and its reader lock.
/// The Swift scratch context may need to be replaced when it gets corrupted,
/// for example due to incompatible ClangImporter options. This locking
/// mechanism guarantees that this won't happen while a client is using the
/// context.
///
/// In Swift there are three use-cases for ASTContexts with
/// different requirements and guarantees.
///
/// - Module ASTContexts are used for the static type system. They
///   are created once for each lldb::Module and live forever.
///
/// - Scratch AST Contexts are used for expressions
///   (thus far everything is like in the Clang language support)
///
/// - Scratch AST Contexts are also used to express the results of
///    any dynamic type resolution done by RemoteAST or Archetype
///    binding.
///
/// Because expressions and dynamic type resolution may trigger the
/// import of another module, the scratch context may become
/// unusable. When a scratch context is in a fatal error state,
/// GetSwiftScratchContext() will create a fresh global context,
/// or even separate scratch contexts for each lldb::Module. But it
/// will only do this if no client holds on to a read lock on \b
/// m_scratch_typesystem_lock.
class SwiftScratchContextReader {
  std::shared_lock<std::shared_mutex> m_lock;
  TypeSystemSwiftTypeRefForExpressions *m_ts;

public:
  SwiftScratchContextReader(std::shared_lock<std::shared_mutex> &&lock,
                            TypeSystemSwiftTypeRefForExpressions &ts);
  SwiftScratchContextReader(const SwiftScratchContextReader &) = delete;
  SwiftScratchContextReader(SwiftScratchContextReader &&other) = default;
  SwiftScratchContextReader &
  operator=(SwiftScratchContextReader &&other) = default;
  TypeSystemSwiftTypeRefForExpressions *get() { return m_ts; }
  TypeSystemSwiftTypeRefForExpressions *operator->() { return get(); }
  TypeSystemSwiftTypeRefForExpressions &operator*() { return *get(); }
};

/// An RAII object that just acquires the reader lock.
struct SwiftScratchContextLock {
  std::shared_lock<std::shared_mutex> lock;
  SwiftScratchContextLock(const ExecutionContextRef *exe_ctx_ref);
  SwiftScratchContextLock(const ExecutionContext *exe_ctx);
};

} // namespace lldb_private
#endif
