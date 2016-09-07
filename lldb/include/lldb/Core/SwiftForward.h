//===-- SwiftForward.h ------------------------------------------*- C++ -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2015 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See http://swift.org/LICENSE.txt for license information
// See http://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_SwiftForward_h_
#define liblldb_SwiftForward_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes

#if defined(__cplusplus)

namespace swift {
class ArchetypeType;
class ArrayType;
class ASTContext;
class ASTMutationListener;
class BoundGenericType;
class ClangImporter;
class ClangImporterOptions;
class ClangModule;
class CompilerInvocation;
class DebuggerClient;
class Decl;
class DeclContext;
class DiagnosticConsumer;
class DiagnosticEngine;
class Expr;
class ExtensionDecl;
class FuncDecl;
class FunctionType;
class Identifier;
class IRGenOptions;
class LangOptions;
class ModuleDecl;
class ModuleLoader;
class ModuleLoadListener;
class NameAliasType;
class NominalType;
class NominalTypeDecl;
class ProtocolDecl;
class SearchPathOptions;
class SerializedModuleLoader;
class SILModule;
class SILOptions;
class SourceFile;
class SourceLoc;
class SourceManager;
class StructDecl;
class Substitution;
class SubstitutableType;
class TupleType;
class TupleTypeElt;
class Type;
class TypeBase;
class TypeDecl;
class TypeInfo;
class UnionElementDecl;
class ValueDecl;
class VarDecl;

namespace irgen {
class FixedTypeInfo;
class IRGenModule;
class IRGenerator;
class TypeInfo;
}
}

#endif // #if defined(__cplusplus)
#endif // liblldb_SwiftForward_h_
