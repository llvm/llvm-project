// RUN: %clang_cc1 -fptrauth-intrinsics -fsyntax-only -ferror-limit 1 -verify -std=c++03 %s

/// Force two errors so we hit the error limit leading to skip of template instantiation
# "" // expected-error {{invalid preprocessing directive}}
# ""
// expected-error@* {{too many errors emitted}}

template <typename>
struct a {};

struct b {
  b(int) {}
  void c() {
    /// Trigger the following call stack:
    ///   ...
    ///   clang::ASTContext::findPointerAuthContent(clang::QualType) const /path/to/llvm-project/clang/lib/AST/ASTContext.cpp
    ///   clang::ASTContext::containsAddressDiscriminatedPointerAuth(clang::QualType) const /path/to/llvm-project/clang/lib/AST/ASTContext.cpp
    ///   clang::QualType::isCXX{11|98}PODType(clang::ASTContext const&) const /path/to/llvm-project/clang/lib/AST/Type.cpp
    ///   clang::QualType::isPODType(clang::ASTContext const&) const /path/to/llvm-project/clang/lib/AST/Type.cpp
    ///   SelfReferenceChecker /path/to/llvm-project/clang/lib/Sema/SemaDecl.cpp
    ///   CheckSelfReference /path/to/llvm-project/clang/lib/Sema/SemaDecl.cpp
    ///   clang::Sema::AddInitializerToDecl(clang::Decl*, clang::Expr*, bool) /path/to/llvm-project/clang/lib/Sema/SemaDecl.cpp
    ///   ...
    b d(0);
  }
  a<int> e;
};

