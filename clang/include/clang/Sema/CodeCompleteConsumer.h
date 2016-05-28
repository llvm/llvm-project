//===---- CodeCompleteConsumer.h - Code Completion Interface ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the CodeCompleteConsumer class.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_SEMA_CODECOMPLETECONSUMER_H
#define LLVM_CLANG_SEMA_CODECOMPLETECONSUMER_H

#include "clang-c/Index.h"
#include "clang/AST/CanonicalType.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/Type.h"
#include "clang/Sema/CodeCompleteOptions.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"
#include <string>
#include <utility>

namespace clang {

class Decl;

/// \brief Default priority values for code-completion results based
/// on their kind.
enum {
  /// \brief Priority for the next initialization in a constructor initializer
  /// list.
  CCP_NextInitializer = 7,
  /// \brief Priority for an enumeration constant inside a switch whose
  /// condition is of the enumeration type.
  CCP_EnumInCase = 7,
  /// \brief Priority for a send-to-super completion.
  CCP_SuperCompletion = 20,
  /// \brief Priority for a declaration that is in the local scope.
  CCP_LocalDeclaration = 34,
  /// \brief Priority for a member declaration found from the current
  /// method or member function.
  CCP_MemberDeclaration = 35,
  /// \brief Priority for a language keyword (that isn't any of the other
  /// categories).
  CCP_Keyword = 40,
  /// \brief Priority for a code pattern.
  CCP_CodePattern = 40,
  /// \brief Priority for a non-type declaration.
  CCP_Declaration = 50,
  /// \brief Priority for a type.
  CCP_Type = CCP_Declaration,
  /// \brief Priority for a constant value (e.g., enumerator).
  CCP_Constant = 65,
  /// \brief Priority for a preprocessor macro.
  CCP_Macro = 70,
  /// \brief Priority for a nested-name-specifier.
  CCP_NestedNameSpecifier = 75,
  /// \brief Priority for a result that isn't likely to be what the user wants,
  /// but is included for completeness.
  CCP_Unlikely = 80,

  /// \brief Priority for the Objective-C "_cmd" implicit parameter.
  CCP_ObjC_cmd = CCP_Unlikely
};

/// \brief Priority value deltas that are added to code-completion results
/// based on the context of the result.
enum {
  /// \brief The result is in a base class.
  CCD_InBaseClass = 2,
  /// \brief The result is a C++ non-static member function whose qualifiers
  /// exactly match the object type on which the member function can be called.
  CCD_ObjectQualifierMatch = -1,
  /// \brief The selector of the given message exactly matches the selector
  /// of the current method, which might imply that some kind of delegation
  /// is occurring.
  CCD_SelectorMatch = -3,

  /// \brief Adjustment to the "bool" type in Objective-C, where the typedef
  /// "BOOL" is preferred.
  CCD_bool_in_ObjC = 1,

  /// \brief Adjustment for KVC code pattern priorities when it doesn't look
  /// like the
  CCD_ProbablyNotObjCCollection = 15,

  /// \brief An Objective-C method being used as a property.
  CCD_MethodAsProperty = 2
};

/// \brief Priority value factors by which we will divide or multiply the
/// priority of a code-completion result.
enum {
  /// \brief Divide by this factor when a code-completion result's type exactly
  /// matches the type we expect.
  CCF_ExactTypeMatch = 4,
  /// \brief Divide by this factor when a code-completion result's type is
  /// similar to the type we expect (e.g., both arithmetic types, both
  /// Objective-C object pointer types).
  CCF_SimilarTypeMatch = 2
};

/// \brief A simplified classification of types used when determining
/// "similar" types for code completion.
enum SimplifiedTypeClass {
  STC_Arithmetic,
  STC_Array,
  STC_Block,
  STC_Function,
  STC_ObjectiveC,
  STC_Other,
  STC_Pointer,
  STC_Record,
  STC_Void
};

/// \brief Determine the simplified type class of the given canonical type.
SimplifiedTypeClass getSimplifiedTypeClass(CanQualType T);

/// \brief Determine the type that this declaration will have if it is used
/// as a type or in an expression.
QualType getDeclUsageType(ASTContext &C, const NamedDecl *ND);

/// \brief Determine the priority to be given to a macro code completion result
/// with the given name.
///
/// \param MacroName The name of the macro.
///
/// \param LangOpts Options describing the current language dialect.
///
/// \param PreferredTypeIsPointer Whether the preferred type for the context
/// of this macro is a pointer type.
unsigned getMacroUsagePriority(StringRef MacroName,
                               const LangOptions &LangOpts,
                               bool PreferredTypeIsPointer = false);

/// \brief Determine the libclang cursor kind associated with the given
/// declaration.
CXCursorKind getCursorKindForDecl(const Decl *D);

class FunctionDecl;
class FunctionType;
class FunctionTemplateDecl;
class IdentifierInfo;
class NamedDecl;
class NestedNameSpecifier;
class Sema;

/// \brief The context in which code completion occurred, so that the
/// code-completion consumer can process the results accordingly.
class CodeCompletionContext {
public:
  enum Kind {
    /// \brief An unspecified code-completion context.
    CCC_Other,
    /// \brief An unspecified code-completion context where we should also add
    /// macro completions.
    CCC_OtherWithMacros,
    /// \brief Code completion occurred within a "top-level" completion context,
    /// e.g., at namespace or global scope.
    CCC_TopLevel,
    /// \brief Code completion occurred within an Objective-C interface,
    /// protocol, or category interface.
    CCC_ObjCInterface,
    /// \brief Code completion occurred within an Objective-C implementation
    /// or category implementation.
    CCC_ObjCImplementation,
    /// \brief Code completion occurred within the instance variable list of
    /// an Objective-C interface, implementation, or category implementation.
    CCC_ObjCIvarList,
    /// \brief Code completion occurred within a class, struct, or union.
    CCC_ClassStructUnion,
    /// \brief Code completion occurred where a statement (or declaration) is
    /// expected in a function, method, or block.
    CCC_Statement,
    /// \brief Code completion occurred where an expression is expected.
    CCC_Expression,
    /// \brief Code completion occurred where an Objective-C message receiver
    /// is expected.
    CCC_ObjCMessageReceiver,
    /// \brief Code completion occurred on the right-hand side of a member
    /// access expression using the dot operator.
    ///
    /// The results of this completion are the members of the type being
    /// accessed. The type itself is available via
    /// \c CodeCompletionContext::getType().
    CCC_DotMemberAccess,
    /// \brief Code completion occurred on the right-hand side of a member
    /// access expression using the arrow operator.
    ///
    /// The results of this completion are the members of the type being
    /// accessed. The type itself is available via
    /// \c CodeCompletionContext::getType().
    CCC_ArrowMemberAccess,
    /// \brief Code completion occurred on the right-hand side of an Objective-C
    /// property access expression.
    ///
    /// The results of this completion are the members of the type being
    /// accessed. The type itself is available via
    /// \c CodeCompletionContext::getType().
    CCC_ObjCPropertyAccess,
    /// \brief Code completion occurred after the "enum" keyword, to indicate
    /// an enumeration name.
    CCC_EnumTag,
    /// \brief Code completion occurred after the "union" keyword, to indicate
    /// a union name.
    CCC_UnionTag,
    /// \brief Code completion occurred after the "struct" or "class" keyword,
    /// to indicate a struct or class name.
    CCC_ClassOrStructTag,
    /// \brief Code completion occurred where a protocol name is expected.
    CCC_ObjCProtocolName,
    /// \brief Code completion occurred where a namespace or namespace alias
    /// is expected.
    CCC_Namespace,
    /// \brief Code completion occurred where a type name is expected.
    CCC_Type,
    /// \brief Code completion occurred where a new name is expected.
    CCC_Name,
    /// \brief Code completion occurred where a new name is expected and a
    /// qualified name is permissible.
    CCC_PotentiallyQualifiedName,
    /// \brief Code completion occurred where an macro is being defined.
    CCC_MacroName,
    /// \brief Code completion occurred where a macro name is expected
    /// (without any arguments, in the case of a function-like macro).
    CCC_MacroNameUse,
    /// \brief Code completion occurred within a preprocessor expression.
    CCC_PreprocessorExpression,
    /// \brief Code completion occurred where a preprocessor directive is
    /// expected.
    CCC_PreprocessorDirective,
    /// \brief Code completion occurred in a context where natural language is
    /// expected, e.g., a comment or string literal.
    ///
    /// This context usually implies that no completions should be added,
    /// unless they come from an appropriate natural-language dictionary.
    CCC_NaturalLanguage,
    /// \brief Code completion for a selector, as in an \@selector expression.
    CCC_SelectorName,
    /// \brief Code completion within a type-qualifier list.
    CCC_TypeQualifiers,
    /// \brief Code completion in a parenthesized expression, which means that
    /// we may also have types here in C and Objective-C (as well as in C++).
    CCC_ParenthesizedExpression,
    /// \brief Code completion where an Objective-C instance message is
    /// expected.
    CCC_ObjCInstanceMessage,
    /// \brief Code completion where an Objective-C class message is expected.
    CCC_ObjCClassMessage,
    /// \brief Code completion where the name of an Objective-C class is
    /// expected.
    CCC_ObjCInterfaceName,
    /// \brief Code completion where an Objective-C category name is expected.
    CCC_ObjCCategoryName,
    /// \brief An unknown context, in which we are recovering from a parsing
    /// error and don't know which completions we should give.
    CCC_Recovery
  };

private:
  enum Kind Kind;

  /// \brief The type that would prefer to see at this point (e.g., the type
  /// of an initializer or function parameter).
  QualType PreferredType;

  /// \brief The type of the base object in a member access expression.
  QualType BaseType;

  /// \brief The identifiers for Objective-C selector parts.
  ArrayRef<IdentifierInfo *> SelIdents;

public:
  /// \brief Construct a new code-completion context of the given kind.
  CodeCompletionContext(enum Kind Kind) : Kind(Kind), SelIdents(None) { }

  /// \brief Construct a new code-completion context of the given kind.
  CodeCompletionContext(enum Kind Kind, QualType T,
                        ArrayRef<IdentifierInfo *> SelIdents = None)
                        : Kind(Kind),
                          SelIdents(SelIdents) {
    if (Kind == CCC_DotMemberAccess || Kind == CCC_ArrowMemberAccess ||
        Kind == CCC_ObjCPropertyAccess || Kind == CCC_ObjCClassMessage ||
        Kind == CCC_ObjCInstanceMessage)
      BaseType = T;
    else
      PreferredType = T;
  }

  /// \brief Retrieve the kind of code-completion context.
  enum Kind getKind() const { return Kind; }

  /// \brief Retrieve the type that this expression would prefer to have, e.g.,
  /// if the expression is a variable initializer or a function argument, the
  /// type of the corresponding variable or function parameter.
  QualType getPreferredType() const { return PreferredType; }

  /// \brief Retrieve the type of the base object in a member-access
  /// expression.
  QualType getBaseType() const { return BaseType; }

  /// \brief Retrieve the Objective-C selector identifiers.
  ArrayRef<IdentifierInfo *> getSelIdents() const { return SelIdents; }

  /// \brief Determines whether we want C++ constructors as results within this
  /// context.
  bool wantConstructorResults() const;
};


/// \brief A "string" used to describe how code completion can
/// be performed for an entity.
///
/// A code completion string typically shows how a particular entity can be
/// used. For example, the code completion string for a function would show
/// the syntax to call it, including the parentheses, placeholders for the
/// arguments, etc.
class CodeCompletionString {
public:
  /// \brief The different kinds of "chunks" that can occur within a code
  /// completion string.
  enum ChunkKind {
    /// \brief The piece of text that the user is expected to type to
    /// match the code-completion string, typically a keyword or the name of a
    /// declarator or macro.
    CK_TypedText,
    /// \brief A piece of text that should be placed in the buffer, e.g.,
    /// parentheses or a comma in a function call.
    CK_Text,
    /// \brief A code completion string that is entirely optional. For example,
    /// an optional code completion string that describes the default arguments
    /// in a function call.
    CK_Optional,
    /// \brief A string that acts as a placeholder for, e.g., a function
    /// call argument.
    CK_Placeholder,
    /// \brief A piece of text that describes something about the result but
    /// should not be inserted into the buffer.
    CK_Informative,
    /// \brief A piece of text that describes the type of an entity or, for
    /// functions and methods, the return type.
    CK_ResultType,
    /// \brief A piece of text that describes the parameter that corresponds
    /// to the code-completion location within a function call, message send,
    /// macro invocation, etc.
    CK_CurrentParameter,
    /// \brief A left parenthesis ('(').
    CK_LeftParen,
    /// \brief A right parenthesis (')').
    CK_RightParen,
    /// \brief A left bracket ('[').
    CK_LeftBracket,
    /// \brief A right bracket (']').
    CK_RightBracket,
    /// \brief A left brace ('{').
    CK_LeftBrace,
    /// \brief A right brace ('}').
    CK_RightBrace,
    /// \brief A left angle bracket ('<').
    CK_LeftAngle,
    /// \brief A right angle bracket ('>').
    CK_RightAngle,
    /// \brief A comma separator (',').
    CK_Comma,
    /// \brief A colon (':').
    CK_Colon,
    /// \brief A semicolon (';').
    CK_SemiColon,
    /// \brief An '=' sign.
    CK_Equal,
    /// \brief Horizontal whitespace (' ').
    CK_HorizontalSpace,
    /// \brief Vertical whitespace ('\\n' or '\\r\\n', depending on the
    /// platform).
    CK_VerticalSpace
  };

  /// \brief One piece of the code completion string.
  struct Chunk {
    /// \brief The kind of data stored in this piece of the code completion
    /// string.
    ChunkKind Kind;

    union {
      /// \brief The text string associated with a CK_Text, CK_Placeholder,
      /// CK_Informative, or CK_Comma chunk.
      /// The string is owned by the chunk and will be deallocated
      /// (with delete[]) when the chunk is destroyed.
      const char *Text;

      /// \brief The code completion string associated with a CK_Optional chunk.
      /// The optional code completion string is owned by the chunk, and will
      /// be deallocated (with delete) when the chunk is destroyed.
      CodeCompletionString *Optional;
    };

    Chunk() : Kind(CK_Text), Text(nullptr) { }

    explicit Chunk(ChunkKind Kind, const char *Text = "");

    /// \brief Create a new text chunk.
    static Chunk CreateText(const char *Text);

    /// \brief Create a new optional chunk.
    static Chunk CreateOptional(CodeCompletionString *Optional);

    /// \brief Create a new placeholder chunk.
    static Chunk CreatePlaceholder(const char *Placeholder);

    /// \brief Create a new informative chunk.
    static Chunk CreateInformative(const char *Informative);

    /// \brief Create a new result type chunk.
    static Chunk CreateResultType(const char *ResultType);

    /// \brief Create a new current-parameter chunk.
    static Chunk CreateCurrentParameter(const char *CurrentParameter);
  };

private:
  /// \brief The number of chunks stored in this string.
  unsigned NumChunks : 16;

  /// \brief The number of annotations for this code-completion result.
  unsigned NumAnnotations : 16;

  /// \brief The priority of this code-completion string.
  unsigned Priority : 16;

  /// \brief The availability of this code-completion result.
  unsigned Availability : 2;
  
  /// \brief The name of the parent context.
  StringRef ParentName;

  /// \brief A brief documentation comment attached to the declaration of
  /// entity being completed by this result.
  const char *BriefComment;
  
  CodeCompletionString(const CodeCompletionString &) = delete;
  void operator=(const CodeCompletionString &) = delete;

  CodeCompletionString(const Chunk *Chunks, unsigned NumChunks,
                       unsigned Priority, CXAvailabilityKind Availability,
                       const char **Annotations, unsigned NumAnnotations,
                       StringRef ParentName,
                       const char *BriefComment);
  ~CodeCompletionString() = default;

  friend class CodeCompletionBuilder;
  friend class CodeCompletionResult;

public:
  typedef const Chunk *iterator;
  iterator begin() const { return reinterpret_cast<const Chunk *>(this + 1); }
  iterator end() const { return begin() + NumChunks; }
  bool empty() const { return NumChunks == 0; }
  unsigned size() const { return NumChunks; }

  const Chunk &operator[](unsigned I) const {
    assert(I < size() && "Chunk index out-of-range");
    return begin()[I];
  }

  /// \brief Returns the text in the TypedText chunk.
  const char *getTypedText() const;

  /// \brief Retrieve the priority of this code completion result.
  unsigned getPriority() const { return Priority; }

  /// \brief Retrieve the availability of this code completion result.
  unsigned getAvailability() const { return Availability; }

  /// \brief Retrieve the number of annotations for this code completion result.
  unsigned getAnnotationCount() const;

  /// \brief Retrieve the annotation string specified by \c AnnotationNr.
  const char *getAnnotation(unsigned AnnotationNr) const;
  
  /// \brief Retrieve the name of the parent context.
  StringRef getParentContextName() const {
    return ParentName;
  }

  const char *getBriefComment() const {
    return BriefComment;
  }
  
  /// \brief Retrieve a string representation of the code completion string,
  /// which is mainly useful for debugging.
  std::string getAsString() const;
};

/// \brief An allocator used specifically for the purpose of code completion.
class CodeCompletionAllocator : public llvm::BumpPtrAllocator {
public:
  /// \brief Copy the given string into this allocator.
  const char *CopyString(const Twine &String);
};

/// \brief Allocator for a cached set of global code completions.
class GlobalCodeCompletionAllocator 
  : public CodeCompletionAllocator,
    public RefCountedBase<GlobalCodeCompletionAllocator>
{

};

class CodeCompletionTUInfo {
  llvm::DenseMap<const DeclContext *, StringRef> ParentNames;
  IntrusiveRefCntPtr<GlobalCodeCompletionAllocator> AllocatorRef;

public:
  explicit CodeCompletionTUInfo(
      IntrusiveRefCntPtr<GlobalCodeCompletionAllocator> Allocator)
      : AllocatorRef(std::move(Allocator)) {}

  IntrusiveRefCntPtr<GlobalCodeCompletionAllocator> getAllocatorRef() const {
    return AllocatorRef;
  }
  CodeCompletionAllocator &getAllocator() const {
    assert(AllocatorRef);
    return *AllocatorRef;
  }

  StringRef getParentName(const DeclContext *DC);
};

} // end namespace clang

namespace llvm {
  template <> struct isPodLike<clang::CodeCompletionString::Chunk> {
    static const bool value = true;
  };
}

namespace clang {

/// \brief A builder class used to construct new code-completion strings.
class CodeCompletionBuilder {
public:
  typedef CodeCompletionString::Chunk Chunk;

private:
  CodeCompletionAllocator &Allocator;
  CodeCompletionTUInfo &CCTUInfo;
  unsigned Priority;
  CXAvailabilityKind Availability;
  StringRef ParentName;
  const char *BriefComment;
  
  /// \brief The chunks stored in this string.
  SmallVector<Chunk, 4> Chunks;

  SmallVector<const char *, 2> Annotations;

public:
  CodeCompletionBuilder(CodeCompletionAllocator &Allocator,
                        CodeCompletionTUInfo &CCTUInfo)
    : Allocator(Allocator), CCTUInfo(CCTUInfo),
      Priority(0), Availability(CXAvailability_Available),
      BriefComment(nullptr) { }

  CodeCompletionBuilder(CodeCompletionAllocator &Allocator,
                        CodeCompletionTUInfo &CCTUInfo,
                        unsigned Priority, CXAvailabilityKind Availability)
    : Allocator(Allocator), CCTUInfo(CCTUInfo),
      Priority(Priority), Availability(Availability),
      BriefComment(nullptr) { }

  /// \brief Retrieve the allocator into which the code completion
  /// strings should be allocated.
  CodeCompletionAllocator &getAllocator() const { return Allocator; }

  CodeCompletionTUInfo &getCodeCompletionTUInfo() const { return CCTUInfo; }

  /// \brief Take the resulting completion string.
  ///
  /// This operation can only be performed once.
  CodeCompletionString *TakeString();

  /// \brief Add a new typed-text chunk.
  void AddTypedTextChunk(const char *Text);

  /// \brief Add a new text chunk.
  void AddTextChunk(const char *Text);

  /// \brief Add a new optional chunk.
  void AddOptionalChunk(CodeCompletionString *Optional);

  /// \brief Add a new placeholder chunk.
  void AddPlaceholderChunk(const char *Placeholder);

  /// \brief Add a new informative chunk.
  void AddInformativeChunk(const char *Text);

  /// \brief Add a new result-type chunk.
  void AddResultTypeChunk(const char *ResultType);

  /// \brief Add a new current-parameter chunk.
  void AddCurrentParameterChunk(const char *CurrentParameter);

  /// \brief Add a new chunk.
  void AddChunk(CodeCompletionString::ChunkKind CK, const char *Text = "");

  void AddAnnotation(const char *A) { Annotations.push_back(A); }

  /// \brief Add the parent context information to this code completion.
  void addParentContext(const DeclContext *DC);

  const char *getBriefComment() const { return BriefComment; }
  void addBriefComment(StringRef Comment);
  
  StringRef getParentName() const { return ParentName; }
};

/// \brief Captures a result of code completion.
class CodeCompletionResult {
public:
  /// \brief Describes the kind of result generated.
  enum ResultKind {
    RK_Declaration = 0, ///< Refers to a declaration
    RK_Keyword,         ///< Refers to a keyword or symbol.
    RK_Macro,           ///< Refers to a macro
    RK_Pattern          ///< Refers to a precomputed pattern.
  };

  /// \brief When Kind == RK_Declaration or RK_Pattern, the declaration we are
  /// referring to. In the latter case, the declaration might be NULL.
  const NamedDecl *Declaration;

  union {
    /// \brief When Kind == RK_Keyword, the string representing the keyword
    /// or symbol's spelling.
    const char *Keyword;

    /// \brief When Kind == RK_Pattern, the code-completion string that
    /// describes the completion text to insert.
    CodeCompletionString *Pattern;

    /// \brief When Kind == RK_Macro, the identifier that refers to a macro.
    const IdentifierInfo *Macro;
  };

  /// \brief The priority of this particular code-completion result.
  unsigned Priority;

  /// \brief Specifies which parameter (of a function, Objective-C method,
  /// macro, etc.) we should start with when formatting the result.
  unsigned StartParameter;

  /// \brief The kind of result stored here.
  ResultKind Kind;

  /// \brief The cursor kind that describes this result.
  CXCursorKind CursorKind;

  /// \brief The availability of this result.
  CXAvailabilityKind Availability;

  /// \brief Whether this result is hidden by another name.
  bool Hidden : 1;

  /// \brief Whether this result was found via lookup into a base class.
  bool QualifierIsInformative : 1;

  /// \brief Whether this declaration is the beginning of a
  /// nested-name-specifier and, therefore, should be followed by '::'.
  bool StartsNestedNameSpecifier : 1;

  /// \brief Whether all parameters (of a function, Objective-C
  /// method, etc.) should be considered "informative".
  bool AllParametersAreInformative : 1;

  /// \brief Whether we're completing a declaration of the given entity,
  /// rather than a use of that entity.
  bool DeclaringEntity : 1;

  /// \brief If the result should have a nested-name-specifier, this is it.
  /// When \c QualifierIsInformative, the nested-name-specifier is
  /// informative rather than required.
  NestedNameSpecifier *Qualifier;

  /// \brief Build a result that refers to a declaration.
  CodeCompletionResult(const NamedDecl *Declaration,
                       unsigned Priority,
                       NestedNameSpecifier *Qualifier = nullptr,
                       bool QualifierIsInformative = false,
                       bool Accessible = true)
    : Declaration(Declaration), Priority(Priority),
      StartParameter(0), Kind(RK_Declaration),
      Availability(CXAvailability_Available), Hidden(false),
      QualifierIsInformative(QualifierIsInformative),
      StartsNestedNameSpecifier(false), AllParametersAreInformative(false),
      DeclaringEntity(false), Qualifier(Qualifier) {
    computeCursorKindAndAvailability(Accessible);
  }

  /// \brief Build a result that refers to a keyword or symbol.
  CodeCompletionResult(const char *Keyword, unsigned Priority = CCP_Keyword)
    : Declaration(nullptr), Keyword(Keyword), Priority(Priority),
      StartParameter(0), Kind(RK_Keyword), CursorKind(CXCursor_NotImplemented),
      Availability(CXAvailability_Available), Hidden(false),
      QualifierIsInformative(0), StartsNestedNameSpecifier(false),
      AllParametersAreInformative(false), DeclaringEntity(false),
      Qualifier(nullptr) {}

  /// \brief Build a result that refers to a macro.
  CodeCompletionResult(const IdentifierInfo *Macro,
                       unsigned Priority = CCP_Macro)
    : Declaration(nullptr), Macro(Macro), Priority(Priority), StartParameter(0),
      Kind(RK_Macro), CursorKind(CXCursor_MacroDefinition),
      Availability(CXAvailability_Available), Hidden(false),
      QualifierIsInformative(0), StartsNestedNameSpecifier(false),
      AllParametersAreInformative(false), DeclaringEntity(false),
      Qualifier(nullptr) {}

  /// \brief Build a result that refers to a pattern.
  CodeCompletionResult(CodeCompletionString *Pattern,
                       unsigned Priority = CCP_CodePattern,
                       CXCursorKind CursorKind = CXCursor_NotImplemented,
                   CXAvailabilityKind Availability = CXAvailability_Available,
                       const NamedDecl *D = nullptr)
    : Declaration(D), Pattern(Pattern), Priority(Priority), StartParameter(0),
      Kind(RK_Pattern), CursorKind(CursorKind), Availability(Availability),
      Hidden(false), QualifierIsInformative(0),
      StartsNestedNameSpecifier(false), AllParametersAreInformative(false),
      DeclaringEntity(false), Qualifier(nullptr)
  {
  }

  /// \brief Build a result that refers to a pattern with an associated
  /// declaration.
  CodeCompletionResult(CodeCompletionString *Pattern, NamedDecl *D,
                       unsigned Priority)
    : Declaration(D), Pattern(Pattern), Priority(Priority), StartParameter(0),
      Kind(RK_Pattern), Availability(CXAvailability_Available), Hidden(false),
      QualifierIsInformative(false), StartsNestedNameSpecifier(false),
      AllParametersAreInformative(false), DeclaringEntity(false),
      Qualifier(nullptr) {
    computeCursorKindAndAvailability();
  }  
  
  /// \brief Retrieve the declaration stored in this result.
  const NamedDecl *getDeclaration() const {
    assert(Kind == RK_Declaration && "Not a declaration result");
    return Declaration;
  }

  /// \brief Retrieve the keyword stored in this result.
  const char *getKeyword() const {
    assert(Kind == RK_Keyword && "Not a keyword result");
    return Keyword;
  }

  /// \brief Create a new code-completion string that describes how to insert
  /// this result into a program.
  ///
  /// \param S The semantic analysis that created the result.
  ///
  /// \param Allocator The allocator that will be used to allocate the
  /// string itself.
  CodeCompletionString *CreateCodeCompletionString(Sema &S,
                                         const CodeCompletionContext &CCContext,
                                           CodeCompletionAllocator &Allocator,
                                           CodeCompletionTUInfo &CCTUInfo,
                                           bool IncludeBriefComments);
  CodeCompletionString *CreateCodeCompletionString(ASTContext &Ctx,
                                                   Preprocessor &PP,
                                         const CodeCompletionContext &CCContext,
                                           CodeCompletionAllocator &Allocator,
                                           CodeCompletionTUInfo &CCTUInfo,
                                           bool IncludeBriefComments);

private:
  void computeCursorKindAndAvailability(bool Accessible = true);
};

bool operator<(const CodeCompletionResult &X, const CodeCompletionResult &Y);

inline bool operator>(const CodeCompletionResult &X,
                      const CodeCompletionResult &Y) {
  return Y < X;
}

inline bool operator<=(const CodeCompletionResult &X,
                      const CodeCompletionResult &Y) {
  return !(Y < X);
}

inline bool operator>=(const CodeCompletionResult &X,
                       const CodeCompletionResult &Y) {
  return !(X < Y);
}


raw_ostream &operator<<(raw_ostream &OS,
                              const CodeCompletionString &CCS);

/// \brief Abstract interface for a consumer of code-completion
/// information.
class CodeCompleteConsumer {
protected:
  const CodeCompleteOptions CodeCompleteOpts;

  /// \brief Whether the output format for the code-completion consumer is
  /// binary.
  bool OutputIsBinary;

public:
  class OverloadCandidate {
  public:
    /// \brief Describes the type of overload candidate.
    enum CandidateKind {
      /// \brief The candidate is a function declaration.
      CK_Function,
      /// \brief The candidate is a function template.
      CK_FunctionTemplate,
      /// \brief The "candidate" is actually a variable, expression, or block
      /// for which we only have a function prototype.
      CK_FunctionType
    };

  private:
    /// \brief The kind of overload candidate.
    CandidateKind Kind;

    union {
      /// \brief The function overload candidate, available when
      /// Kind == CK_Function.
      FunctionDecl *Function;

      /// \brief The function template overload candidate, available when
      /// Kind == CK_FunctionTemplate.
      FunctionTemplateDecl *FunctionTemplate;

      /// \brief The function type that describes the entity being called,
      /// when Kind == CK_FunctionType.
      const FunctionType *Type;
    };

  public:
    OverloadCandidate(FunctionDecl *Function)
      : Kind(CK_Function), Function(Function) { }

    OverloadCandidate(FunctionTemplateDecl *FunctionTemplateDecl)
      : Kind(CK_FunctionTemplate), FunctionTemplate(FunctionTemplateDecl) { }

    OverloadCandidate(const FunctionType *Type)
      : Kind(CK_FunctionType), Type(Type) { }

    /// \brief Determine the kind of overload candidate.
    CandidateKind getKind() const { return Kind; }

    /// \brief Retrieve the function overload candidate or the templated
    /// function declaration for a function template.
    FunctionDecl *getFunction() const;

    /// \brief Retrieve the function template overload candidate.
    FunctionTemplateDecl *getFunctionTemplate() const {
      assert(getKind() == CK_FunctionTemplate && "Not a function template");
      return FunctionTemplate;
    }

    /// \brief Retrieve the function type of the entity, regardless of how the
    /// function is stored.
    const FunctionType *getFunctionType() const;

    /// \brief Create a new code-completion string that describes the function
    /// signature of this overload candidate.
    CodeCompletionString *CreateSignatureString(unsigned CurrentArg,
                                                Sema &S,
                                      CodeCompletionAllocator &Allocator,
                                      CodeCompletionTUInfo &CCTUInfo,
                                      bool IncludeBriefComments) const;
  };

  CodeCompleteConsumer(const CodeCompleteOptions &CodeCompleteOpts,
                       bool OutputIsBinary)
    : CodeCompleteOpts(CodeCompleteOpts), OutputIsBinary(OutputIsBinary)
  { }

  /// \brief Whether the code-completion consumer wants to see macros.
  bool includeMacros() const {
    return CodeCompleteOpts.IncludeMacros;
  }

  /// \brief Whether the code-completion consumer wants to see code patterns.
  bool includeCodePatterns() const {
    return CodeCompleteOpts.IncludeCodePatterns;
  }

  /// \brief Whether to include global (top-level) declaration results.
  bool includeGlobals() const {
    return CodeCompleteOpts.IncludeGlobals;
  }

  /// \brief Whether to include brief documentation comments within the set of
  /// code completions returned.
  bool includeBriefComments() const {
    return CodeCompleteOpts.IncludeBriefComments;
  }

  /// \brief Determine whether the output of this consumer is binary.
  bool isOutputBinary() const { return OutputIsBinary; }

  /// \brief Deregisters and destroys this code-completion consumer.
  virtual ~CodeCompleteConsumer();

  /// \name Code-completion callbacks
  //@{
  /// \brief Process the finalized code-completion results.
  virtual void ProcessCodeCompleteResults(Sema &S,
                                          CodeCompletionContext Context,
                                          CodeCompletionResult *Results,
                                          unsigned NumResults) { }

  /// \param S the semantic-analyzer object for which code-completion is being
  /// done.
  ///
  /// \param CurrentArg the index of the current argument.
  ///
  /// \param Candidates an array of overload candidates.
  ///
  /// \param NumCandidates the number of overload candidates
  virtual void ProcessOverloadCandidates(Sema &S, unsigned CurrentArg,
                                         OverloadCandidate *Candidates,
                                         unsigned NumCandidates) { }
  //@}

  /// \brief Retrieve the allocator that will be used to allocate
  /// code completion strings.
  virtual CodeCompletionAllocator &getAllocator() = 0;

  virtual CodeCompletionTUInfo &getCodeCompletionTUInfo() = 0;
};

/// \brief A simple code-completion consumer that prints the results it
/// receives in a simple format.
class PrintingCodeCompleteConsumer : public CodeCompleteConsumer {
  /// \brief The raw output stream.
  raw_ostream &OS;

  CodeCompletionTUInfo CCTUInfo;

public:
  /// \brief Create a new printing code-completion consumer that prints its
  /// results to the given raw output stream.
  PrintingCodeCompleteConsumer(const CodeCompleteOptions &CodeCompleteOpts,
                               raw_ostream &OS)
    : CodeCompleteConsumer(CodeCompleteOpts, false), OS(OS),
      CCTUInfo(new GlobalCodeCompletionAllocator) {}

  /// \brief Prints the finalized code-completion results.
  void ProcessCodeCompleteResults(Sema &S, CodeCompletionContext Context,
                                  CodeCompletionResult *Results,
                                  unsigned NumResults) override;

  void ProcessOverloadCandidates(Sema &S, unsigned CurrentArg,
                                 OverloadCandidate *Candidates,
                                 unsigned NumCandidates) override;

  CodeCompletionAllocator &getAllocator() override {
    return CCTUInfo.getAllocator();
  }

  CodeCompletionTUInfo &getCodeCompletionTUInfo() override { return CCTUInfo; }
};

} // end namespace clang

#endif // LLVM_CLANG_SEMA_CODECOMPLETECONSUMER_H
