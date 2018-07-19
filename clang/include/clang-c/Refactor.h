/*==-- clang-c/Refactor.h - Refactoring Public C Interface --------*- C -*-===*\
|*                                                                            *|
|*                     The LLVM Compiler Infrastructure                       *|
|*                                                                            *|
|* This file is distributed under the University of Illinois Open Source      *|
|* License. See LICENSE.TXT for details.                                      *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This header provides a public inferface to a Clang library for performing  *|
|* refactoring actions on projects without exposing the full Clang C++ API.   *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef LLVM_CLANG_C_REFACTOR_H
#define LLVM_CLANG_C_REFACTOR_H

#include "clang-c/Index.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup CINDEX_REFACTOR Refactoring options.
 *
 * @{
 */

/**
 * \brief The refactoring options that can be specified for each refactoring
 * action.
 */
enum CXRefactoringOption {
  /**
   * \brief The refactoring actions like 'rename' will avoid looking for
   * occurrences of the renamed symbol in comments if this option is enabled.
   */
  CXRefactorOption_AvoidTextualMatches = 1
};

/**
 * \brief Opaque pointer representing a set of options that can be given to
 * a refactoring action.
 */
typedef void *CXRefactoringOptionSet;

/**
 * \brief Returns a new option set.
 */
CINDEX_LINKAGE
CXRefactoringOptionSet clang_RefactoringOptionSet_create(void);

/**
 * \brief Parses and returns a new option set or NULL if the given string is
 * invalid.
 */
CINDEX_LINKAGE
CXRefactoringOptionSet
clang_RefactoringOptionSet_createFromString(const char *String);

/**
 * \brief Adds a new option to the given refactoring option set.
 */
CINDEX_LINKAGE
void clang_RefactoringOptionSet_add(CXRefactoringOptionSet Set,
                                    enum CXRefactoringOption Option);

/**
 * \brief Converts the given refactoring option set to a string value.
 */
CINDEX_LINKAGE
CXString clang_RefactoringOptionSet_toString(CXRefactoringOptionSet Set);

/**
 * \brief Free the given option set.
 *
 * Option sets should be freed by this function only when they were created
 * using the \c clang_RefactoringOptionSet_create* methods.
 */
CINDEX_LINKAGE
void clang_RefactoringOptionSet_dispose(CXRefactoringOptionSet Set);

/**
 * @}
 */

/**
 * \defgroup CINDEX_REFACTOR Refactoring actions.
 *
 * @{
 */

/**
 * \brief The refactoring actions that can be performed by libclang.
 */
enum CXRefactoringActionType {
  /**
   * \brief The 'rename' refactoring action.
   */
  CXRefactor_Rename = 0,

  /**
   * \brief The local 'rename' refactoring action.
   */
  CXRefactor_Rename_Local = 1,

  /**
   * \brief The 'extract' refactoring action extracts source code into a
   * new function.
   */
  CXRefactor_Extract = 2,

  /**
   * \brief The sub-action of 'extract' that extracts source code into a new
   * method.
   */
  CXRefactor_Extract_Method = 3,

  /**
  * \brief The action that converts an if/else constructs to a switch block.
  */
  CXRefactor_IfSwitchConversion = 4,

  /**
  * \brief The action that wraps an Objective-C string literal in an
  * NSLocalizedString macro.
  */
  CXRefactor_LocalizeObjCStringLiteral = 5,

  /**
  * \brief The action that adds missing switch cases to an switch over an enum.
  */
  CXRefactor_FillInEnumSwitchCases = 6,

  /**
  * \brief The action that adds missing protocol methods to an Objective-C
  * class.
  */
  CXRefactor_FillInMissingProtocolStubs = 7,

  /**
  * \brief The action that extracts an expression that's repeated in a function
  * into a new variable.
  */
  CXRefactor_ExtractRepeatedExpressionIntoVariable = 8,

  /**
  * \brief The action that adds missing abstract class method overrides to a
  * class.
  */
  CXRefactor_FillInMissingMethodStubsFromAbstractClasses = 9,

  /**
  * \brief The action that generates dummy method definitions for method
  * declarations without respective definitions.
  */
  CXRefactor_ImplementDeclaredMethods = 10,

  /**
   * \brief The sub-action of 'extract' that extracts source expression into a
   * new variable.
   */
  CXRefactor_Extract_Expression = 11,
};

/**
 * \brief Return the name of the given refactoring action.
 */
CINDEX_LINKAGE
CXString
clang_RefactoringActionType_getName(enum CXRefactoringActionType Action);

/**
 * \brief A set of refactoring actions that can be performed at some specific
 * location in a source file.
 *
 * The actions in the action set are ordered by their priority: most important
 * actions are placed before the less important ones.
 */
typedef struct {
  const enum CXRefactoringActionType *Actions;
  unsigned NumActions;
} CXRefactoringActionSet;

/**
 * \brief Free the given refactoring action set.
 */
CINDEX_LINKAGE void
clang_RefactoringActionSet_dispose(CXRefactoringActionSet *Set);

typedef struct {
  enum CXRefactoringActionType Action;
  /**
   * \brief The set of diagnostics that describes the reason why this action
   * couldn't be initiated. This set of diagnostics is managed by the
   * \c CXRefactoringActionSetWithDiagnostics and shouldn't be freed manually.
   */
  CXDiagnosticSet Diagnostics;
} CXRefactoringActionWithDiagnostics;

/**
 * \brief A set of refactoring actions that couldn't be initiated at some
 * location and their respective diagnostics that describe the reason why
 * the initiation failed.
 */
typedef struct {
  CXRefactoringActionWithDiagnostics *Actions;
  unsigned NumActions;
} CXRefactoringActionSetWithDiagnostics;

/**
 * \brief Free the given refactoring action set with diagnostics.
 */
CINDEX_LINKAGE void clang_RefactoringActionSetWithDiagnostics_dispose(
    CXRefactoringActionSetWithDiagnostics *Set);

/**
 * \brief Find the set of refactoring actions that can be performed at the given
 * location.
 *
 * This function examines the AST around the given source range and creates a
 * \c CXRefactoringActionSet that contains all of the actions that can be
 * performed in the given source range.
 *
 * \param TU The translation unit which contains the given source range.
 *
 * \param Location The location at which the refactoring action will be
 * performed.
 *
 * \param SelectionRange The range in which the AST should be checked. Usually
 * corresponds to the selection range or location of the cursor in the editor.
 * Can be a null range.
 *
 * \param Options The optional refactoring options that might influence the way
 * the search is performed.
 *
 * \param[out] OutSet A non-NULL pointer to store the created
 * \c CXRefactoringActionSet.
 *
 * \returns Zero on success, CXError_RefactoringActionUnavailable when
 * there are no actions available in the given range, or an error code
 * otherwise.
 */
CINDEX_LINKAGE
enum CXErrorCode
clang_Refactoring_findActionsAt(CXTranslationUnit TU, CXSourceLocation Location,
                                CXSourceRange SelectionRange,
                                CXRefactoringOptionSet Options,
                                CXRefactoringActionSet *OutSet);

/**
 * \brief Find the set of refactoring actions that can be performed at the given
 * location.
 *
 * This function examines the AST around the given source range and creates a
 * \c CXRefactoringActionSet that contains all of the actions that can be
 * performed in the given source range. It also creates a
 * \c CXRefactoringActionSetWithDiagnostics that might describe the reason why
 * some refactoring actions are not be available.
 *
 * \param TU The translation unit which contains the given source range.
 *
 * \param Location The location at which the refactoring action will be
 * performed.
 *
 * \param SelectionRange The range in which the AST should be checked. Usually
 * corresponds to the selection range or location of the cursor in the editor.
 * Can be a null range.
 *
 * \param Options The optional refactoring options that might influence the way
 * the search is performed.
 *
 * \param[out] OutSet A non-NULL pointer to store the created
 * \c CXRefactoringActionSet.
 *
 * \param[out] OutFailureSet An optional pointer to store the created
 * \c CXRefactoringActionSetWithDiagnostics that describes the failures reasons
 * for some of the refactoring actions.
 *
 * \returns Zero on success, CXError_RefactoringActionUnavailable when
 * there are no actions available in the given range, or an error code
 * otherwise.
 */
CINDEX_LINKAGE
enum CXErrorCode clang_Refactoring_findActionsWithInitiationFailureDiagnosicsAt(
    CXTranslationUnit TU, CXSourceLocation Location,
    CXSourceRange SelectionRange, CXRefactoringOptionSet Options,
    CXRefactoringActionSet *OutSet,
    CXRefactoringActionSetWithDiagnostics *OutFailureSet);

/**
 * @}
 */

/**
 * \defgroup CINDEX_REFACTOR_INITIATE Refactoring initiation
 *
 * @{
 */

/**
 * \brief Opaque pointer representing the initiated refactoring action.
 */
typedef void *CXRefactoringAction;

/**
 * \brief Free the given refactoring action.
 *
 * The refactoring action should be freed before the initiation and/or
 * implementation translation units.
 */
CINDEX_LINKAGE void clang_RefactoringAction_dispose(CXRefactoringAction Action);

/**
 * \brief Return the source range that's associated with the initiated
 * refactoring action.
 *
 * The returned source range covers the source that will be modified by the
 * given refactoring action. If the action has no associated source range,
 * then this function will return a null \c CXSourceRange.
 */
CINDEX_LINKAGE CXSourceRange
clang_RefactoringAction_getSourceRangeOfInterest(CXRefactoringAction Action);

/**
 * \brief Return the type of the initiated action, which might be different
 * to the type of the requested action. For an operation 'rename', the action
 * could actually initiate the local 'rename' operation.
 */
CINDEX_LINKAGE
enum CXRefactoringActionType
clang_RefactoringAction_getInitiatedActionType(CXRefactoringAction Action);

/**
 * \brief Return a non-zero value when the refactoring action requires access
 * to an additional translation unit that contains an implementation of some
 * declaration.
 */
// TODO: Remove (this is no longer needed due to refactoring continuations).
CINDEX_LINKAGE
int clang_RefactoringAction_requiresImplementationTU(
    CXRefactoringAction Action);

/**
 * \brief Return a USR that corresponds to the declaration whose implementation
 * is required in order for the given refactoring action to work correctly.
 */
// TODO: Remove (this is no longer needed due to refactoring continuations).
CINDEX_LINKAGE
CXString clang_RefactoringAction_getUSRThatRequiresImplementationTU(
    CXRefactoringAction Action);

/**
 * \brief Set the translation unit that contains the declaration whose
 * implementation is required for the given refactoring action to work
 * correctly.
 */
// TODO: Remove (this is no longer needed due to refactoring continuations).
CINDEX_LINKAGE
enum CXErrorCode
clang_RefactoringAction_addImplementationTU(CXRefactoringAction Action,
                                            CXTranslationUnit TU);

/**
 * \brief A refactoring candidate determines on which piece of source code the
 * action should be applied.
 *
 * Most refactoring actions have just one candidate, but some actions, like
 * 'Extract' can produce multiple candidates.
 *
 * The candidates are managed by the refactoring action, and their description
 * string doesn't need to be freed manually.
 */
typedef struct { CXString Description; } CXRefactoringCandidate;

/**
 * \brief A set of refactoring candidates on which the previously initiatied
 * refactoring action can be performed.
 *
 * The candidates in the candidate set are ordered by their priority: the
 * ones that are more likely to be selected are placed before the other ones.
 *
 * A non-empty refactoring candidate set always has more than one refactoring
 * candidate, because when a refactoring action has just one candidate,
 * \c clang_RefactoringAction_getRefactoringCandidates will return an empty
 * candidate set.
 */
typedef struct {
  const CXRefactoringCandidate *Candidates;
  unsigned NumCandidates;
} CXRefactoringCandidateSet;

/**
 * \brief Returns the given action's refactoring candidates.
 *
 * The resulting refactoring candidate set will be empty when the given \c
 * CXRefactoringAction has just one refactoring candidate.
 *
 * \param Action A previously initiated \c CXRefactoringAction.
 *
 * \param[out] OutRefactoringCandidateSet An pointer to store the action's
 * refactoring candidate set.
 *
 * \returns Zero on success, or an error code otherwise.
 */
CINDEX_LINKAGE
enum CXErrorCode clang_RefactoringAction_getRefactoringCandidates(
    CXRefactoringAction Action,
    CXRefactoringCandidateSet *OutRefactoringCandidateSet);

/**
 * \brief Tells the given refactoring action that it has to perform the
 * operation on the refactoring candidate that's located at \p Index in the \c
 * CXRefactoringCandidateSet.
 */
CINDEX_LINKAGE
enum CXErrorCode
clang_RefactoringAction_selectRefactoringCandidate(CXRefactoringAction Action,
                                                   unsigned Index);

// TODO: Remove.
CINDEX_LINKAGE
enum CXErrorCode clang_Refactoring_initiateActionAt(
    CXTranslationUnit TU, CXSourceLocation Location,
    CXSourceRange SelectionRange, enum CXRefactoringActionType ActionType,
    CXRefactoringOptionSet Options, CXRefactoringAction *OutAction,
    CXString *OutFailureReason);

/**
 * \brief Initiate a specific refactoring action at the given location.
 *
 * This function initiates an \p ActionType refactoring action when it can
 * be initiated at the given location and creates a \c CXRefactoringAction
 * action that will allow the control.
 *
 * \param TU The translation unit in which the action should be initiated.
 *
 * \param Location The location at which the refactoring action will be
 * performed.
 *
 * \param SelectionRange The range in which the AST should be checked. Usually
 * corresponds to the selection range or location of the cursor in the editor.
 * Can be a null range.
 *
 * \param ActionType The type of action that should be initiated.
 *
 * \param Options The optional refactoring options that might have an influence
 * on the initiation process.
 *
 * \param[out] OutAction A non-NULL pointer to store the created
 * \c CXRefactoringAction.
 *
 * \param[out] OutDiagnostics An optional pointer to store any diagnostics that
 * describe why the action wasn't initiated.
 *
 * \returns Zero on success, CXError_RefactoringActionUnavailable when
 * the given refactoring action can't be performed at the given location, or an
 * error code otherwise.
 */
CINDEX_LINKAGE
enum CXErrorCode clang_Refactoring_initiateAction(
    CXTranslationUnit TU, CXSourceLocation Location,
    CXSourceRange SelectionRange, enum CXRefactoringActionType ActionType,
    CXRefactoringOptionSet Options, CXRefactoringAction *OutAction,
    CXDiagnosticSet *OutDiagnostics);

/**
 * \brief Initiate a specific refactoring action on a particular declaration.
 *
 * This function searches for the declaration that corresponds to \p DeclUSR
 * and initiates an \p ActionType a refactoring action on that declaration
 * if possible.
 *
 * \param TU The translation unit in which the declaration is defined.
 *
 * \param DeclUSR The USR that corresponds to the declaration of interest.
 *
 * \param ActionType The type of action that should be initiated.
 *
 * \param Options The optional refactoring options that might have an influence
 * on the initiation process.
 *
 * \param[out] OutAction A non-NULL pointer to store the created
 * \c CXRefactoringAction.
 *
 * \returns Zero on success, CXError_RefactoringActionUnavailable when
 * the given refactoring action can't be performed on the found declaration, or
 * an error code otherwise.
 */
// TODO: Remove (not needed).
CINDEX_LINKAGE
enum CXErrorCode clang_Refactoring_initiateActionOnDecl(
    CXTranslationUnit TU, const char *DeclUSR,
    enum CXRefactoringActionType ActionType, CXRefactoringOptionSet Options,
    CXRefactoringAction *OutAction, CXString *OutFailureReason);

/**
 * @}
 */

/**
 * \defgroup CINDEX_REFACTOR_REPLACEMENT Refactoring replacement
 *
 * @{
 */

/**
 * \brief A source location in a single file that is independent of \c
 * CXTranslationUnit.
 */
typedef struct { unsigned Line, Column; } CXFileLocation;

/**
 * \brief A source range in a single file that is independent of \c
 * CXTranslationUnit.
 */
typedef struct { CXFileLocation Begin, End; } CXFileRange;

// TODO: Remove
typedef struct {
  CXFileRange Range;
  CXString ReplacementString;
} CXRefactoringReplacement_Old;

// TODO: Remove
typedef struct {
  CXString Filename;
  const CXRefactoringReplacement_Old *Replacements;
  unsigned NumReplacements;
} CXRefactoringFileReplacementSet_Old;

// TODO: Remove
typedef struct {
  const CXRefactoringFileReplacementSet_Old *FileReplacementSets;
  unsigned NumFileReplacementSets;
} CXRefactoringReplacements_Old;

/**
 * \brief Identifies a character range in the source code of a single file that
 * should be replaced with the replacement string.
 *
 * Replacements are managed by the result of a specific refactoring action,
 * like \c CXRenamingResult, and are invalidated when the refactoring result is
 * destroyed.
 */
typedef struct {
  CXFileRange Range;
  CXString ReplacementString;
  void *AssociatedData;
} CXRefactoringReplacement;

/**
* \brief A set of refactoring replacements that are applicable to a certain
 * file.
 */
typedef struct {
  CXString Filename;
  const CXRefactoringReplacement *Replacements;
  unsigned NumReplacements;
} CXRefactoringFileReplacementSet;

/**
 * \brief A set of refactoring replacements that have been produced by a
 * refactoring operation.
 *
 * The refactoring replacements depend on \c CXRefactoringResult, and can't be
 * used after the refactoring result is freed.
 */
typedef struct {
  const CXRefactoringFileReplacementSet *FileReplacementSets;
  unsigned NumFileReplacementSets;
} CXRefactoringReplacements;

/**
 * @}
 */

/**
 * \defgroup CINDEX_SYMBOL_OPERATION Symbol-based refactoring operation
 * (e.g. Rename).
 *
 * @{
 */

/**
 * \brief The type of a symbol occurrence.
 *
 * The occurrence kind determines if an occurrence can be renamed automatically
 * or if the user has to make the decision whether or not this occurrence
 * should be renamed.
 */
enum CXSymbolOccurrenceKind {
  /**
   * \brief This occurrence is an exact match and can be renamed automatically.
   */
  CXSymbolOccurrence_MatchingSymbol = 0,

  /**
  * \brief This is an occurrence of a matching selector. It can't be renamed
  * automatically unless the indexer proves that this selector refers only
  * to the declarations that correspond to the renamed symbol.
  */
  CXSymbolOccurrence_MatchingSelector = 1,

  /**
  * \brief This is an occurrence of an implicit property that uses the
  * renamed method.
  */
  CXSymbolOccurrence_MatchingImplicitProperty = 2,

  /**
  * \brief This is an occurrence of an symbol name in a comment.
  */
  CXSymbolOccurrence_MatchingCommentString = 3,

  /**
  * \brief This is an occurrence of an symbol name in a documentation comment.
  */
  CXSymbolOccurrence_MatchingDocCommentString = 4,

  /**
  * \brief This is an occurrence of an symbol name in a filename in an inclusion
  * directive.
  */
  CXSymbolOccurrence_MatchingFilename = 5,

  /**
  * \brief This is an occurrence of an symbol name in a string literal.
  */
  CXSymbolOccurrence_MatchingStringLiteral = 6,

  /**
  * \brief This is an occurrence of a symbol name that belongs to the extracted
  * declaration. Note: this occurrence can be in two replacements as we might
  * extract an out-of-line method that will be both declared and defined.
  */
  CXSymbolOccurrence_ExtractedDeclaration = 100,

  /**
  * \brief This is an occurrence of a symbol name that references the extracted
  * declaration.
  */
  CXSymbolOccurrence_ExtractedDeclaration_Reference = 101,
};

// TODO: Remove
typedef struct {
  const CXRefactoringReplacement_Old *Replacements;
  unsigned ReplacementCount;
  enum CXSymbolOccurrenceKind Kind;
  /**
   * Whether or not this occurrence is inside a macro. When this is true, the
   * replacements of the occurrence contain just a single empty replacement that
   * points to the location of the macro expansion.
   */
  int IsMacroExpansion;
} CXRenamedSymbolOccurrence;

/**
 * \brief An occurrence of a symbol.
 *
 * Contains the source ranges that represent the pieces of the name of the
 * symbol. The occurrences are managed by \c CXRenamingResult, and are
 * invalidated when \c CXRenamingResult is destroyed.
 */
typedef struct {
  const CXFileRange *NamePieces;
  unsigned NumNamePieces;
  enum CXSymbolOccurrenceKind Kind;
  /**
   * Whether or not this occurrence is inside a macro. When this is true, the
   * replacements of the occurrence contain just a single empty replacement that
   * points to the location of the macro expansion.
   */
  int IsMacroExpansion;
  unsigned SymbolIndex;
} CXSymbolOccurrence;

// TODO: Remove
typedef struct {
  CXString Filename;
  const CXRenamedSymbolOccurrence *Occurrences;
  unsigned NumOccurrences;
} CXFileRenamingResult; // TODO: Remove

/**
* \brief A set of symbol occurrences that occur in a single file.
 */
typedef struct {
  CXString Filename;
  /**
   * The set of occurrences for each symbol of interest.
   */
  const CXSymbolOccurrence *Occurrences;
  unsigned NumOccurrences;
} CXSymbolOccurrencesInFile;

/**
 * \brief Opaque pointer representing all of the renames that should take place
 * in a single translation unit.
 *
 * The result of a renaming action is indepedent from \c CXRenamingAction, and
 * remains valid after \c CXRenamingAction is destroyed.
 */
typedef void *CXRenamingResult;

/**
 * \brief Opaque pointer representing all of the symbol occurrences from a
 * single TU/file.
 *
 * The result of a symbol search occurrence search operation is indepedent from
 * \c CXRefactoringAction, and remains valid after \c CXRefactoringAction is
 * destroyed.
 */
typedef void *CXSymbolOccurrencesResult;

/**
 * \brief Find the cursor that's being renamed at the given location.
 *
 * \param TU The translation unit in which the cursor is present.
 *
 * \param Location The location at which the refactoring action will be
 * performed.
 *
 * \param SelectionRange The range in which the AST should be checked. Usually
 * corresponds to the selection range or location of the cursor in the editor.
 * Can be a null range.
 *
 * \returns Zero on success, CXError_RefactoringActionUnavailable when
 * there's no suitable cursor at the given location, or an error code otherwise.
 */
CINDEX_LINKAGE
enum CXErrorCode clang_Refactoring_findRenamedCursor(
    CXTranslationUnit TU, CXSourceLocation Location,
    CXSourceRange SelectionRange, CXCursor *OutCursor);

/**
 * \brief Initiates a renaming operation on a previously initiated refactoring
 * action.
 *
 * The initiation process finds the symbols that have to be renamed for a
 * previously initiated \c CXRefactor_Rename refactoring action.
 *
 * \returns Zero on success, or an error code otherwise.
 */
// TODO: Remove
CINDEX_LINKAGE
enum CXErrorCode
clang_Refactoring_initiateRenamingOperation(CXRefactoringAction Action);

/**
 * \brief Set the new name of the renamed symbol in the given \c
 * RenamingAction.
 *
 * \returns Zero on success, CXError_RefactoringNameInvalid when the new name
 * isn't a valid identifier, CXError_RefactoringNameSizeMismatch when the new
 * name has an incorrect number of pieces or a different error code otherwise.
 */
// TODO: Remove
CINDEX_LINKAGE
enum CXErrorCode clang_RenamingOperation_setNewName(CXRefactoringAction Action,
                                                    const char *NewName);

/**
 * \brief Return the number of symbols that are renamed by the given renaming
 * action.
 *
 * A renaming action typically works on just one symbol. However, there are
 * certain language constructs that require work with more than one symbol in
 * order for them to be renamed correctly. Property declarations in Objective-C
 * are the perfect example: in addition to the actual property, the action has
 * to rename the corresponding getters and setters, as well as the backing ivar.
 */
// TODO: Remove
CINDEX_LINKAGE
unsigned clang_RenamingOperation_getNumSymbols(CXRefactoringAction Action);

/**
 * \brief Return the USR of the declaration that was found for the symbol at the
 * given \p Index in the given renaming action.
 */
// TODO: Remove
CINDEX_LINKAGE
CXString clang_RenamingOperation_getUSRForSymbol(CXRefactoringAction Action,
                                                 unsigned Index);

// TODO: Remove
CINDEX_LINKAGE
CXRenamingResult clang_Refactoring_findRenamedOccurrencesInPrimaryTUs(
    CXRefactoringAction Action, const char *const *CommandLineArgs,
    int NumCommandLineArgs, struct CXUnsavedFile *UnsavedFiles,
    unsigned NumUnsavedFiles);

/**
 * \brief Find all of the occurrences of the symbol that is being searched for
 * by the given refactoring action in the translation unit that was used to
 * initiate the refactoring action.
 *
 * This function searches for all of the \c CXSymbolOccurrence in the
 * translation units that are referenced by the given \c CXRefactoringAction by
 * iterating through the AST of the each translation unit. The occurrences that
 * are found don't have to be from the main file in the translation unit, they
 * can be from files included in that translation unit.
 *
 * \param Action The \c CXRefactoringAction operation that was inititated by
 * \c clang_Refactoring_initiateActionAt().
 *
 * \param CommandLineArgs The command-line arguments that would be
 * passed to the \c clang executable if it were being invoked out-of-process.
 *
 * \param NumCommandLineArgs The number of command-line arguments in
 * \c CommandLineArgs.
 *
 * \param UnsavedFiles the files that have not yet been saved to disk
 * but may be required for parsing, including the contents of
 * those files.  The contents and name of these files (as specified by
 * CXUnsavedFile) are copied when necessary, so the client only needs to
 * guarantee their validity until the call to this function returns.
 *
 * \param NumUnsavedFiles the number of unsaved file entries in \p
 * UnsavedFiles.
 *
 * \returns If successful, a new \c CXSymbolOccurrencesResult structure
 * containing the occurrences of the symbol in the initiation translation unit,
 * which should eventually be freed with \c clang_SymbolOccurrences_dispose().
 * If symbol search fails, returns NULL.
 */
CINDEX_LINKAGE
CXSymbolOccurrencesResult clang_Refactoring_findSymbolOccurrencesInInitiationTU(
    CXRefactoringAction Action, const char *const *CommandLineArgs,
    int NumCommandLineArgs, struct CXUnsavedFile *UnsavedFiles,
    unsigned NumUnsavedFiles);

// TODO: Remove
typedef struct {
  CXFileLocation Location;
  /**
   * The kind of the declaration/expression that was indexed at this location.
   * This is particularly important for Objective-C selectors. The refactoring
   * engine requires the following cursor kinds for the following indexed
   * occurrences:
   *   - ObjC method declaration:  CXCursor_ObjC(Instance/Class)MethodDecl
   *   - ObjC method message send: CXCursor_ObjCMessageExpr
   * Other occurrences can use any other cursor cursor kinds.
   */
  enum CXCursorKind CursorKind;
} CXRenamedIndexedSymbolLocation;

// TODO: Remove
typedef struct {
  /**
   * An array of occurrences that represent indexed occurrences of a symbol.
   * It's valid to pass-in no indexed locations, the refactoring engine will
   * just perform textual search in that case.
   */
  const CXRenamedIndexedSymbolLocation *IndexedLocations;
  unsigned IndexedLocationCount;
  /**
  * The kind of the declaration that is being renamed.
  * This is particularly important for Objective-C selectors. The refactoring
  * engine requires the following cursor kinds for the following renamed
  * declaration:
  *   - ObjC methods:  CXCursor_ObjC(Instance/Class)MethodDecl
  * Other declarations can use any other cursor cursor kinds.
  */
  enum CXCursorKind CursorKind;
  const char *Name;
  const char *NewName;
} CXRenamedIndexedSymbol;

// TODO: Remove
CINDEX_LINKAGE
enum CXErrorCode clang_Refactoring_findRenamedOccurrencesInIndexedFile(
    const CXRenamedIndexedSymbol *Symbols, unsigned NumSymbols, CXIndex CIdx,
    const char *Filename, const char *const *CommandLineArgs,
    int NumCommandLineArgs, struct CXUnsavedFile *UnsavedFiles,
    unsigned NumUnsavedFiles, CXRefactoringOptionSet Options,
    CXRenamingResult *OutResult);

/**
 * \brief A location of an already known occurrence of a symbol.
 *
 * Used for rename-indexed operation where the renaming is performed on an
 * already indexed source file.
 */
typedef struct {
  CXFileLocation Location;
  /**
   * The kind of the declaration/expression that was indexed at this location.
   * This is particularly important for Objective-C selectors. The refactoring
   * engine requires the following cursor kinds for the following indexed
   * occurrences:
   *   - ObjC method declaration:  CXCursor_ObjC(Instance/Class)MethodDecl
   *   - ObjC method message send: CXCursor_ObjCMessageExpr
   *   - filename in an #include: CXCursor_InclusionDirective
   * Other occurrences can use any other cursor cursor kinds.
   */
  enum CXCursorKind CursorKind;
} CXIndexedSymbolLocation;

/**
 * \brief A symbol that should be found the an indexer symbol search operation.
 *
 * Used for rename-indexed operation where the renaming is performed on an
 * already indexed source file.
 */
typedef struct {
  /**
   * An array of occurrences that represent indexed occurrences of a symbol.
   * It's valid to pass-in no indexed locations, the refactoring engine will
   * just perform textual search in that case.
   */
  const CXIndexedSymbolLocation *IndexedLocations;
  unsigned IndexedLocationCount;
  /**
   * The kind of the declaration that is being renamed.
   * This is particularly important for Objective-C selectors. The refactoring
   * engine requires the following cursor kinds for the following renamed
   * declaration:
   *   - ObjC methods:  CXCursor_ObjC(Instance/Class)MethodDecl
   *   - ObjC class:    CXCursor_ObjCInterfaceDecl
   * Other declarations can use any other cursor cursor kinds.
   */
  enum CXCursorKind CursorKind;
  /**
   * The name of the symbol. Objective-C selector names should be specified
   * using the ':' separator for selector pieces.
   */
  const char *Name;
} CXIndexedSymbol;

/**
 * \brief Find all of the occurrences of a symbol in an indexed file.
 *
 * This function searches for all of the \c CXIndexedSymbol in the
 * given file by inspecting the source code at the given indexed locations.
 *
 * The indexed operations are thread-safe and can be performed concurrently.
 *
 * \param Symbols The information about the symbols that includes the locations
 * for a symbol in the file as determined by the indexer.
 *
 * \param NumSymbols The number of symbols in \p Symbols.
 *
 * \param CIdx The index object with which the translation unit will be
 * associated.
 *
 * \param Filename The name of the source file that contains the given
 * \p Locations.
 *
 * \param CommandLineArgs The command-line arguments that would be
 * passed to the \c clang executable if it were being invoked out-of-process.
 * These command-line options will be parsed and will affect how the translation
 * unit is parsed.
 *
 * \param NumCommandLineArgs The number of command-line arguments in
 * \c CommandLineArgs.
 *
 * \param UnsavedFiles the files that have not yet been saved to disk
 * but may be required for parsing, including the contents of
 * those files.  The contents and name of these files (as specified by
 * CXUnsavedFile) are copied when necessary, so the client only needs to
 * guarantee their validity until the call to this function returns.
 *
 * \param NumUnsavedFiles the number of unsaved file entries in \p
 * UnsavedFiles.
 *
 * \param Options The optional refactoring options that might have an influence
 * on the initiation process.
 *
 * \param[out] OutResult A non-NULL pointer to store the created
 * \c CXSymbolOccurrencesResult.
 *
 * \returns Zero on success, or a different error code otherwise.
 */
CINDEX_LINKAGE
enum CXErrorCode clang_Refactoring_findSymbolOccurrencesInIndexedFile(
    const CXIndexedSymbol *Symbols, unsigned NumSymbols, CXIndex CIdx,
    const char *Filename, const char *const *CommandLineArgs,
    int NumCommandLineArgs, struct CXUnsavedFile *UnsavedFiles,
    unsigned NumUnsavedFiles, CXRefactoringOptionSet Options,
    CXSymbolOccurrencesResult *OutResult);

// TODO: Remove
CINDEX_LINKAGE
unsigned clang_RenamingResult_getNumModifiedFiles(CXRenamingResult Result);

// TODO: Remove
CINDEX_LINKAGE
void clang_RenamingResult_getResultForFile(CXRenamingResult Result,
                                           unsigned FileIndex,
                                           CXFileRenamingResult *OutResult);

// TODO: Remove
CINDEX_LINKAGE
void clang_RenamingResult_dispose(CXRenamingResult Result);

/**
 * \brief Return the number of files that have occurrences of the specific
 * symbol.
 */
CINDEX_LINKAGE
unsigned clang_SymbolOccurrences_getNumFiles(CXSymbolOccurrencesResult Result);

/**
 * \brief Return the set of symbol occurrences in a single file.
 *
 * The resulting \c CXSymbolOccurrencesInFile is managed by the
 * \c CXSymbolOccurrencesResult and doesn't have to be disposed of manually.
 */
CINDEX_LINKAGE
void clang_SymbolOccurrences_getOccurrencesForFile(
    CXSymbolOccurrencesResult Result, unsigned FileIndex,
    CXSymbolOccurrencesInFile *OutResult);

// TODO: Support refactoring continuations for \c CXSymbolOccurrencesResult,
// e.g. for function parameter name rename.

/**
 * \brief Free the given symbol occurrences result.
 */
CINDEX_LINKAGE
void clang_SymbolOccurrences_dispose(CXSymbolOccurrencesResult Result);

/**
 * @}
 */

/**
 * \defgroup CINDEX_REFACTOR_PERFORM Performing refactoring operations.
 *
 * @{
 */

/**
 * \brief Opaque pointer representing the results of the refactoring operation.
 *
 * The result of a refactoring action depends on the \c CXRefactoringAction, and
 * is invalidated after \c CXRefactoringAction is destroyed.
 */
typedef void *CXRefactoringResult;

/**
 * \brief Opaque pointer representing a refactoring continuation.
 *
 * Refactoring continuations allow refactoring operations to run in external
 * AST units with some results that were obtained after querying the indexer.
 *
 * The refactoring continuation is not dependent on the \c CXRefactoringAction
 * or \c CXRefactoringResult. It does depend on the initiation
 * \c CXTranslationUnit initially, but that dependency can be terminated.
 */
typedef void *CXRefactoringContinuation;

/**
 * \brief Opaque pointer representing a query to the indexer.
 */
typedef void *CXIndexerQuery;

/**
 * \brief Performs the previously initiated refactoring operation.
 *
 * This function executes the refactoring operation which produces a set of
 * candidate source replacements that can be applied to the source files.
 *
 * \param Action The refactoring action.
 *
 * \param CommandLineArgs The command-line arguments that would be
 * passed to the \c clang executable if it were being invoked out-of-process.
 * These command-line options will be parsed and will affect how the translation
 * unit is parsed.
 *
 * \param NumCommandLineArgs The number of command-line arguments in
 * \c CommandLineArgs.
 *
 * \param UnsavedFiles the files that have not yet been saved to disk
 * but may be required for parsing, including the contents of
 * those files.  The contents and name of these files (as specified by
 * CXUnsavedFile) are copied when necessary, so the client only needs to
 * guarantee their validity until the call to this function returns.
 *
 * \param NumUnsavedFiles the number of unsaved file entries in \p
 * UnsavedFiles.
 *
 * \param Options The optional refactoring options that might have an influence
 * on the way the particular action will be performed.
 *
 * \param[out] OutFailureReason An optional pointer to store a message that
 * describes why the action wasn't performed.
 *
 * \returns If successful, a new \c CXRefactoringResult structure containing the
 * source replacement candidates, which should eventually be freed with
 * \c clang_RefactoringResult_dispose(). If the refactoring operation fails,
 * returns NULL.
 */
CINDEX_LINKAGE
CXRefactoringResult clang_Refactoring_performOperation(
    CXRefactoringAction Action, const char *const *CommandLineArgs,
    int NumCommandLineArgs, struct CXUnsavedFile *UnsavedFiles,
    unsigned NumUnsavedFiles, CXRefactoringOptionSet Options,
    CXString *OutFailureReason);

// TODO: Remove. This is the deprecated API.
CINDEX_LINKAGE
void clang_RefactoringResult_getReplacements(
    CXRefactoringResult Result, CXRefactoringReplacements_Old *OutReplacements);

/**
 * \brief Return the set of refactoring source replacements.
 *
 * The resulting \c CXRefactoringReplacements are managed by the
 * \c CXRefactoringResult and don't have to be disposed of manually.
 */
CINDEX_LINKAGE
CXRefactoringReplacements
clang_RefactoringResult_getSourceReplacements(CXRefactoringResult Result);

/**
 * \brief Represents a set of symbol occurrences that are associated with a
 * single refactoring replacement.
 *
 * The symbol occurrences depend on \c CXRefactoringResult, and can't be
 * used after the refactoring result is freed.
 */
typedef struct {
  const CXSymbolOccurrence *AssociatedSymbolOccurrences;
  unsigned NumAssociatedSymbolOccurrences;
} CXRefactoringReplacementAssociatedSymbolOccurrences;

/**
 * \brief Return the set of symbol occurrences that are associated with the
 * given \p Replacement.
 */
CXRefactoringReplacementAssociatedSymbolOccurrences
clang_RefactoringReplacement_getAssociatedSymbolOccurrences(
    CXRefactoringReplacement Replacement);

/**
 * \brief Returns the refactoring continuation associated with this result, or
 * NULL if this result has no refactoring continuation.
 */
CINDEX_LINKAGE
CXRefactoringContinuation
clang_RefactoringResult_getContinuation(CXRefactoringResult Result);

/**
 * \brief Free the given refactoring result.
 */
CINDEX_LINKAGE
void clang_RefactoringResult_dispose(CXRefactoringResult Result);

/**
 * \brief Load the indexer query results from a YAML string.
 *
 * Mainly used for testing.
 */
CINDEX_LINKAGE
enum CXErrorCode
clang_RefactoringContinuation_loadSerializedIndexerQueryResults(
    CXRefactoringContinuation Continuation, const char *Source);

/**
 * \brief Return the number of indexer queries that a refactoring continuation
 * has.
 */
CINDEX_LINKAGE
unsigned clang_RefactoringContinuation_getNumIndexerQueries(
    CXRefactoringContinuation Continuation);

/**
 * \brief Return the indexer query at index \p Index.
 */
CINDEX_LINKAGE
CXIndexerQuery clang_RefactoringContinuation_getIndexerQuery(
    CXRefactoringContinuation Continuation, unsigned Index);

/**
 * \brief Verify that the all of the indexer queries are satisfied by the
 * continuation.
 *
 * \returns Null if all of the queries are satisfied an no errors have been
 * reported, or a set of diagnostics that describes why the continuation should
 * not be run.
 */
CINDEX_LINKAGE
CXDiagnosticSet clang_RefactoringContinuation_verifyBeforeFinalizing(
    CXRefactoringContinuation Continuation);

/**
 * \brief Terminate the connection between the initiation TU and the refactoring
 * continuation.
 *
 * The continuation converts all the TU-specific state to TU-independent state.
 * The indexer queries that are associate with this continuation are also
 * invalidated.
 */
CINDEX_LINKAGE
void clang_RefactoringContinuation_finalizeEvaluationInInitationTU(
    CXRefactoringContinuation Continuation);

/**
 * \brief Continue performing the previously initiated and performed refactoring
 * operation in the given translation unit \p TU.
 */
CINDEX_LINKAGE
CXRefactoringResult clang_RefactoringContinuation_continueOperationInTU(
    CXRefactoringContinuation Continuation, CXTranslationUnit TU,
    CXString *OutFailureReason);

/**
 * \brief Free the given refactoring continuation.
 */
CINDEX_LINKAGE
void clang_RefactoringContinuation_dispose(
    CXRefactoringContinuation Continuation);

/**
 * @}
 */

/**
 * \defgroup CINDEX_REFACTOR_INDEXER_QUERY Indexer Queries.
 *
 * @{
 */

/**
 * \brief The types of indexer queries.
 */
enum CXIndexerQueryKind {
  CXIndexerQuery_Unknown = 0,

  /**
   * \brief The indexer should find the file that contains/should contain the
   * implementation of some declaration.
   * A file result is expected.
   */
  CXIndexerQuery_Decl_FileThatShouldImplement = 1,

  /**
   * \brief The indexer should determine if the some declaration is defined.
   * An integer result is expected.
   */
  CXIndexerQuery_Decl_IsDefined = 2,
};

/**
 * \brief Return the kind of the indexer query \p Query.
 */
CINDEX_LINKAGE
enum CXIndexerQueryKind clang_IndexerQuery_getKind(CXIndexerQuery Query);

/**
 * \brief Return the number of cursors that the \p Query has.
 */
CINDEX_LINKAGE
unsigned clang_IndexerQuery_getNumCursors(CXIndexerQuery Query);

/**
 * \brief Return the cursor at the given \p CursorIndex.
 */
CINDEX_LINKAGE
CXCursor clang_IndexerQuery_getCursor(CXIndexerQuery Query,
                                      unsigned CursorIndex);

/**
 * \brief The action that the indexer should take after evaluating the query.
 */
enum CXIndexerQueryAction {
  /**
   * \brief This result requires no further action.
   */
  CXIndexerQueryAction_None = 0,

  /**
   * \brief The indexer should run the \c CXRefactoringContinuaton in a
   * translation unit that contains this file.
   */
  CXIndexerQueryAction_RunContinuationInTUThatHasThisFile = 1,
};

/**
 * \brief Consumes an integer/boolean query result.
 */
CINDEX_LINKAGE
enum CXIndexerQueryAction
clang_IndexerQuery_consumeIntResult(CXIndexerQuery Query, unsigned CursorIndex,
                                    int Value);

/**
 * \brief Consumes a filename query result.
 *
 * This function may return
 * \c CXIndexerQueryAction_RunContinuationInTUThatHasThisFile which
 * should tell the indexer that it has to run the refactoring continuation in
 * the TU that contains this file.
 */
CINDEX_LINKAGE
enum CXIndexerQueryAction
clang_IndexerQuery_consumeFileResult(CXIndexerQuery Query, unsigned CursorIndex,
                                     const char *Filename);

/**
 * @}
 */

#ifdef __cplusplus
}
#endif

#endif /* LLVM_CLANG_C_REFACTOR_H */
