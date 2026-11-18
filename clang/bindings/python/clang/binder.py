# pyright: reportPrivateUsage=false

# Enable delayed evaluation of function annotations.
from __future__ import annotations

import os
from ctypes import cdll
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union

from typing_extensions import Annotated

from .ctyped import *
from .ctyped import (ANNO_PARAMETER, ANNO_RESULT, ANNO_RESULT_CONVERTER,
                     generate_metadata)

if TYPE_CHECKING:
    from ctypes import CDLL
    from types import EllipsisType

    from .cindex import (CCRStructure, CodeCompletionResults,
                         CompilationDatabase, CompileCommands, Cursor,
                         CursorKind, Diagnostic, File, FileInclusion, Index,
                         Rewriter, SourceLocation, SourceRange, StrPath,
                         TemplateArgumentKind, Token, TranslationUnit)
    from .cindex import Type as ASTType
    from .cindex import _CXString, _CXUnsavedFile
else:
    EllipsisType = type(Ellipsis)


# delayed imports, a list of import name and their alias
# if alias is same as name, use `...`
CINDEX_DELAYED_IMPORTS: List[Tuple[str, Union[str, EllipsisType]]] = [
    ('CCRStructure', ...),
    ('CodeCompletionResults', ...),
    ('CompilationDatabase', ...),
    ('CompileCommands', ...),
    ('Cursor', ...),
    ('CursorKind', ...),
    ('Diagnostic', ...),
    ('File', ...),
    ('FileInclusion', ...),
    ('Index', ...),
    ('Rewriter', ...),
    ('SourceLocation', ...),
    ('SourceRange', ...),
    ('TemplateArgumentKind', ...),
    ('Token', ...),
    ('TranslationUnit', ...),
    ('Type', 'ASTType'),
    ('_CXString', ...),
    ('_CXUnsavedFile', ...),
    ('c_interop_string', ...),
]

def load_cindex_types() -> None:
    cindex_imports: Dict[str, Any] = {}
    from . import cindex
    for name, alias in CINDEX_DELAYED_IMPORTS:
        if isinstance(alias, EllipsisType): alias = name
        cindex_imports[alias] = getattr(cindex, name)
    globals().update(cindex_imports)


# ctypes doesn't implicitly convert c_void_p to the appropriate wrapper
# object. This is a problem, because it means that from_parameter will see an
# integer and pass the wrong value on platforms where int != void*. Work around
# this by marshalling object arguments as void**.
CObjectP = CPointer[c_void_p]
c_object_p: Type[CObjectP] = convert_annotation(CObjectP)


# Register callback types
TranslationUnitIncludesCallback = Annotated[CFuncPointer, None, c_object_p, CPointer['SourceLocation'], c_uint, py_object]
CursorVisitCallback = Annotated[CFuncPointer, c_int, 'Cursor', 'Cursor', py_object]
FieldsVisitCallback = Annotated[CFuncPointer, c_int, 'Cursor', py_object]

# TODO: these lines should replace the definition in cindex.py
#translation_unit_includes_callback: Type[CFuncPointer] = convert_annotation(TranslationUnitIncludesCallback, globals())
#cursor_visit_callback: Type[CFuncPointer] = convert_annotation(CursorVisitCallback, globals())
#fields_visit_callback: Type[CFuncPointer] = convert_annotation(FieldsVisitCallback, globals())


# Misc object param/result types
# A type may only have param type or result type, this is normal.
ASTTypeResult = Annotated['ASTType', ANNO_RESULT, 'ASTType', 'ASTType.from_result']

CInteropStringParam = Annotated[Union[str, bytes, None], ANNO_PARAMETER, 'c_interop_string']
CInteropStringResult = Annotated[Optional[str], ANNO_RESULT, 'c_interop_string', 'c_interop_string.to_python_string']

CXStringResult = Annotated[str, ANNO_RESULT, '_CXString', '_CXString.from_result']

CompilationDatabaseParam = Annotated['CompilationDatabase', ANNO_PARAMETER, c_object_p]
CompilationDatabaseResult = Annotated['CompilationDatabase', ANNO_RESULT, c_object_p, 'CompilationDatabase.from_result']

CompileCommandsResult = Annotated['CompileCommands', ANNO_RESULT, c_object_p, 'CompileCommands.from_result']

CursorResult = Annotated['Cursor', ANNO_RESULT, 'Cursor', 'Cursor.from_cursor_result']
CursorNullableResult = Annotated[Optional['Cursor'], ANNO_RESULT, 'Cursor', 'Cursor.from_result']

DiagnosticParam = Annotated['Diagnostic', ANNO_PARAMETER, c_object_p]

FileResult = Annotated['File', ANNO_RESULT, c_object_p, 'File.from_result']

TemplateArgumentKindResult = Annotated['TemplateArgumentKind', ANNO_RESULT_CONVERTER, 'TemplateArgumentKind.from_id']

TranslationUnitParam = Annotated['TranslationUnit', ANNO_PARAMETER, c_object_p]


# Functions strictly alphabetical order.
# NOTE:
#   - These functions are stubs, they are not implemented, and is replaced by C functions at runtime.
#   - If Config.compatibility_check is set to `False`, then a function is allowed to be missing.
#   - If a function is missing in C library, it will not be replaced, thus causing NotImplementedError when called.
#   - Missing functions are given a `_missing_` attribute, you can check it with `hasattr(conf.lib.xxx, '_missing_')`.
#   - These stub functions are generated with a script from old data and manually corrected, so parameter names are missing.
class LibclangExports:
    def clang_annotateTokens(self, p1: TranslationUnit, p2: CPointerParam[Token], p3: CUlongParam, p4: CPointerParam[Cursor]) -> CLongResult:
        raise NotImplementedError

    def clang_CompilationDatabase_dispose(self, p1: CompilationDatabaseParam) -> CLongResult:
        raise NotImplementedError

    def clang_CompilationDatabase_fromDirectory(self, p1: CInteropStringParam, p2: CPointerParam[c_ulong]) -> CompilationDatabaseResult:
        raise NotImplementedError

    def clang_CompilationDatabase_getAllCompileCommands(self, p1: CompilationDatabaseParam) -> CompileCommandsResult:
        raise NotImplementedError

    def clang_CompilationDatabase_getCompileCommands(self, p1: CompilationDatabaseParam, p2: CInteropStringParam) -> CompileCommandsResult:
        raise NotImplementedError

    def clang_CompileCommands_dispose(self, p1: CObjectP) -> CLongResult:
        raise NotImplementedError

    def clang_CompileCommands_getCommand(self, p1: CObjectP, p2: CUlongParam) -> CObjectP:
        raise NotImplementedError

    def clang_CompileCommands_getSize(self, p1: CObjectP) -> CUlongResult:
        raise NotImplementedError

    def clang_CompileCommand_getArg(self, p1: CObjectP, p2: CUlongParam) -> CXStringResult:
        raise NotImplementedError

    def clang_CompileCommand_getDirectory(self, p1: CObjectP) -> CXStringResult:
        raise NotImplementedError

    def clang_CompileCommand_getFilename(self, p1: CObjectP) -> CXStringResult:
        raise NotImplementedError

    def clang_CompileCommand_getNumArgs(self, p1: CObjectP) -> CUlongResult:
        raise NotImplementedError

    def clang_codeCompleteAt(self, p1: TranslationUnit, p2: CInteropStringParam, p3: CLongParam, p4: CLongParam, p5: CPointerParam[_CXUnsavedFile], p6: CLongParam, p7: CLongParam) -> CPointer[CCRStructure]:
        raise NotImplementedError

    def clang_codeCompleteGetDiagnostic(self, p1: CodeCompletionResults, p2: CLongParam) -> Diagnostic:
        raise NotImplementedError

    def clang_codeCompleteGetNumDiagnostics(self, p1: CodeCompletionResults) -> CLongResult:
        raise NotImplementedError

    def clang_createIndex(self, p1: CLongParam, p2: CLongParam) -> CObjectP:
        raise NotImplementedError

    def clang_createTranslationUnit(self, p1: Index, p2: CInteropStringParam) -> CObjectP:
        raise NotImplementedError

    def clang_CXRewriter_create(self, p1: TranslationUnit) -> CObjectP:
        raise NotImplementedError

    def clang_CXRewriter_dispose(self, p1: Rewriter) -> CLongResult:
        raise NotImplementedError

    def clang_CXRewriter_insertTextBefore(self, p1: Rewriter, p2: SourceLocation, p3: CInteropStringParam) -> CLongResult:
        raise NotImplementedError

    def clang_CXRewriter_overwriteChangedFiles(self, p1: Rewriter) -> CLongResult:
        raise NotImplementedError

    def clang_CXRewriter_removeText(self, p1: Rewriter, p2: SourceRange) -> CLongResult:
        raise NotImplementedError

    def clang_CXRewriter_replaceText(self, p1: Rewriter, p2: SourceRange, p3: CInteropStringParam) -> CLongResult:
        raise NotImplementedError

    def clang_CXRewriter_writeMainFileToStdOut(self, p1: Rewriter) -> CLongResult:
        raise NotImplementedError

    def clang_CXXConstructor_isConvertingConstructor(self, p1: Cursor) -> bool:
        raise NotImplementedError

    def clang_CXXConstructor_isCopyConstructor(self, p1: Cursor) -> bool:
        raise NotImplementedError

    def clang_CXXConstructor_isDefaultConstructor(self, p1: Cursor) -> bool:
        raise NotImplementedError

    def clang_CXXConstructor_isMoveConstructor(self, p1: Cursor) -> bool:
        raise NotImplementedError

    def clang_CXXField_isMutable(self, p1: Cursor) -> bool:
        raise NotImplementedError

    def clang_CXXMethod_isConst(self, p1: Cursor) -> bool:
        raise NotImplementedError

    def clang_CXXMethod_isDefaulted(self, p1: Cursor) -> bool:
        raise NotImplementedError

    def clang_CXXMethod_isDeleted(self, p1: Cursor) -> bool:
        raise NotImplementedError

    def clang_CXXMethod_isCopyAssignmentOperator(self, p1: Cursor) -> bool:
        raise NotImplementedError

    def clang_CXXMethod_isMoveAssignmentOperator(self, p1: Cursor) -> bool:
        raise NotImplementedError

    def clang_CXXMethod_isExplicit(self, p1: Cursor) -> bool:
        raise NotImplementedError

    def clang_CXXMethod_isPureVirtual(self, p1: Cursor) -> bool:
        raise NotImplementedError

    def clang_CXXMethod_isStatic(self, p1: Cursor) -> bool:
        raise NotImplementedError

    def clang_CXXMethod_isVirtual(self, p1: Cursor) -> bool:
        raise NotImplementedError

    def clang_CXXRecord_isAbstract(self, p1: Cursor) -> bool:
        raise NotImplementedError

    def clang_Cursor_getStorageClass(self, p1: Cursor) -> CIntResult:
        raise NotImplementedError

    def clang_EnumDecl_isScoped(self, p1: Cursor) -> bool:
        raise NotImplementedError

    def clang_defaultDiagnosticDisplayOptions(self) -> CUlongResult:
        raise NotImplementedError

    def clang_defaultSaveOptions(self, p1: TranslationUnit) -> CUlongResult:
        raise NotImplementedError

    def clang_disposeCodeCompleteResults(self, p1: CodeCompletionResults) -> CLongResult:
        raise NotImplementedError

    def clang_disposeDiagnostic(self, p1: Diagnostic) -> CLongResult:
        raise NotImplementedError

    def clang_disposeIndex(self, p1: Index) -> CLongResult:
        raise NotImplementedError

    def clang_disposeString(self, p1: _CXString) -> CLongResult:
        raise NotImplementedError

    def clang_disposeTokens(self, p1: TranslationUnit, p2: CPointer[Token], p3: CUintParam) -> CLongResult:
        raise NotImplementedError

    def clang_disposeTranslationUnit(self, p1: TranslationUnit) -> CLongResult:
        raise NotImplementedError

    def clang_equalCursors(self, p1: Cursor, p2: Cursor) -> bool:
        raise NotImplementedError

    def clang_equalLocations(self, p1: SourceLocation, p2: SourceLocation) -> bool:
        raise NotImplementedError

    def clang_equalRanges(self, p1: SourceRange, p2: SourceRange) -> bool:
        raise NotImplementedError

    def clang_equalTypes(self, p1: ASTType, p2: ASTType) -> bool:
        raise NotImplementedError

    def clang_formatDiagnostic(self, p1: Diagnostic, p2: CUlongParam) -> CXStringResult:
        raise NotImplementedError

    def clang_getAddressSpace(self, p1: ASTType) -> CIntResult:
        raise NotImplementedError

    def clang_getArgType(self, p1: ASTType, p2: CUlongParam) -> ASTTypeResult:
        raise NotImplementedError

    def clang_getArrayElementType(self, p1: ASTType) -> ASTTypeResult:
        raise NotImplementedError

    def clang_getArraySize(self, p1: ASTType) -> CLonglongResult:
        raise NotImplementedError

    def clang_getFieldDeclBitWidth(self, p1: Cursor) -> CLongResult:
        raise NotImplementedError

    def clang_getCanonicalCursor(self, p1: Cursor) -> CursorResult:
        raise NotImplementedError

    def clang_getCanonicalType(self, p1: ASTType) -> ASTTypeResult:
        raise NotImplementedError

    def clang_getChildDiagnostics(self, p1: Diagnostic) -> CObjectP:
        raise NotImplementedError

    def clang_getCompletionAvailability(self, p1: CObjectP) -> CLongResult:
        raise NotImplementedError

    def clang_getCompletionBriefComment(self, p1: CObjectP) -> CXStringResult:
        raise NotImplementedError

    def clang_getCompletionChunkCompletionString(self, p1: CObjectP, p2: CLongParam) -> CObjectP:
        raise NotImplementedError

    def clang_getCompletionChunkKind(self, p1: CObjectP, p2: CLongParam) -> CLongResult:
        raise NotImplementedError

    def clang_getCompletionChunkText(self, p1: CObjectP, p2: CLongParam) -> CXStringResult:
        raise NotImplementedError

    def clang_getCompletionPriority(self, p1: CObjectP) -> CLongResult:
        raise NotImplementedError

    def clang_getCString(self, p1: _CXString) -> CInteropStringResult:
        raise NotImplementedError

    def clang_getCursor(self, p1: TranslationUnit, p2: SourceLocation) -> Cursor:
        raise NotImplementedError

    def clang_getCursorAvailability(self, p1: Cursor) -> CLongResult:
        raise NotImplementedError

    def clang_getCursorDefinition(self, p1: Cursor) -> CursorNullableResult:
        raise NotImplementedError

    def clang_getCursorDisplayName(self, p1: Cursor) -> CXStringResult:
        raise NotImplementedError

    def clang_getCursorExceptionSpecificationType(self, p1: Cursor) -> CIntResult:
        raise NotImplementedError

    def clang_getCursorExtent(self, p1: Cursor) -> SourceRange:
        raise NotImplementedError

    def clang_getCursorLexicalParent(self, p1: Cursor) -> CursorResult:
        raise NotImplementedError

    def clang_getCursorLinkage(self, p1: Cursor) -> CIntResult:
        raise NotImplementedError

    def clang_getCursorLocation(self, p1: Cursor) -> SourceLocation:
        raise NotImplementedError

    def clang_getCursorReferenced(self, p1: Cursor) -> CursorNullableResult:
        raise NotImplementedError

    def clang_getCursorReferenceNameRange(self, p1: Cursor, p2: CUlongParam, p3: CUlongParam) -> SourceRange:
        raise NotImplementedError

    def clang_getCursorResultType(self, p1: Cursor) -> ASTTypeResult:
        raise NotImplementedError

    def clang_getCursorSemanticParent(self, p1: Cursor) -> CursorResult:
        raise NotImplementedError

    def clang_getCursorSpelling(self, p1: Cursor) -> CXStringResult:
        raise NotImplementedError

    def clang_getCursorTLSKind(self, p1: Cursor) -> CIntResult:
        raise NotImplementedError

    def clang_getCursorType(self, p1: Cursor) -> ASTTypeResult:
        raise NotImplementedError

    def clang_getCursorUSR(self, p1: Cursor) -> CXStringResult:
        raise NotImplementedError

    def clang_Cursor_getMangling(self, p1: Cursor) -> CXStringResult:
        raise NotImplementedError

    def clang_getCXXAccessSpecifier(self, p1: Cursor) -> CUlongResult:
        raise NotImplementedError

    def clang_getDeclObjCTypeEncoding(self, p1: Cursor) -> CXStringResult:
        raise NotImplementedError

    def clang_getDiagnostic(self, p1: TranslationUnitParam, p2: CUlongParam) -> CObjectP:
        raise NotImplementedError

    def clang_getDiagnosticCategory(self, p1: Diagnostic) -> CUlongResult:
        raise NotImplementedError

    def clang_getDiagnosticCategoryText(self, p1: Diagnostic) -> CXStringResult:
        raise NotImplementedError

    def clang_getDiagnosticFixIt(self, p1: Diagnostic, p2: CUlongParam, p3: CPointerParam[SourceRange]) -> CXStringResult:
        raise NotImplementedError

    def clang_getDiagnosticInSet(self, p1: CObjectP, p2: CUlongParam) -> CObjectP:
        raise NotImplementedError

    def clang_getDiagnosticLocation(self, p1: Diagnostic) -> SourceLocation:
        raise NotImplementedError

    def clang_getDiagnosticNumFixIts(self, p1: Diagnostic) -> CUlongResult:
        raise NotImplementedError

    def clang_getDiagnosticNumRanges(self, p1: Diagnostic) -> CUlongResult:
        raise NotImplementedError

    def clang_getDiagnosticOption(self, p1: Diagnostic, p2: CPointerParam[_CXString]) -> CXStringResult:
        raise NotImplementedError

    def clang_getDiagnosticRange(self, p1: Diagnostic, p2: CUlongParam) -> SourceRange:
        raise NotImplementedError

    def clang_getDiagnosticSeverity(self, p1: Diagnostic) -> CLongResult:
        raise NotImplementedError

    def clang_getDiagnosticSpelling(self, p1: Diagnostic) -> CXStringResult:
        raise NotImplementedError

    def clang_getElementType(self, p1: ASTType) -> ASTTypeResult:
        raise NotImplementedError

    def clang_getEnumConstantDeclUnsignedValue(self, p1: Cursor) -> CUlonglongResult:
        raise NotImplementedError

    def clang_getEnumConstantDeclValue(self, p1: Cursor) -> CLonglongResult:
        raise NotImplementedError

    def clang_getEnumDeclIntegerType(self, p1: Cursor) -> ASTTypeResult:
        raise NotImplementedError

    def clang_getExceptionSpecificationType(self, p1: ASTType) -> CIntResult:
        raise NotImplementedError

    def clang_getFile(self, p1: TranslationUnit, p2: CInteropStringParam) -> CObjectP:
        raise NotImplementedError

    def clang_getFileName(self, p1: File) -> CXStringResult:
        raise NotImplementedError

    def clang_getFileTime(self, p1: File) -> CUlongResult:
        raise NotImplementedError

    def clang_getIBOutletCollectionType(self, p1: Cursor) -> ASTTypeResult:
        raise NotImplementedError

    def clang_getIncludedFile(self, p1: Cursor) -> FileResult:
        raise NotImplementedError

    def clang_getInclusions(self, p1: TranslationUnit, p2: TranslationUnitIncludesCallback, p3: CPyObject[List[FileInclusion]]) -> CLongResult:
        raise NotImplementedError

    def clang_getInstantiationLocation(self, p1: SourceLocation, p2: CPointerParam[CObjectP], p3: CPointerParam[c_ulong], p4: CPointerParam[c_ulong], p5: CPointerParam[c_ulong]) -> CLongResult:
        raise NotImplementedError

    def clang_getLocation(self, p1: TranslationUnit, p2: File, p3: CUlongParam, p4: CUlongParam) -> SourceLocation:
        raise NotImplementedError

    def clang_getLocationForOffset(self, p1: TranslationUnit, p2: File, p3: CUlongParam) -> SourceLocation:
        raise NotImplementedError

    def clang_getNullCursor(self) -> Cursor:
        raise NotImplementedError

    def clang_getNumArgTypes(self, p1: ASTType) -> CUlongResult:
        raise NotImplementedError

    def clang_getNumCompletionChunks(self, p1: CObjectP) -> CLongResult:
        raise NotImplementedError

    def clang_getNumDiagnostics(self, p1: TranslationUnitParam) -> CUlongResult:
        raise NotImplementedError

    def clang_getNumDiagnosticsInSet(self, p1: CObjectP) -> CUlongResult:
        raise NotImplementedError

    def clang_getNumElements(self, p1: ASTType) -> CLonglongResult:
        raise NotImplementedError

    def clang_getNumOverloadedDecls(self, p1: Cursor) -> CUlongResult:
        raise NotImplementedError

    def clang_getOverloadedDecl(self, p1: Cursor, p2: CUlongParam) -> CursorResult:
        raise NotImplementedError

    def clang_getPointeeType(self, p1: ASTType) -> ASTTypeResult:
        raise NotImplementedError

    def clang_getRange(self, p1: SourceLocation, p2: SourceLocation) -> SourceRange:
        raise NotImplementedError

    def clang_getRangeEnd(self, p1: SourceRange) -> SourceLocation:
        raise NotImplementedError

    def clang_getRangeStart(self, p1: SourceRange) -> SourceLocation:
        raise NotImplementedError

    def clang_getResultType(self, p1: ASTType) -> ASTTypeResult:
        raise NotImplementedError

    def clang_getSpecializedCursorTemplate(self, p1: Cursor) -> CursorResult:
        raise NotImplementedError

    def clang_getTemplateCursorKind(self, p1: Cursor) -> CUlongResult:
        raise NotImplementedError

    def clang_getTokenExtent(self, p1: TranslationUnit, p2: Token) -> SourceRange:
        raise NotImplementedError

    def clang_getTokenKind(self, p1: Token) -> CUlongResult:
        raise NotImplementedError

    def clang_getTokenLocation(self, p1: TranslationUnit, p2: Token) -> SourceLocation:
        raise NotImplementedError

    def clang_getTokenSpelling(self, p1: TranslationUnit, p2: Token) -> CXStringResult:
        raise NotImplementedError

    def clang_getTranslationUnitCursor(self, p1: TranslationUnit) -> CursorNullableResult:
        raise NotImplementedError

    def clang_getTranslationUnitSpelling(self, p1: TranslationUnit) -> CXStringResult:
        raise NotImplementedError

    def clang_getTUResourceUsageName(self, p1: CUlongParam) -> CInteropStringResult:
        raise NotImplementedError

    def clang_getTypeDeclaration(self, p1: ASTType) -> CursorNullableResult:
        raise NotImplementedError

    def clang_getTypedefDeclUnderlyingType(self, p1: Cursor) -> ASTTypeResult:
        raise NotImplementedError

    def clang_getTypedefName(self, p1: ASTType) -> CXStringResult:
        raise NotImplementedError

    def clang_getTypeKindSpelling(self, p1: CUlongParam) -> CXStringResult:
        raise NotImplementedError

    def clang_getTypeSpelling(self, p1: ASTType) -> CXStringResult:
        raise NotImplementedError

    def clang_hashCursor(self, p1: Cursor) -> CUlongResult:
        raise NotImplementedError

    def clang_isAttribute(self, p1: CursorKind) -> bool:
        raise NotImplementedError

    def clang_isConstQualifiedType(self, p1: ASTType) -> bool:
        raise NotImplementedError

    def clang_isCursorDefinition(self, p1: Cursor) -> bool:
        raise NotImplementedError

    def clang_isDeclaration(self, p1: CursorKind) -> bool:
        raise NotImplementedError

    def clang_isExpression(self, p1: CursorKind) -> bool:
        raise NotImplementedError

    def clang_isFileMultipleIncludeGuarded(self, p1: TranslationUnit, p2: File) -> bool:
        raise NotImplementedError

    def clang_isFunctionTypeVariadic(self, p1: ASTType) -> bool:
        raise NotImplementedError

    def clang_isInvalid(self, p1: CursorKind) -> bool:
        raise NotImplementedError

    def clang_isPODType(self, p1: ASTType) -> bool:
        raise NotImplementedError

    def clang_isPreprocessing(self, p1: CursorKind) -> bool:
        raise NotImplementedError

    def clang_isReference(self, p1: CursorKind) -> bool:
        raise NotImplementedError

    def clang_isRestrictQualifiedType(self, p1: ASTType) -> bool:
        raise NotImplementedError

    def clang_isStatement(self, p1: CursorKind) -> bool:
        raise NotImplementedError

    def clang_isTranslationUnit(self, p1: CursorKind) -> bool:
        raise NotImplementedError

    def clang_isUnexposed(self, p1: CursorKind) -> bool:
        raise NotImplementedError

    def clang_isVirtualBase(self, p1: Cursor) -> bool:
        raise NotImplementedError

    def clang_isVolatileQualifiedType(self, p1: ASTType) -> bool:
        raise NotImplementedError

    def clang_parseTranslationUnit(self, p1: Index, p2: CInteropStringParam, p3: CPointerParam[c_char_p], p4: CLongParam, p5: CPointerParam[_CXUnsavedFile], p6: CLongParam, p7: CLongParam) -> CObjectP:
        raise NotImplementedError

    def clang_reparseTranslationUnit(self, p1: TranslationUnit, p2: CLongParam, p3: CPointerParam[_CXUnsavedFile], p4: CLongParam) -> CLongResult:
        raise NotImplementedError

    def clang_saveTranslationUnit(self, p1: TranslationUnit, p2: CInteropStringParam, p3: CUlongParam) -> CLongResult:
        raise NotImplementedError

    def clang_tokenize(self, p1: TranslationUnit, p2: SourceRange, p3: CPointerParam[CPointer[Token]], p4: CPointerParam[c_ulong]) -> CLongResult:
        raise NotImplementedError

    def clang_visitChildren(self, p1: Cursor, p2: CursorVisitCallback, p3: CPyObject[List[Cursor]]) -> CUlongResult:
        raise NotImplementedError

    def clang_Cursor_getNumArguments(self, p1: Cursor) -> CLongResult:
        raise NotImplementedError

    def clang_Cursor_getArgument(self, p1: Cursor, p2: CUlongParam) -> CursorNullableResult:
        raise NotImplementedError

    def clang_Cursor_getNumTemplateArguments(self, p1: Cursor) -> CLongResult:
        raise NotImplementedError

    def clang_Cursor_getTemplateArgumentKind(self, p1: Cursor, p2: CUlongParam) -> TemplateArgumentKindResult:
        raise NotImplementedError

    def clang_Cursor_getTemplateArgumentType(self, p1: Cursor, p2: CUlongParam) -> ASTTypeResult:
        raise NotImplementedError

    def clang_Cursor_getTemplateArgumentValue(self, p1: Cursor, p2: CUlongParam) -> CLonglongResult:
        raise NotImplementedError

    def clang_Cursor_getTemplateArgumentUnsignedValue(self, p1: Cursor, p2: CUlongParam) -> CUlonglongResult:
        raise NotImplementedError

    def clang_Cursor_isAnonymous(self, p1: Cursor) -> bool:
        raise NotImplementedError

    def clang_Cursor_isBitField(self, p1: Cursor) -> bool:
        raise NotImplementedError

    def clang_Cursor_getBinaryOpcode(self, p1: Cursor) -> CLongResult:
        raise NotImplementedError

    def clang_Cursor_getBriefCommentText(self, p1: Cursor) -> CXStringResult:
        raise NotImplementedError

    def clang_Cursor_getRawCommentText(self, p1: Cursor) -> CXStringResult:
        raise NotImplementedError

    def clang_Cursor_getOffsetOfField(self, p1: Cursor) -> CLonglongResult:
        raise NotImplementedError

    def clang_Location_isInSystemHeader(self, p1: SourceLocation) -> bool:
        raise NotImplementedError

    def clang_Type_getAlignOf(self, p1: ASTType) -> CLonglongResult:
        raise NotImplementedError

    def clang_Type_getClassType(self, p1: ASTType) -> ASTTypeResult:
        raise NotImplementedError

    def clang_Type_getNumTemplateArguments(self, p1: ASTType) -> CLongResult:
        raise NotImplementedError

    def clang_Type_getTemplateArgumentAsType(self, p1: ASTType, p2: CUlongParam) -> ASTTypeResult:
        raise NotImplementedError

    def clang_Type_getOffsetOf(self, p1: ASTType, p2: CInteropStringParam) -> CLonglongResult:
        raise NotImplementedError

    def clang_Type_getSizeOf(self, p1: ASTType) -> CLonglongResult:
        raise NotImplementedError

    def clang_Type_getCXXRefQualifier(self, p1: ASTType) -> CUlongResult:
        raise NotImplementedError

    def clang_Type_getNamedType(self, p1: ASTType) -> ASTTypeResult:
        raise NotImplementedError

    def clang_Type_visitFields(self, p1: ASTType, p2: FieldsVisitCallback, p3: CPyObject[List[Cursor]]) -> CUlongResult:
        raise NotImplementedError


class LibclangError(Exception):
    m: str

    def __init__(self, message: str):
        self.m = message

    def __str__(self) -> str:
        return self.m


class Config:
    _lib: Optional[LibclangExports] = None

    library_path: Optional[str] = None
    library_file: Optional[str] = None
    compatibility_check: bool = True
    loaded: bool = False

    @staticmethod
    def set_library_path(path: StrPath) -> None:
        """Set the path in which to search for libclang"""
        if Config.loaded:
            raise Exception(
                "library path must be set before before using "
                "any other functionalities in libclang."
            )

        Config.library_path = os.fspath(path)

    @staticmethod
    def set_library_file(filename: StrPath) -> None:
        """Set the exact location of libclang"""
        if Config.loaded:
            raise Exception(
                "library file must be set before before using "
                "any other functionalities in libclang."
            )

        Config.library_file = os.fspath(filename)

    @staticmethod
    def set_compatibility_check(check_status: bool) -> None:
        """Perform compatibility check when loading libclang

        The python bindings are only tested and evaluated with the version of
        libclang they are provided with. To ensure correct behavior a (limited)
        compatibility check is performed when loading the bindings. This check
        will throw an exception, as soon as it fails.

        In case these bindings are used with an older version of libclang, parts
        that have been stable between releases may still work. Users of the
        python bindings can disable the compatibility check. This will cause
        the python bindings to load, even though they are written for a newer
        version of libclang. Failures now arise if unsupported or incompatible
        features are accessed. The user is required to test themselves if the
        features they are using are available and compatible between different
        libclang versions.
        """
        if Config.loaded:
            raise Exception(
                "compatibility_check must be set before before "
                "using any other functionalities in libclang."
            )

        Config.compatibility_check = check_status

    @property
    def lib(self) -> LibclangExports:
        if self._lib is None:
            clib = self.get_cindex_library()
            load_cindex_types()
            exports, missing = load_annotated_library(clib, LibclangExports, globals())
            if Config.compatibility_check and missing:
                raise LibclangError(
                    f"Missing functions: {missing}. Please ensure that your python"
                    "bindings are compatible with your libclang.so version."
                )
            Config.loaded = True
            self._lib = exports
        return self._lib

    @staticmethod
    def cfunc_metadata() -> Dict[str, Dict[str, Any]]:
        ''' Generate ctypes metadata for debugging purpose. '''
        load_cindex_types()
        return {name: info for name, info in generate_metadata(LibclangExports, globals())}

    def get_filename(self) -> str:
        if Config.library_file:
            return Config.library_file

        from platform import system as sysname

        name = sysname()

        if name == "Darwin":
            file = "libclang.dylib"
        elif name == "Windows":
            file = "libclang.dll"
        else:
            file = "libclang.so"

        if Config.library_path:
            file = Config.library_path + "/" + file

        return file

    def get_cindex_library(self) -> CDLL:
        try:
            library = cdll.LoadLibrary(self.get_filename())
        except OSError as e:
            msg = (
                str(e) + ". To provide a path to libclang use "
                "Config.set_library_path() or "
                "Config.set_library_file()."
            )
            raise LibclangError(msg)

        return library

    def function_exists(self, name: str) -> bool:
        return not hasattr(getattr(self.lib, name), '_missing_')
