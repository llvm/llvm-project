# mypy: ignore-errors

import re
import unittest
from ctypes import *  # pyright: ignore[reportWildcardImportFromLibrary]
from typing import Any, Dict, List, Set, Tuple

from dictdiffer import diff as dictdiff  # type: ignore

from clang.cindex import *
from clang.cindex import _CXString  # pyright: ignore[reportPrivateUsage]
from clang.cindex import (CCRStructure, Rewriter, c_interop_string, c_object_p,
                          cursor_visit_callback, fields_visit_callback,
                          generate_metadata_debug,
                          translation_unit_includes_callback)


# Functions strictly alphabetical order.
# This is previous version of ctypes metadata, we check equality to this so
# that we can ensure `ctyped` doesn't break anything in its conversion.
FUNCTION_LIST: List[Tuple[Any, ...]] = [
    (
        "clang_annotateTokens",
        [TranslationUnit, POINTER(Token), c_uint, POINTER(Cursor)],
    ),
    ("clang_CompilationDatabase_dispose", [c_object_p]),
    (
        "clang_CompilationDatabase_fromDirectory",
        [c_interop_string, POINTER(c_uint)],
        c_object_p,
        CompilationDatabase.from_result,
    ),
    (
        "clang_CompilationDatabase_getAllCompileCommands",
        [c_object_p],
        c_object_p,
        CompileCommands.from_result,
    ),
    (
        "clang_CompilationDatabase_getCompileCommands",
        [c_object_p, c_interop_string],
        c_object_p,
        CompileCommands.from_result,
    ),
    ("clang_CompileCommands_dispose", [c_object_p]),
    ("clang_CompileCommands_getCommand", [c_object_p, c_uint], c_object_p),
    ("clang_CompileCommands_getSize", [c_object_p], c_uint),
    (
        "clang_CompileCommand_getArg",
        [c_object_p, c_uint],
        _CXString,
        _CXString.from_result,
    ),
    (
        "clang_CompileCommand_getDirectory",
        [c_object_p],
        _CXString,
        _CXString.from_result,
    ),
    (
        "clang_CompileCommand_getFilename",
        [c_object_p],
        _CXString,
        _CXString.from_result,
    ),
    ("clang_CompileCommand_getNumArgs", [c_object_p], c_uint),
    (
        "clang_codeCompleteAt",
        [TranslationUnit, c_interop_string, c_int, c_int, c_void_p, c_int, c_int],
        POINTER(CCRStructure),
    ),
    ("clang_codeCompleteGetDiagnostic", [CodeCompletionResults, c_int], Diagnostic),
    ("clang_codeCompleteGetNumDiagnostics", [CodeCompletionResults], c_int),
    ("clang_createIndex", [c_int, c_int], c_object_p),
    ("clang_createTranslationUnit", [Index, c_interop_string], c_object_p),
    ("clang_CXRewriter_create", [TranslationUnit], c_object_p),
    ("clang_CXRewriter_dispose", [Rewriter]),
    ("clang_CXRewriter_insertTextBefore", [Rewriter, SourceLocation, c_interop_string]),
    ("clang_CXRewriter_overwriteChangedFiles", [Rewriter], c_int),
    ("clang_CXRewriter_removeText", [Rewriter, SourceRange]),
    ("clang_CXRewriter_replaceText", [Rewriter, SourceRange, c_interop_string]),
    ("clang_CXRewriter_writeMainFileToStdOut", [Rewriter]),
    ("clang_CXXConstructor_isConvertingConstructor", [Cursor], bool),
    ("clang_CXXConstructor_isCopyConstructor", [Cursor], bool),
    ("clang_CXXConstructor_isDefaultConstructor", [Cursor], bool),
    ("clang_CXXConstructor_isMoveConstructor", [Cursor], bool),
    ("clang_CXXField_isMutable", [Cursor], bool),
    ("clang_CXXMethod_isConst", [Cursor], bool),
    ("clang_CXXMethod_isDefaulted", [Cursor], bool),
    ("clang_CXXMethod_isDeleted", [Cursor], bool),
    ("clang_CXXMethod_isCopyAssignmentOperator", [Cursor], bool),
    ("clang_CXXMethod_isMoveAssignmentOperator", [Cursor], bool),
    ("clang_CXXMethod_isExplicit", [Cursor], bool),
    ("clang_CXXMethod_isPureVirtual", [Cursor], bool),
    ("clang_CXXMethod_isStatic", [Cursor], bool),
    ("clang_CXXMethod_isVirtual", [Cursor], bool),
    ("clang_CXXRecord_isAbstract", [Cursor], bool),
    ("clang_EnumDecl_isScoped", [Cursor], bool),
    ("clang_defaultDiagnosticDisplayOptions", [], c_uint),
    ("clang_defaultSaveOptions", [TranslationUnit], c_uint),
    ("clang_disposeCodeCompleteResults", [CodeCompletionResults]),
    # ("clang_disposeCXTUResourceUsage",
    #  [CXTUResourceUsage]),
    ("clang_disposeDiagnostic", [Diagnostic]),
    ("clang_disposeIndex", [Index]),
    ("clang_disposeString", [_CXString]),
    ("clang_disposeTokens", [TranslationUnit, POINTER(Token), c_uint]),
    ("clang_disposeTranslationUnit", [TranslationUnit]),
    ("clang_equalCursors", [Cursor, Cursor], bool),
    ("clang_equalLocations", [SourceLocation, SourceLocation], bool),
    ("clang_equalRanges", [SourceRange, SourceRange], bool),
    ("clang_equalTypes", [Type, Type], bool),
    ("clang_formatDiagnostic", [Diagnostic, c_uint], _CXString, _CXString.from_result),
    ("clang_getArgType", [Type, c_uint], Type, Type.from_result),
    ("clang_getArrayElementType", [Type], Type, Type.from_result),
    ("clang_getArraySize", [Type], c_longlong),
    ("clang_getFieldDeclBitWidth", [Cursor], c_int),
    ("clang_getCanonicalCursor", [Cursor], Cursor, Cursor.from_cursor_result),
    ("clang_getCanonicalType", [Type], Type, Type.from_result),
    ("clang_getChildDiagnostics", [Diagnostic], c_object_p),
    ("clang_getCompletionAvailability", [c_void_p], c_int),
    ("clang_getCompletionBriefComment", [c_void_p], _CXString, _CXString.from_result),
    ("clang_getCompletionChunkCompletionString", [c_void_p, c_int], c_object_p),
    ("clang_getCompletionChunkKind", [c_void_p, c_int], c_int),
    (
        "clang_getCompletionChunkText",
        [c_void_p, c_int],
        _CXString,
        _CXString.from_result,
    ),
    ("clang_getCompletionPriority", [c_void_p], c_int),
    (
        "clang_getCString",
        [_CXString],
        c_interop_string,
        c_interop_string.to_python_string,
    ),
    ("clang_getCursor", [TranslationUnit, SourceLocation], Cursor),
    ("clang_getCursorAvailability", [Cursor], c_int),
    ("clang_getCursorDefinition", [Cursor], Cursor, Cursor.from_result),
    ("clang_getCursorDisplayName", [Cursor], _CXString, _CXString.from_result),
    ("clang_getCursorExtent", [Cursor], SourceRange),
    ("clang_getCursorLexicalParent", [Cursor], Cursor, Cursor.from_cursor_result),
    ("clang_getCursorLocation", [Cursor], SourceLocation),
    ("clang_getCursorReferenced", [Cursor], Cursor, Cursor.from_result),
    ("clang_getCursorReferenceNameRange", [Cursor, c_uint, c_uint], SourceRange),
    ("clang_getCursorResultType", [Cursor], Type, Type.from_result),
    ("clang_getCursorSemanticParent", [Cursor], Cursor, Cursor.from_cursor_result),
    ("clang_getCursorSpelling", [Cursor], _CXString, _CXString.from_result),
    ("clang_getCursorType", [Cursor], Type, Type.from_result),
    ("clang_getCursorUSR", [Cursor], _CXString, _CXString.from_result),
    ("clang_Cursor_getMangling", [Cursor], _CXString, _CXString.from_result),
    # ("clang_getCXTUResourceUsage",
    #  [TranslationUnit],
    #  CXTUResourceUsage),
    ("clang_getCXXAccessSpecifier", [Cursor], c_uint),
    ("clang_getDeclObjCTypeEncoding", [Cursor], _CXString, _CXString.from_result),
    ("clang_getDiagnostic", [c_object_p, c_uint], c_object_p),
    ("clang_getDiagnosticCategory", [Diagnostic], c_uint),
    ("clang_getDiagnosticCategoryText", [Diagnostic], _CXString, _CXString.from_result),
    (
        "clang_getDiagnosticFixIt",
        [Diagnostic, c_uint, POINTER(SourceRange)],
        _CXString,
        _CXString.from_result,
    ),
    ("clang_getDiagnosticInSet", [c_object_p, c_uint], c_object_p),
    ("clang_getDiagnosticLocation", [Diagnostic], SourceLocation),
    ("clang_getDiagnosticNumFixIts", [Diagnostic], c_uint),
    ("clang_getDiagnosticNumRanges", [Diagnostic], c_uint),
    (
        "clang_getDiagnosticOption",
        [Diagnostic, POINTER(_CXString)],
        _CXString,
        _CXString.from_result,
    ),
    ("clang_getDiagnosticRange", [Diagnostic, c_uint], SourceRange),
    ("clang_getDiagnosticSeverity", [Diagnostic], c_int),
    ("clang_getDiagnosticSpelling", [Diagnostic], _CXString, _CXString.from_result),
    ("clang_getElementType", [Type], Type, Type.from_result),
    ("clang_getEnumConstantDeclUnsignedValue", [Cursor], c_ulonglong),
    ("clang_getEnumConstantDeclValue", [Cursor], c_longlong),
    ("clang_getEnumDeclIntegerType", [Cursor], Type, Type.from_result),
    ("clang_getFile", [TranslationUnit, c_interop_string], c_object_p),
    ("clang_getFileName", [File], _CXString, _CXString.from_result),
    ("clang_getFileTime", [File], c_uint),
    ("clang_getIBOutletCollectionType", [Cursor], Type, Type.from_result),
    ("clang_getIncludedFile", [Cursor], c_object_p, File.from_result),
    (
        "clang_getInclusions",
        [TranslationUnit, translation_unit_includes_callback, py_object],
    ),
    (
        "clang_getInstantiationLocation",
        [
            SourceLocation,
            POINTER(c_object_p),
            POINTER(c_uint),
            POINTER(c_uint),
            POINTER(c_uint),
        ],
    ),
    ("clang_getLocation", [TranslationUnit, File, c_uint, c_uint], SourceLocation),
    ("clang_getLocationForOffset", [TranslationUnit, File, c_uint], SourceLocation),
    ("clang_getNullCursor", None, Cursor),
    ("clang_getNumArgTypes", [Type], c_uint),
    ("clang_getNumCompletionChunks", [c_void_p], c_int),
    ("clang_getNumDiagnostics", [c_object_p], c_uint),
    ("clang_getNumDiagnosticsInSet", [c_object_p], c_uint),
    ("clang_getNumElements", [Type], c_longlong),
    ("clang_getNumOverloadedDecls", [Cursor], c_uint),
    ("clang_getOverloadedDecl", [Cursor, c_uint], Cursor, Cursor.from_cursor_result),
    ("clang_getPointeeType", [Type], Type, Type.from_result),
    ("clang_getRange", [SourceLocation, SourceLocation], SourceRange),
    ("clang_getRangeEnd", [SourceRange], SourceLocation),
    ("clang_getRangeStart", [SourceRange], SourceLocation),
    ("clang_getResultType", [Type], Type, Type.from_result),
    ("clang_getSpecializedCursorTemplate", [Cursor], Cursor, Cursor.from_cursor_result),
    ("clang_getTemplateCursorKind", [Cursor], c_uint),
    ("clang_getTokenExtent", [TranslationUnit, Token], SourceRange),
    ("clang_getTokenKind", [Token], c_uint),
    ("clang_getTokenLocation", [TranslationUnit, Token], SourceLocation),
    (
        "clang_getTokenSpelling",
        [TranslationUnit, Token],
        _CXString,
        _CXString.from_result,
    ),
    ("clang_getTranslationUnitCursor", [TranslationUnit], Cursor, Cursor.from_result),
    (
        "clang_getTranslationUnitSpelling",
        [TranslationUnit],
        _CXString,
        _CXString.from_result,
    ),
    (
        "clang_getTUResourceUsageName",
        [c_uint],
        c_interop_string,
        c_interop_string.to_python_string,
    ),
    ("clang_getTypeDeclaration", [Type], Cursor, Cursor.from_result),
    ("clang_getTypedefDeclUnderlyingType", [Cursor], Type, Type.from_result),
    ("clang_getTypedefName", [Type], _CXString, _CXString.from_result),
    ("clang_getTypeKindSpelling", [c_uint], _CXString, _CXString.from_result),
    ("clang_getTypeSpelling", [Type], _CXString, _CXString.from_result),
    ("clang_hashCursor", [Cursor], c_uint),
    ("clang_isAttribute", [CursorKind], bool),
    ("clang_isConstQualifiedType", [Type], bool),
    ("clang_isCursorDefinition", [Cursor], bool),
    ("clang_isDeclaration", [CursorKind], bool),
    ("clang_isExpression", [CursorKind], bool),
    ("clang_isFileMultipleIncludeGuarded", [TranslationUnit, File], bool),
    ("clang_isFunctionTypeVariadic", [Type], bool),
    ("clang_isInvalid", [CursorKind], bool),
    ("clang_isPODType", [Type], bool),
    ("clang_isPreprocessing", [CursorKind], bool),
    ("clang_isReference", [CursorKind], bool),
    ("clang_isRestrictQualifiedType", [Type], bool),
    ("clang_isStatement", [CursorKind], bool),
    ("clang_isTranslationUnit", [CursorKind], bool),
    ("clang_isUnexposed", [CursorKind], bool),
    ("clang_isVirtualBase", [Cursor], bool),
    ("clang_isVolatileQualifiedType", [Type], bool),
    (
        "clang_parseTranslationUnit",
        [Index, c_interop_string, c_void_p, c_int, c_void_p, c_int, c_int],
        c_object_p,
    ),
    ("clang_reparseTranslationUnit", [TranslationUnit, c_int, c_void_p, c_int], c_int),
    ("clang_saveTranslationUnit", [TranslationUnit, c_interop_string, c_uint], c_int),
    (
        "clang_tokenize",
        [TranslationUnit, SourceRange, POINTER(POINTER(Token)), POINTER(c_uint)],
    ),
    ("clang_visitChildren", [Cursor, cursor_visit_callback, py_object], c_uint),
    ("clang_Cursor_getNumArguments", [Cursor], c_int),
    ("clang_Cursor_getArgument", [Cursor, c_uint], Cursor, Cursor.from_result),
    ("clang_Cursor_getNumTemplateArguments", [Cursor], c_int),
    (
        "clang_Cursor_getTemplateArgumentKind",
        [Cursor, c_uint],
        TemplateArgumentKind.from_id,
    ),
    ("clang_Cursor_getTemplateArgumentType", [Cursor, c_uint], Type, Type.from_result),
    ("clang_Cursor_getTemplateArgumentValue", [Cursor, c_uint], c_longlong),
    ("clang_Cursor_getTemplateArgumentUnsignedValue", [Cursor, c_uint], c_ulonglong),
    ("clang_Cursor_isAnonymous", [Cursor], bool),
    ("clang_Cursor_isBitField", [Cursor], bool),
    ("clang_Cursor_getBinaryOpcode", [Cursor], c_int),
    ("clang_Cursor_getBriefCommentText", [Cursor], _CXString, _CXString.from_result),
    ("clang_Cursor_getRawCommentText", [Cursor], _CXString, _CXString.from_result),
    ("clang_Cursor_getOffsetOfField", [Cursor], c_longlong),
    ("clang_Location_isInSystemHeader", [SourceLocation], bool),
    ("clang_Type_getAlignOf", [Type], c_longlong),
    ("clang_Type_getClassType", [Type], Type, Type.from_result),
    ("clang_Type_getNumTemplateArguments", [Type], c_int),
    ("clang_Type_getTemplateArgumentAsType", [Type, c_uint], Type, Type.from_result),
    ("clang_Type_getOffsetOf", [Type, c_interop_string], c_longlong),
    ("clang_Type_getSizeOf", [Type], c_longlong),
    ("clang_Type_getCXXRefQualifier", [Type], c_uint),
    ("clang_Type_getNamedType", [Type], Type, Type.from_result),
    ("clang_Type_visitFields", [Type, fields_visit_callback, py_object], c_uint),
]


# Sadly, ctypes provides no API to check if type is pointer or array.
# Here we use regex to check type name.
arr_regex = re.compile(r'(?P<typ>[A-Za-z0-9_]+)_Array_(?P<count>[0-9]+)')
ptr_regex = re.compile(r'LP_(?P<typ>[A-Za-z0-9_]+)')

def is_ptr_type(typ: Any):
    return typ in (c_void_p, c_char_p, c_wchar_p) or ptr_regex.fullmatch(typ.__name__) is not None

def is_arr_type(typ: Any):
    return arr_regex.fullmatch(typ.__name__) is not None

# If we change a c_void_p parameter to a more exact pointer types, it
# should still be working.
def is_void_specialization(old_type: Any, new_type: Any):
    return old_type == c_void_p and is_ptr_type(new_type)


def old_data_to_dict(data: List[Any]):
    result: Dict[str, Any] = {}
    result['argtypes'], *data = data
    if not result['argtypes']: result['argtypes'] = None
    if data: result['restype'], *data = data
    else: result['restype'] = c_int
    if data: result['errcheck'], *data = data
    return result


def is_incompatible_diff(diff: Any):
    kind, path, detail = diff # pyright: ignore[reportUnusedVariable]
    if kind == 'add': return True
    old_type, new_type = detail
    if is_void_specialization(old_type, new_type): return False
    return True


class TestStubConversion(unittest.TestCase):
    def test_equality(self):
        """Ensure that ctyped does not break anything."""
        old_function_dict: Dict[str, Dict[str, Any]] = {name: old_data_to_dict(val) for name, *val in FUNCTION_LIST}
        new_function_dict = generate_metadata_debug()

        missing_functions = set(old_function_dict.keys())
        stable_functions: Set[str] = set()
        for new_func in new_function_dict:
            if new_func in missing_functions:
                missing_functions.remove(new_func)
                stable_functions.add(new_func)

        type_diff = [list(dictdiff(old_function_dict[name], new_function_dict[name])) for name in stable_functions] # type: ignore
        type_break = [diffset for diffset in type_diff if diffset and any(is_incompatible_diff(diff) for diff in diffset)] # type: ignore

        self.assertTrue(not missing_functions, f'Functions {missing_functions} are missing after stub conversion!')
        self.assertTrue(not type_break, f'Type break happens after stub conversion!')
