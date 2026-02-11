# pyright: reportPrivateUsage=false

import sys
from ctypes import (CFUNCTYPE, POINTER, c_bool, c_byte, c_char, c_char_p,
                    c_double, c_float, c_int, c_long, c_longdouble, c_longlong,
                    c_short, c_size_t, c_ssize_t, c_ubyte, c_uint, c_ulong,
                    c_ulonglong, c_ushort, c_void_p, c_wchar, c_wchar_p,
                    py_object)
from inspect import Parameter, signature
from typing import (TYPE_CHECKING, Any, Callable, Dict, ForwardRef, Generator, Generic,
                    List, Optional, Tuple, Type, TypeVar, Union, cast)

from typing_extensions import Annotated, ParamSpec, TypeAlias

_T = TypeVar('_T')

if TYPE_CHECKING:
    from ctypes import _CArgObject, _CData

AnyCData = TypeVar('AnyCData', bound='_CData')


if TYPE_CHECKING:
    from ctypes import Array as _Array
    from ctypes import _FuncPointer as _FuncPointer
    from ctypes import _Pointer as _Pointer

    # ctypes documentation noted implicit conversion for pointers:
    # "For example, you can pass compatible array instances instead of pointer
    #  types. So, for POINTER(c_int), ctypes accepts an array of c_int:"
    # "In addition, if a function argument is explicitly declared to be a
    #  pointer type (such as POINTER(c_int)) in argtypes, an object of the
    #  pointed type (c_int in this case) can be passed to the function. ctypes
    #  will apply the required byref() conversion in this case automatically."
    # also, current ctype typeshed thinks byref returns _CArgObject
    _PointerCompatible: TypeAlias = Union[_CArgObject, _Pointer[AnyCData], None, _Array[AnyCData], AnyCData]
    _PyObject: TypeAlias = Union[py_object[_T], _T]
else:
    # at runtime we don't really import those symbols
    class _Array(Generic[AnyCData]): ...
    class _Pointer(Generic[AnyCData]): ...
    class _PointerCompatible(Generic[AnyCData]): ...
    class _FuncPointer: ...
    class _PyObject(Generic[AnyCData]): ...


if sys.platform == "win32":
    from ctypes import WINFUNCTYPE
else:
    def WINFUNCTYPE(
        restype: Type['_CData'] | None,
        *argtypes: Type['_CData'],
        use_errno: bool = False,
        use_last_error: bool = False
    ) -> Type[_FuncPointer]:
        raise NotImplementedError


# ANNO_CONVETIBLE can be used to declare that a class have a `from_param`
# method which can convert other types when used as `argtypes`.
# For example: `CClass = Annotated[bytes, ANNO_CONVERTIBLE, c_class]` means
# `c_class`(name of your class) can convert `bytes` parameters. Then use
# `CClass` as parameter type in stub function declaration and you will get what
# you want.

ANNO_BASIC = object()
ANNO_PARAMETER = ANNO_BASIC
ANNO_RESULT = object() # ANNO_RESULT_ERRCHECK
ANNO_RESULT_CONVERTER = object() # deprecated by ctypes
ANNO_ARRAY = object()
ANNO_POINTER = object()
ANNO_CFUNC = object()
ANNO_WINFUNC = object()
ANNO_PYOBJ = object()


# corresponding annotated python types for ctypes
# use C*Param for parameters, C*Result for returns

CBoolResult = Annotated[bool, ANNO_BASIC, c_bool]
CCharResult = Annotated[bytes, ANNO_BASIC, c_char]
CWcharResult = Annotated[str, ANNO_BASIC, c_wchar]
CByteResult = Annotated[int, ANNO_BASIC, c_byte]
CUbyteResult = Annotated[int, ANNO_BASIC, c_ubyte]
CShortResult = Annotated[int, ANNO_BASIC, c_short]
CUshortResult = Annotated[int, ANNO_BASIC, c_ushort]
CIntResult = Annotated[int, ANNO_BASIC, c_int]
CUintResult = Annotated[int, ANNO_BASIC, c_uint]
CLongResult = Annotated[int, ANNO_BASIC, c_long]
CUlongResult = Annotated[int, ANNO_BASIC, c_ulong]
CLonglongResult = Annotated[int, ANNO_BASIC, c_longlong]
CUlonglongResult = Annotated[int, ANNO_BASIC, c_ulonglong]
CSizeTResult = Annotated[int, ANNO_BASIC, c_size_t]
CSsizeTResult = Annotated[int, ANNO_BASIC, c_ssize_t]
CFloatResult = Annotated[float, ANNO_BASIC, c_float]
CDoubleResult = Annotated[float, ANNO_BASIC, c_double]
CLongdoubleResult = Annotated[float, ANNO_BASIC, c_longdouble]
CCharPResult = Annotated[Optional[bytes], ANNO_BASIC, c_char_p]
CWcharPResult = Annotated[Optional[str], ANNO_BASIC, c_wchar_p]
CVoidPResult = Annotated[Optional[int], ANNO_BASIC, c_void_p]

CBoolParam = Annotated[Union[c_bool, bool], ANNO_BASIC, c_bool]
CCharParam = Annotated[Union[c_char, bytes], ANNO_BASIC, c_char]
CWcharParam = Annotated[Union[c_wchar, str], ANNO_BASIC, c_wchar]
CByteParam = Annotated[Union[c_byte, int], ANNO_BASIC, c_byte]
CUbyteParam = Annotated[Union[c_ubyte, int], ANNO_BASIC, c_ubyte]
CShortParam = Annotated[Union[c_short, int], ANNO_BASIC, c_short]
CUshortParam = Annotated[Union[c_ushort, int], ANNO_BASIC, c_ushort]
CIntParam = Annotated[Union[c_int, int], ANNO_BASIC, c_int]
CUintParam = Annotated[Union[c_uint, int], ANNO_BASIC, c_uint]
CLongParam = Annotated[Union[c_long, int], ANNO_BASIC, c_long]
CUlongParam = Annotated[Union[c_ulong, int], ANNO_BASIC, c_ulong]
CLonglongParam = Annotated[Union[c_longlong, int], ANNO_BASIC, c_longlong]
CUlonglongParam = Annotated[Union[c_ulonglong, int], ANNO_BASIC, c_ulonglong]
CSizeTParam = Annotated[Union[c_size_t, int], ANNO_BASIC, c_size_t]
CSsizeTParam = Annotated[Union[c_ssize_t, int], ANNO_BASIC, c_ssize_t]
CFloatParam = Annotated[Union[c_float, float], ANNO_BASIC, c_float]
CDoubleParam = Annotated[Union[c_double, float], ANNO_BASIC, c_double]
CLongDoubleParam = Annotated[Union[c_longdouble, float], ANNO_BASIC, c_longdouble]
CCharPParam = Annotated[Union[_Array[c_wchar], c_char_p, bytes, None], ANNO_BASIC, c_char_p]
CWcharPParam = Annotated[Union[_Array[c_wchar], c_wchar_p, str, None], ANNO_BASIC, c_wchar_p]
CVoidPParam = Annotated[Union['_CArgObject', _Pointer[Any], _Array[Any], c_void_p, int, None], ANNO_BASIC, c_void_p]

# export Pointer, PointerCompatible, Array and FuncPointer annotation

CArray = Annotated[_Array[AnyCData], ANNO_ARRAY]
CPointer = Annotated[_Pointer[AnyCData], ANNO_POINTER]
CPointerParam = Annotated[_PointerCompatible[AnyCData], ANNO_POINTER]
CFuncPointer = Annotated[_FuncPointer, ANNO_CFUNC]
WinFuncPointer = Annotated[_FuncPointer, ANNO_WINFUNC]
CPyObject = Annotated[_PyObject[_T], ANNO_PYOBJ]


# using decorators to declare errcheck and converter is convenient
# but you will need to use ANNO_RESULT instead if you need delayed evaluation

_Params = ParamSpec('_Params')
_OrigRet = TypeVar('_OrigRet')
_NewRet = TypeVar('_NewRet')

def with_errcheck(checker: Callable[[_OrigRet, Callable[..., _OrigRet], Tuple[Any, ...]], _NewRet]) -> Callable[[Callable[_Params, _OrigRet]], Callable[_Params, _NewRet]]:
    ''' Decorates a stub function with an error checker. '''
    def decorator(wrapped: Callable[_Params, _OrigRet]) -> Callable[_Params, _NewRet]:
        def wrapper(*args: _Params.args, **kwargs: _Params.kwargs) -> _NewRet:
            raise NotImplementedError

        # attach original declaration and error checker to wrapper
        setattr(wrapper, '_decl_errcheck_', (wrapped, checker))
        return wrapper

    return decorator

# NOTE: Actually, converter is a deprecated form of `restype`.
# According to ctypes documentation:
# "It is possible to assign a callable Python object that is not a ctypes
#  type, in this case the function is assumed to return a C int, and the
#  callable will be called with this integer, allowing further processing
#  or error checking. Using this is deprecated, for more flexible post
#  processing or error checking use a ctypes data type as restype and
#  assign a callable to the errcheck attribute."

def with_converter(converter: Callable[[int], _NewRet]) -> Callable[[Callable[_Params, CIntResult]], Callable[_Params, _NewRet]]:
    ''' Decorates a stub function with a converter, its return type MUST be `r_int`. '''
    def decorator(wrapped: Callable[_Params, CIntResult]) -> Callable[_Params, _NewRet]:
        def wrapper(*args: _Params.args, **kwargs: _Params.kwargs) -> _NewRet:
            raise NotImplementedError

        # attach original declaration and converter to wrapper
        setattr(wrapper, '_decl_converter_', (wrapped, converter))
        return wrapper

    return decorator


def convert_annotation(typ: Any, global_ns: Optional[Dict[str, Any]] = None) -> Type[Any]:
    ''' Convert an annotation to effective runtime type. '''
    if global_ns is None:
        global_ns = globals()

    if isinstance(typ, ForwardRef):
        typ = typ.__forward_arg__

    if isinstance(typ, str):
        try: typ = eval(typ, global_ns)
        except Exception as exc:
            raise ValueError('Evaluation of delayed annotation failed!') from exc

    if not hasattr(typ, '__metadata__'):
        return cast(Type[Any], typ)

    # type is Annotated
    ident, *detail = typ.__metadata__

    if ident is ANNO_BASIC:
        ctyp, = detail
        return convert_annotation(ctyp, global_ns=global_ns)

    elif ident is ANNO_RESULT:
        ctyp, _ = detail
        return convert_annotation(ctyp, global_ns=global_ns)

    elif ident is ANNO_RESULT_CONVERTER:
        return c_int

    elif ident is ANNO_ARRAY:
        try: count, = detail
        except ValueError:
            raise ValueError('CArray needs to be annotated with its size')
        ctyp, = typ.__args__[0].__args__
        return cast(Type[Any], convert_annotation(ctyp, global_ns=global_ns) * count)

    elif ident is ANNO_POINTER:
        assert not detail
        ctyp, = typ.__args__[0].__args__
        return POINTER(convert_annotation(ctyp, global_ns=global_ns)) # pyright: ignore

    elif ident is ANNO_CFUNC:
        if not detail:
            raise ValueError('CFuncPointer needs to be annotated with its signature')
        return CFUNCTYPE(*(convert_annotation(t, global_ns=global_ns) for t in detail))

    elif ident is ANNO_WINFUNC:
        if not detail:
            raise ValueError('WinFuncPointer needs to be annotated with its signature')
        return WINFUNCTYPE(*(convert_annotation(t, global_ns=global_ns) for t in detail))

    elif ident is ANNO_PYOBJ:
        assert not detail
        return py_object

    else:
        raise ValueError(f'Unexpected annotated type {typ}')


def get_resconv_info(typ: Any, global_ns: Optional[Dict[str, Any]] = None) -> Optional[Tuple[Any, Any]]:
    if global_ns is None:
        global_ns = globals()

    if isinstance(typ, str):
        try: typ = eval(typ, global_ns)
        except Exception as exc:
            raise ValueError('Evaluation of delayed annotation failed!') from exc

    if not hasattr(typ, '__metadata__'):
        return None
    # type is Annotated
    ident, *detail = typ.__metadata__
    if ident not in (ANNO_RESULT, ANNO_RESULT_CONVERTER):
        return None

    if ident is ANNO_RESULT:
        _, conv = detail
    else:
        conv, = detail
    if isinstance(conv, str):
        conv = eval(conv, global_ns)
    if ident is ANNO_RESULT:
        return (conv, None)
    else:
        return (None, conv)


def convert_func_decl(decl: Callable[..., Any], global_ns: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ''' Converts a stub function to ctypes metadata. '''
    if global_ns is None:
        global_ns = globals()

    result: Dict[str, Any] = {}

    errcheck = None
    converter = None

    while True:
        if hasattr(decl, '_decl_errcheck_'):
            if errcheck is not None or converter is not None:
                raise ValueError('duplicate return conversion specifications, burst your legs')
            decl, errcheck = getattr(decl, '_decl_errcheck_')
            continue

        if hasattr(decl, '_decl_converter_'):
            if errcheck is not None or converter is not None:
                raise ValueError('duplicate return conversion specifications, burst your legs')
            decl, converter = getattr(decl, '_decl_converter_')
            continue

        break

    sig = signature(decl)

    param_annos = [p.annotation for p in sig.parameters.values() if p.name != 'self']
    if all(anno is not Parameter.empty for anno in param_annos):
        result['argtypes'] = [convert_annotation(anno, global_ns=global_ns) for anno in param_annos] or None

    if sig.return_annotation is not Parameter.empty:
        resconv = get_resconv_info(sig.return_annotation, global_ns=global_ns)
        if resconv is not None:
            if errcheck is not None or converter is not None:
                ValueError('duplicate return conversion specifications, burst your legs')
            errcheck, converter = resconv
        result['restype'] = convert_annotation(sig.return_annotation, global_ns=global_ns)

    if errcheck is not None: result['errcheck'] = errcheck
    if converter is not None: result['restype'] = converter

    return result


if TYPE_CHECKING:
    from ctypes import CDLL, WinDLL
_LibDecl = TypeVar('_LibDecl')

def generate_metadata(decl_cls: Type[_LibDecl], global_ns: Optional[Dict[str, Any]] = None) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
    ''' Generate ctypes metadata for a stub class. '''
    if global_ns is None:
        global_ns = globals()

    for name in dir(decl_cls):
        if name.startswith('_'): continue
        value = getattr(decl_cls, name)
        if not callable(value): continue

        yield name, convert_func_decl(value, global_ns=global_ns)

def load_annotated_library(loader: 'Union[CDLL, WinDLL]', decl_cls: Type[_LibDecl], global_ns: Optional[Dict[str, Any]] = None) -> Tuple[_LibDecl, List[str]]:
    ''' Load a library and set signature metadata according to python type hints.
        `decl_cls` is a class which should only contain method declarations.
        Note: you should only name `self` as `self`, the converter depends on this.
    '''
    if global_ns is None:
        global_ns = globals()

    result = decl_cls()
    missing: List[str] = []

    for name, info in generate_metadata(decl_cls, global_ns=global_ns):
        try: func = getattr(loader, name)
        except AttributeError:
            stub = getattr(result, name)
            stub._missing_ = True
            missing.append(name)
            continue

        for attr, infoval in info.items():
            setattr(func, attr, infoval)

        setattr(result, name, func)

    return result, missing


__all__ = [
    'ANNO_PARAMETER',
    'AnyCData',

    'c_bool',
    'c_char',
    'c_wchar',
    'c_byte',
    'c_ubyte',
    'c_short',
    'c_ushort',
    'c_int',
    'c_uint',
    'c_long',
    'c_ulong',
    'c_longlong',
    'c_ulonglong',
    'c_size_t',
    'c_ssize_t',
    'c_float',
    'c_double',
    'c_longdouble',
    'c_char_p',
    'c_wchar_p',
    'c_void_p',
    'py_object',

    'CBoolParam',
    'CCharParam',
    'CWcharParam',
    'CByteParam',
    'CUbyteParam',
    'CShortParam',
    'CUshortParam',
    'CIntParam',
    'CUintParam',
    'CLongParam',
    'CUlongParam',
    'CLonglongParam',
    'CUlonglongParam',
    'CSizeTParam',
    'CSsizeTParam',
    'CFloatParam',
    'CDoubleParam',
    'CLongDoubleParam',
    'CCharPParam',
    'CWcharPParam',
    'CVoidPParam',

    'CBoolResult',
    'CCharResult',
    'CWcharResult',
    'CByteResult',
    'CUbyteResult',
    'CShortResult',
    'CUshortResult',
    'CIntResult',
    'CUintResult',
    'CLongResult',
    'CUlongResult',
    'CLonglongResult',
    'CUlonglongResult',
    'CSizeTResult',
    'CSsizeTResult',
    'CFloatResult',
    'CDoubleResult',
    'CLongdoubleResult',
    'CCharPResult',
    'CWcharPResult',
    'CVoidPResult',

    'CArray',
    'CPointer',
    'CPointerParam',
    'CFuncPointer',
    'WinFuncPointer',
    'CPyObject',

    'convert_annotation',
    'with_errcheck',
    'with_converter',
    'load_annotated_library',
]
