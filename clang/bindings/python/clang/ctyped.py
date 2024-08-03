from ctypes import (CFUNCTYPE, POINTER, WINFUNCTYPE, c_bool, c_byte, c_char,
                    c_char_p, c_double, c_float, c_int, c_long, c_longdouble,
                    c_longlong, c_short, c_size_t, c_ssize_t, c_ubyte, c_uint,
                    c_ulong, c_ulonglong, c_ushort, c_void_p, c_wchar,
                    c_wchar_p, py_object)
from inspect import Parameter, signature
from typing import (TYPE_CHECKING, Any, Callable, Dict, Generator, Generic,
                    List, Optional, Tuple, Type, TypeVar, Union, cast)

from typing_extensions import Annotated, ParamSpec, TypeAlias

_T = TypeVar('_T')

if TYPE_CHECKING:
    from ctypes import _CArgObject  # pyright: ignore[reportPrivateUsage]
    from ctypes import _CData  # pyright: ignore[reportPrivateUsage]

AnyCData = TypeVar('AnyCData', bound='_CData')

if TYPE_CHECKING:
    from ctypes import Array as _Array  # pyright: ignore[reportPrivateUsage]
    from ctypes import \
        _FuncPointer as _FuncPointer  # pyright: ignore[reportPrivateUsage]
    from ctypes import \
        _Pointer as _Pointer  # pyright: ignore[reportPrivateUsage]

    # ctypes documentation noted implicit conversion for pointers:
    # "For example, you can pass compatible array instances instead of pointer
    #  types. So, for POINTER(c_int), ctypes accepts an array of c_int:"
    # "In addition, if a function argument is explicitly declared to be a
    #  pointer type (such as POINTER(c_int)) in argtypes, an object of the
    #  pointed type (c_int in this case) can be passed to the function. ctypes
    #  will apply the required byref() conversion in this case automatically."
    # also, current ctype typeshed thinks byref returns _CArgObject
    _PointerCompatible: TypeAlias = Union['_CArgObject', _Pointer[AnyCData], None, _Array[AnyCData], AnyCData]
    _PyObject: TypeAlias = Union['py_object[_T]', _T]
else:
    # at runtime we don't really import those symbols
    class _Array(Generic[AnyCData]): ...
    class _Pointer(Generic[AnyCData]): ...
    class _PointerCompatible(Generic[AnyCData]): ...
    class _FuncPointer: ...
    class _PyObject(Generic[AnyCData]): ...


# ANNO_CONVETIBLE can be used to declare that a class have a `from_param`
# method which can convert other types when used as `argtypes`.
# For example: `CClass = Annotated[bytes, ANNO_CONVERTIBLE, c_class]` means
# `c_class`(name of your class) can convert `bytes` parameters. Then use
# `CClass` as parameter type in stub function declaration and you will get what
# you want.

ANNO_CONVERTIBLE = object()
ANNO_BASIC = ANNO_CONVERTIBLE
ANNO_ARRAY = object()
ANNO_POINTER = object()
ANNO_CFUNC = object()
ANNO_WINFUNC = object()
ANNO_PYOBJ = object()


# corresponding annotated python types for ctypes
# use p_* for parameters, r_* for returns

r_bool = Annotated[bool, ANNO_BASIC, c_bool]
r_char = Annotated[bytes, ANNO_BASIC, c_char]
r_wchar = Annotated[str, ANNO_BASIC, c_wchar]
r_byte = Annotated[int, ANNO_BASIC, c_byte]
r_ubyte = Annotated[int, ANNO_BASIC, c_ubyte]
r_short = Annotated[int, ANNO_BASIC, c_short]
r_ushort = Annotated[int, ANNO_BASIC, c_ushort]
r_int = Annotated[int, ANNO_BASIC, c_int]
r_uint = Annotated[int, ANNO_BASIC, c_uint]
r_long = Annotated[int, ANNO_BASIC, c_long]
r_ulong = Annotated[int, ANNO_BASIC, c_ulong]
r_longlong = Annotated[int, ANNO_BASIC, c_longlong]
r_ulonglong = Annotated[int, ANNO_BASIC, c_ulonglong]
r_size_t = Annotated[int, ANNO_BASIC, c_size_t]
r_ssize_t = Annotated[int, ANNO_BASIC, c_ssize_t]
r_float = Annotated[float, ANNO_BASIC, c_float]
r_double = Annotated[float, ANNO_BASIC, c_double]
r_longdouble = Annotated[float, ANNO_BASIC, c_longdouble]
r_char_p = Annotated[Optional[bytes], ANNO_BASIC, c_char_p]
r_wchar_p = Annotated[Optional[str], ANNO_BASIC, c_wchar_p]
r_void_p = Annotated[Optional[int], ANNO_BASIC, c_void_p]

p_bool = Annotated[Union[c_bool, bool], ANNO_BASIC, c_bool]
p_char = Annotated[Union[c_char, bytes], ANNO_BASIC, c_char]
p_wchar = Annotated[Union[c_wchar, str], ANNO_BASIC, c_wchar]
p_byte = Annotated[Union[c_byte, int], ANNO_BASIC, c_byte]
p_ubyte = Annotated[Union[c_ubyte, int], ANNO_BASIC, c_ubyte]
p_short = Annotated[Union[c_short, int], ANNO_BASIC, c_short]
p_ushort = Annotated[Union[c_ushort, int], ANNO_BASIC, c_ushort]
p_int = Annotated[Union[c_int, int], ANNO_BASIC, c_int]
p_uint = Annotated[Union[c_uint, int], ANNO_BASIC, c_uint]
p_long = Annotated[Union[c_long, int], ANNO_BASIC, c_long]
p_ulong = Annotated[Union[c_ulong, int], ANNO_BASIC, c_ulong]
p_longlong = Annotated[Union[c_longlong, int], ANNO_BASIC, c_longlong]
p_ulonglong = Annotated[Union[c_ulonglong, int], ANNO_BASIC, c_ulonglong]
p_size_t = Annotated[Union[c_size_t, int], ANNO_BASIC, c_size_t]
p_ssize_t = Annotated[Union[c_ssize_t, int], ANNO_BASIC, c_ssize_t]
p_float = Annotated[Union[c_float, float], ANNO_BASIC, c_float]
p_double = Annotated[Union[c_double, float], ANNO_BASIC, c_double]
p_longdouble = Annotated[Union[c_longdouble, float], ANNO_BASIC, c_longdouble]
p_char_p = Annotated[Union[c_char_p, _Array[c_wchar], bytes, None], ANNO_BASIC, c_char_p]
p_wchar_p = Annotated[Union[c_wchar_p, _Array[c_wchar], str, None], ANNO_BASIC, c_wchar_p]
p_void_p = Annotated[Union['_CArgObject', c_void_p, int, None], ANNO_BASIC, c_void_p]

# export Pointer, PointerCompatible, Array and FuncPointer annotation

CArray = Annotated[_Array[AnyCData], ANNO_ARRAY]
CPointer = Annotated[_Pointer[AnyCData], ANNO_POINTER]
CPointerParam = Annotated[_PointerCompatible[AnyCData], ANNO_POINTER]
CFuncPointer = Annotated[_FuncPointer, ANNO_CFUNC]
WinFuncPointer = Annotated[_FuncPointer, ANNO_WINFUNC]
CPyObject = Annotated[_PyObject[_T], ANNO_PYOBJ]


_Params = ParamSpec('_Params')
_OrigRet = TypeVar('_OrigRet')
_NewRet = TypeVar('_NewRet')

def with_errcheck(checker: Callable[[_OrigRet, Callable[..., _OrigRet], Tuple[Any, ...]], _NewRet]) -> Callable[[Callable[_Params, _OrigRet]], Callable[_Params, _NewRet]]:
    ''' Decorates a stub function with an error checker. '''
    def decorator(wrapped: Callable[_Params, _OrigRet]) -> Callable[_Params, _NewRet]:
        def wrapper(*args: _Params.args, **kwargs: _Params.kwargs) -> _NewRet:
            return cast(_NewRet, None)  # make type checker happy

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

def with_converter(converter: Callable[[int], _NewRet]) -> Callable[[Callable[_Params, r_int]], Callable[_Params, _NewRet]]:
    ''' Decorates a stub function with a converter, its return type MUST be `r_int`. '''
    def decorator(wrapped: Callable[_Params, r_int]) -> Callable[_Params, _NewRet]:
        def wrapper(*args: _Params.args, **kwargs: _Params.kwargs) -> _NewRet:
            return cast(_NewRet, None)  # make type checker happy

        # attach original declaration and converter to wrapper
        setattr(wrapper, '_decl_converter_', (wrapped, converter))
        return wrapper

    return decorator


def convert_annotation(typ: Any, global_ns: Optional[Dict[str, Any]] = None) -> Type[Any]:
    ''' Convert an annotation to effective runtime type. '''
    if global_ns is None:
        global_ns = globals()

    if isinstance(typ, str):
        try: typ = eval(typ, global_ns)
        except Exception as exc:
            raise ValueError('Evaluation of delayed annotation failed!') from exc

    if not hasattr(typ, '__metadata__'):
        return cast(Type[Any], typ)

    # type is Annotated
    ident, *detail = typ.__metadata__
    if ident is ANNO_CONVERTIBLE:
        ctyp, = detail
        return cast(Type[Any], ctyp)
    elif ident is ANNO_ARRAY:
        try: count, = detail
        except ValueError:
            raise ValueError('CArray needs to be annotated with its size')
        ctyp, = typ.__args__[0].__args__
        return cast(Type[Any], convert_annotation(ctyp, global_ns) * count)
    elif ident is ANNO_POINTER:
        assert not detail
        ctyp, = typ.__args__[0].__args__
        return POINTER(convert_annotation(ctyp, global_ns)) # pyright: ignore
    elif ident is ANNO_CFUNC:
        if not detail:
            raise ValueError('CFuncPointer needs to be annotated with its signature')
        return CFUNCTYPE(*(convert_annotation(t, global_ns) for t in detail))
    elif ident is ANNO_WINFUNC:
        if not detail:
            raise ValueError('WinFuncPointer needs to be annotated with its signature')
        return WINFUNCTYPE(*(convert_annotation(t, global_ns) for t in detail))
    elif ident is ANNO_PYOBJ:
        assert not detail
        return py_object
    else:
        raise ValueError(f'Unexpected annotated type {typ}')


def convert_func_decl(decl: Callable[..., Any], errcheck: Any = None, converter: Any = None, global_ns: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ''' Converts a stub function to ctypes metadata. '''
    if global_ns is None:
        global_ns = globals()

    result: Dict[str, Any] = {}

    if hasattr(decl, '_decl_errcheck_'):
        rdecl, errcheck = getattr(decl, '_decl_errcheck_')
        return convert_func_decl(rdecl, errcheck, converter, global_ns)
    
    if hasattr(decl, '_decl_converter_'):
        rdecl, converter = getattr(decl, '_decl_converter_')
        return convert_func_decl(rdecl, errcheck, converter, global_ns)
    
    sig = signature(decl)

    param_annos = [p.annotation for p in sig.parameters.values() if p.name != 'self']
    if all(anno is not Parameter.empty for anno in param_annos):
        result['argtypes'] = [convert_annotation(anno, global_ns) for anno in param_annos] or None

    if sig.return_annotation is not Parameter.empty:
        result['restype'] = convert_annotation(sig.return_annotation, global_ns)

    if errcheck is not None: result['errcheck'] = errcheck
    if converter is not None: result['restype'] = converter

    return result


if TYPE_CHECKING:
    from ctypes import CDLL, WinDLL
    _DLLT = TypeVar('_DLLT', bound=CDLL)
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

    for name, info in generate_metadata(decl_cls, global_ns):
        try: func = getattr(loader, name)
        except AttributeError:
            stub = getattr(result, name)
            stub._missing_ = True
            missing.append(name); continue

        for attr, infoval in info.items():
            setattr(func, attr, infoval)

        setattr(result, name, func)

    return result, missing


__all__ = [
    'ANNO_CONVERTIBLE',
    'AnyCData',

    'p_bool',
    'p_char',
    'p_wchar',
    'p_byte',
    'p_ubyte',
    'p_short',
    'p_ushort',
    'p_int',
    'p_uint',
    'p_long',
    'p_ulong',
    'p_longlong',
    'p_ulonglong',
    'p_size_t',
    'p_ssize_t',
    'p_float',
    'p_double',
    'p_longdouble',
    'p_char_p',
    'p_wchar_p',
    'p_void_p',

    'r_bool',
    'r_char',
    'r_wchar',
    'r_byte',
    'r_ubyte',
    'r_short',
    'r_ushort',
    'r_int',
    'r_uint',
    'r_long',
    'r_ulong',
    'r_longlong',
    'r_ulonglong',
    'r_size_t',
    'r_ssize_t',
    'r_float',
    'r_double',
    'r_longdouble',
    'r_char_p',
    'r_wchar_p',
    'r_void_p',

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
