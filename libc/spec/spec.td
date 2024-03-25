class Type {}

class NamedType<string name> : Type {
  string Name = name;
}

class Field<string name, Type type> {
  string Name = name;
  Type FieldType = type;
}

// Class to describe concrete structs specified by a standard.
class Struct<string name> : NamedType<name> {
  list<Field> Fields;
}

class EnumNameValue<string name, string value = "__default_enum_value__"> {
  string Name = name;
  string Value = value;
}

class Enum<string name, list<EnumNameValue> enumerations> : NamedType<name> {
  list<EnumNameValue> Enumerations = enumerations;
}

class PtrType<Type type> : Type {
  Type PointeeType = type;
}

class ConstType<Type type> : Type {
  Type UnqualifiedType = type;
}

class RestrictedPtrType<Type type> : Type {
  Type PointeeType = type;
}

// Builtin types.
def VarArgType : NamedType<"...">;
def VaListType : NamedType<"va_list">;
def VoidType : NamedType<"void">;
def IntType : NamedType<"int">;
def UnsignedIntType : NamedType<"unsigned int">;
def LongType : NamedType<"long">;
def UnsignedLongType : NamedType<"unsigned long">;
def LongLongType : NamedType<"long long">;
def UnsignedLongLongType : NamedType<"unsigned long long">;
def FloatType : NamedType<"float">;
def DoubleType : NamedType<"double">;
def LongDoubleType : NamedType<"long double">;
def CharType : NamedType<"char">;
def UnsignedCharType : NamedType<"unsigned char">;
def UnsignedShortType : NamedType<"unsigned short">;
def BoolType : NamedType<"bool">;

def Float128Type : NamedType<"float128">;

// Common types
def VoidPtr : PtrType<VoidType>;
def VoidPtrPtr : PtrType<VoidPtr>;
def RestrictedVoidPtrPtr : RestrictedPtrType<VoidPtr>;
def ConstVoidPtr : ConstType<VoidPtr>;

def SizeTType : NamedType<"size_t">;
def SizeTPtr : PtrType<SizeTType>;
def RestrictedSizeTPtr : RestrictedPtrType<SizeTType>;

def WCharType : NamedType<"wchar_t">;
def WIntType : NamedType<"wint_t">;

def LongDoublePtr : PtrType<LongDoubleType>;

def IntMaxTType : NamedType<"intmax_t">;
def UIntMaxTType : NamedType<"uintmax_t">;

def UInt16Type : NamedType<"uint16_t">;
def UInt32Type : NamedType<"uint32_t">;

def OffTType : NamedType<"off_t">;
def OffTPtr : PtrType<OffTType>;
def SSizeTType : NamedType<"ssize_t">;

// _Noreturn is really not a type, but it is convenient to treat it as a type.
def NoReturn : NamedType<"_Noreturn void">;

//types moved from stdc.td
def VoidRestrictedPtr : RestrictedPtrType<VoidType>;
def ConstVoidRestrictedPtr : ConstType<VoidRestrictedPtr>;

def CharPtr : PtrType<CharType>;
def ConstCharPtr : ConstType<CharPtr>;
def CharRestrictedPtr : RestrictedPtrType<CharType>;
def CharRestrictedPtrPtr : RestrictedPtrType<CharPtr>;
def ConstCharRestrictedPtr : ConstType<CharRestrictedPtr>;
def ConstCharRestrictedPtrPtr : PtrType<ConstCharRestrictedPtr>;

def OnceFlagType : NamedType<"once_flag">;
def OnceFlagTypePtr : PtrType<OnceFlagType>;
// TODO(sivachandra): Remove this non-standard type when a formal
// way to describe callable types is available.
def CallOnceFuncType : NamedType<"__call_once_func_t">;
def MtxTType : NamedType<"mtx_t">;
def MtxTTypePtr : PtrType<MtxTType>;
def CndTType : NamedType<"cnd_t">;
def CndTTypePtr : PtrType<CndTType>;
def ThrdStartTType : NamedType<"thrd_start_t">;
def ThrdTType : NamedType<"thrd_t">;
def ThrdTTypePtr : PtrType<ThrdTType>;

def IntPtr : PtrType<IntType>;
def RestrictedIntPtr : RestrictedPtrType<IntType>;
def FloatPtr : PtrType<FloatType>;
def DoublePtr : PtrType<DoubleType>;
def Float128Ptr : PtrType<Float128Type>;
def UnsignedCharPtr : PtrType<UnsignedCharType>;

def SigHandlerT : NamedType<"__sighandler_t">;

def TimeTType : NamedType<"time_t">;

def StructTimeSpec : NamedType<"struct timespec">;
def StructTimeSpecPtr : PtrType<StructTimeSpec>;
def ConstStructTimeSpecPtr : ConstType<StructTimeSpecPtr>;

def BSearchCompareT : NamedType<"__bsearchcompare_t">;
def QSortCompareT : NamedType<"__qsortcompare_t">;

def AtexitHandlerT : NamedType<"__atexithandler_t">;

def FILE : NamedType<"FILE">;
def FILEPtr : PtrType<FILE>;
def FILERestrictedPtr : RestrictedPtrType<FILE>;

def PThreadTType : NamedType<"pthread_t">;

def PidT : NamedType<"pid_t">;
def RestrictedPidTPtr : RestrictedPtrType<PidT>;

def StructRUsage : NamedType<"struct rusage">;
def StructRUsagePtr : PtrType<StructRUsage>;

def StructTimevalType : NamedType<"struct timeval">;
def StructTimevalPtr : PtrType<StructTimevalType>;
def RestrictedStructTimevalPtr : RestrictedPtrType<StructTimevalType>;

def SuSecondsT : NamedType<"suseconds_t">;

//added because __assert_fail needs it.
def UnsignedType : NamedType<"unsigned">;

def ActionType : NamedType<"ACTION">;
def EntryType : NamedType<"ENTRY">;
def EntryTypePtr : PtrType<EntryType>;
def EntryTypePtrPtr : PtrType<EntryTypePtr>;

def MBStateTType : NamedType<"mbstate_t">;

class Macro<string name> {
  string Name = name;
}

class EnumeratedNameValue<string name, string value = "__default__"> {
  string Name = name;
  string Value = value;
}

class Annotation {}

class RetValSpec<Type type, list<Annotation> annotations = []> {
  Type ReturnType = type;
  list<Annotation> Annotations = annotations;
}

class ArgSpec<Type type, list<Annotation> annotations = [], string name = ""> {
  Type ArgType = type;
  list<Annotation> Annotations = annotations;
  string Name = name;
}

// The following classes are used to describe function attributes.
// In the future, we may consider supporting parameter attributes as well.
// https://clang.llvm.org/docs/AttributeReference.html
class FunctionAttr<string style, string attr> {
  string Attr = attr;
  // The style of the attribute, e.g. "gnu", "cxx11", "declspec".
  // - "gnu" is for GNU-style attributes: __attribute__((...))
  // - "cxx11" is for C++11-style attributes: [[...]]
  // - "declspec" is for Microsoft-style attributes: __declspec(...)
  string Style = style;

  // For the time being, we are only interested in identifer-like attributes.
  // We can extend this to support function-like attributes if needed.
  // For example, in the future, we can #define __LIBC_ATTRIBUTE_NODISCARD(...) [[nodiscard(__VA_ARGS__)]]
  // int FunctionLike = 0;
}
class GnuFunctionAttr<string attr> : FunctionAttr<"gnu", attr> {}
class Cxx11FunctionAttr<string attr, string namespace> : FunctionAttr<"cxx11", attr> {
  // The namespace of the attribute, e.g. "gnu" or "clang". Empty string means there is no namespace.
  string Namespace = namespace;
}
class DeclspecFunctionAttr<string attr> : FunctionAttr<"declspec", attr> {}
class FunctionAttrSpec<string macro, list<FunctionAttr> instances> {
  list<FunctionAttr> Instances = instances;
  string Macro = macro;
}

class FunctionSpec<string name, RetValSpec return, list<ArgSpec> args, list<FunctionAttrSpec> attrs = []> {
  string Name = name;
  RetValSpec Return = return;
  list<ArgSpec> Args = args;
  list<FunctionAttrSpec> Attributes = attrs;
}

class GuardedFunctionSpec<string name, RetValSpec return, list<ArgSpec> args, string guard_macro> : FunctionSpec<name, return, args> {
  string Guard = guard_macro;
}

class ObjectSpec<string name, string type> {
  string Name = name;
  string Type = type;
}

class HeaderSpec<string name,
                list<Macro> macros = [],
                list<Type> types = [],
                list<EnumeratedNameValue> enumerations = [],
                list<FunctionSpec> functions = [],
                list<ObjectSpec> objects = []> {
  string Name = name;
  list<FunctionSpec> Functions = functions;
  list<Type> Types = types;
  list<Macro> Macros = macros;
  list<EnumeratedNameValue> Enumerations = enumerations;
  list<ObjectSpec> Objects = objects;
}

class StandardSpec<string name> {
  string Name = name;
  list<HeaderSpec> Headers;
}
