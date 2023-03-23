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

def SigHandlerT : NamedType<"__sighandler_t">;

def TimeTType : NamedType<"time_t">;

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

class FunctionSpec<string name, RetValSpec return, list<ArgSpec> args> {
  string Name = name;
  RetValSpec Return = return;
  list<ArgSpec> Args = args;
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
