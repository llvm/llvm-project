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
def VarArgType : Type {}
def VoidType : NamedType<"void">;
def IntType : NamedType<"int">;
def FloatType : NamedType<"float">;
def DoubleType : NamedType<"double">;
def LongDoubleType : NamedType<"long double">;
def CharType : NamedType<"char">;

// Common types
def VoidPtr : PtrType<VoidType>;
def ConstVoidPtr : ConstType<VoidPtr>;
def SizeTType : NamedType<"size_t">;
def FloatPtr : PtrType<FloatType>;
def LongDoublePtr : PtrType<LongDoubleType>;

// _Noreturn is really not a type, but it is convenient to treat it as a type.
def NoReturn : NamedType<"_Noreturn void">;

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

class HeaderSpec<string name,
                list<Macro> macros = [],
                list<Type> types = [],
                list<EnumeratedNameValue> enumerations = [],
                list<FunctionSpec> functions = []> {
  string Name = name;
  list<FunctionSpec> Functions = functions;
  list<Type> Types = types;
  list<Macro> Macros = macros;
  list<EnumeratedNameValue> Enumerations = enumerations;
}

class StandardSpec<string name> {
  string Name = name;
  list<HeaderSpec> Headers;
}
