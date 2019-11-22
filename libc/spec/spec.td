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

class Macro<string name> {
  string Name = name;
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
                list<Macro> macros,
                list<Type> types,
                list<FunctionSpec> functions> {
  string Name = name;
  list<FunctionSpec> Functions = functions;
  list<Type> Types = types;
  list<Macro> Macros = macros;
}

class StandardSpec<string name> {
  string Name = name;
  list<HeaderSpec> Headers;
}
