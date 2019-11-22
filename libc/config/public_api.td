include "spec/spec.td"

class TypeDecl<string name> {
  string Name = name;
  string Decl = "";
}

class MacroDef<string name> {
  string Name = name;
  string Defn = "";
}

class PublicAPI<string name> {
  string HeaderName = name;
  list<MacroDef> Macros = [];
  list<TypeDecl> TypeDeclarations = [];
  list<string> Structs = [];
  list<string> Functions = [];
}
