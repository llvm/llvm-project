include "spec/spec.td"

class TypeDecl<string name> {
  string Name = name;
  string Decl = "";
}

class MacroDef<string name> {
  string Name = name;
  string Defn = "";
}

class SimpleMacroDef<string name, string value> : MacroDef<name> {
  let Defn = !strconcat("#define ", name, " ", value);
}

class MacroDefineIfNot<string name, string value> : MacroDef<name> {
  let Defn = !strconcat("#ifndef ", name, "\n",
                        "#define " , name, " ", value, "\n",
                        "#endif // ", name);
}

class PublicAPI<string name> {
  string HeaderName = name;
  list<MacroDef> Macros = [];
  list<TypeDecl> TypeDeclarations = [];
  list<string> Enumerations = [];
  list<string> Structs = [];
  list<string> Functions = [];
}
