include "llvm/Option/LibraryOptions.td"

def ClangDiffOptions : OptionsStruct<"">;

defm : BoolField<"ast-dump", "false", "Print the internal representation of the AST">;
defm : BoolField<"ast-dump-json", "false", "Print the internal representation of the AST as JSON">;
defm : BoolField<"dump-matches", "false", "Print the matched nodes">;
defm : RepeatedField<"extra-arg", "StringRef", "Additional argument to append to the compiler command line">, MetaVarName<"<arg>">;
defm : RepeatedField<"extra-arg-before", "StringRef", "Additional argument to prepend to the compiler command line">, MetaVarName<"<arg>">;
defm : BoolField<"help", "false", "Display available options">;
defm : BoolField<"html", "false", "Output a side-by-side diff in HTML">;
defm : ValueField<"p", "StringRef", "", "Build path">, MetaVarName<"<build-path>">;
defm : OptionalValueField<"s", "int", "Maximum subtree size for the optimal bottom-up mapping">, MetaVarName<"<maxsize>">;
defm stop_after_topdown : EnumField<"stop-diff-after", "bool", "false", "topdown,bottomup", ["true", "false"], "Stop after the given matching phase">;
