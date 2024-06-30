include "llvm/Option/OptParser.td"

multiclass Eq<string name, string help> {
  def NAME#_EQ : Joined<["--", "-"], name#"=">, HelpText<help>;
  def : Separate<["--", "-"], name>, Alias<!cast<Joined>(NAME#_EQ)>;
}

defm build_path : Eq<"p", "Build path.">;
defm extra_arg
    : Eq<"extra-arg",
         "Additional argument to append to the compiler command line.">;
defm extra_arg_before
    : Eq<"extra-arg-before",
         "Additional argument to prepend to the compiler command line.">;
