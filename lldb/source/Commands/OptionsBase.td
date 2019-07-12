// Base class for all options.
class Option<string fullname, string shortname> {
  string FullName = fullname;
  string ShortName = shortname;
  // The full associated command/subcommand such as "settings set".
  string Command;
}

// Moves the option into a list of option groups.
class Groups<list<int> groups> {
  list<int> Groups = groups;
}

// Moves the option in all option groups in a range.
// Start and end values are inclusive.
class GroupRange<int start, int end> {
  int GroupStart = start;
  int GroupEnd = end;
}
// Moves the option in a single option group.
class Group<int group> {
  int GroupStart = group;
  int GroupEnd = group;
}

// Sets the description for the option that should be
// displayed to the user.
class Desc<string description> {
  string Description = description;
}

// Marks the option as required when calling the
// associated command.
class Required {
  bit Required = 1;
}

// Gives the option an optional argument.
class OptionalArg<string type> {
  string ArgType = type;
  bit OptionalArg = 1;
}

// Gives the option an required argument.
class Arg<string type> {
  string ArgType = type;
}

// Gives the option an required argument.
class EnumArg<string type, string enum> {
  string ArgType = type;
  string ArgEnum = enum;
}

// Sets the available completions for the given option.
class Completions<list<string> completions> {
  list<string> Completions = completions;
}
// Sets a single completion for the given option.
class Completion<string completion> {
  list<string> Completions = [completion];
}
