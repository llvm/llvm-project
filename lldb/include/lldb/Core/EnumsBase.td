class EnumValue<string value, string name, string description> {
  /// The suffix of the enumerator.
  /// For example `Public` for a prefix of `eAccessType` (set in `EnumDef`).
  string Value = value;
  /// The user-visible name of this value. Usually kebab-case.
  string Name = name;
  /// A description for this value.
  string Description = description;
}

/// Usage:
/// ```
/// def script_languages : EnumDef<"lldb::eScriptLanguage", [
///   EnumValue<"Python", "python", "something about Python">,
///   EnumValue<"Lua", "lua", "something about Lua">,
///   // ...
/// ]>;
/// ```
/// This will result in a `g_script_languages` definition.
class EnumDef<string prefix, list<EnumValue> values> {
  /// Prefix prepended to all `Value`s in `Values`.
  /// For example, if the prefix is `lldb::eScriptLanguage`, then
  /// `EnumValue<"Python", "python", "..">` will result in
  /// `lldb::eScriptLanguagePython` being added.
  string Prefix = prefix;

  list<EnumValue> Values = values;
}
