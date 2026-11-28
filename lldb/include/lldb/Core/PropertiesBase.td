// Base class for all options.
class Property<string name, string type> {
  string Name = name;
  string Type = type;
  string Definition;
  string Path;
}

// Sets the description for the property that should be displayed to the user.
class Desc<string description> {
  string Description = description;
}

// Marks the property as global.
class Global {
  bit Global = 1;
}

// Marks the property as deprecated. Deprecated properties can still be set and
// shown by name, but setting them prints a warning, and they are tagged as
// deprecated wherever they are shown. Unless deprecated settings are explicitly
// requested, they are hidden from "settings list", and from "settings show" and
// "settings write" while they have their default value. The message is appended
// to the deprecation warning, and should tell users what to set instead. Like
// any diagnostic, it should start in lowercase and have no trailing period.
class Deprecated<string message> {
  bit Deprecated = 1;
  string DeprecationMessage = message;
}

class DefaultTrue {
  int DefaultUnsignedValue = 1;
  bit HasDefaultUnsignedValue = 1;
  bit HasDefaultBooleanValue = 1;
}

class DefaultFalse {
  int DefaultUnsignedValue = 0;
  bit HasDefaultUnsignedValue = 1;
  bit HasDefaultBooleanValue = 1;
}

// Gives the property a default string value.
class DefaultStringValue<string value> {
  string DefaultStringValue = value;
  bit HasDefaultStringValue = 1;
}

// Gives the property a default enum value.
class DefaultEnumValue<string value> {
  string DefaultEnumValue = value;
  bit HasDefaultEnumValue = 1;
}

// Gives the property a default string value.
class DefaultUnsignedValue<int value> {
  int DefaultUnsignedValue = value;
  bit HasDefaultUnsignedValue = 1;
}

// Gives the property enum values.
class EnumValues<string enum> {
  string EnumValues = enum;
}

// Determines the element type for arrays and dictionaries.
class ElementType<string value> {
  string ElementType = value;
  bit HasElementType = 1;
}
