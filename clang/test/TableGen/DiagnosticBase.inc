// Define the diagnostic mappings.
class DiagMapping;
def MAP_IGNORE  : DiagMapping;
def MAP_WARNING : DiagMapping;
def MAP_ERROR   : DiagMapping;
def MAP_FATAL   : DiagMapping;

// Define the diagnostic classes.
class DiagClass;
def CLASS_NOTE      : DiagClass;
def CLASS_WARNING   : DiagClass;
def CLASS_EXTENSION : DiagClass;
def CLASS_ERROR     : DiagClass;

class DiagGroup<string Name, list<DiagGroup> subgroups = []> {
  string GroupName = Name;
  list<DiagGroup> SubGroups = subgroups;
  string CategoryName = "";
}
class InGroup<DiagGroup G> { DiagGroup Group = G; }

// All diagnostics emitted by the compiler are an indirect subclass of this.
class Diagnostic<string text, DiagClass DC, DiagMapping defaultmapping> {
  string      Text = text;
  DiagClass   Class = DC;
  DiagMapping DefaultMapping = defaultmapping;
  DiagGroup   Group;
  string      CategoryName = "";
}

class Error<string str>     : Diagnostic<str, CLASS_ERROR, MAP_ERROR>;
class Warning<string str>   : Diagnostic<str, CLASS_WARNING, MAP_WARNING>;
class Extension<string str> : Diagnostic<str, CLASS_EXTENSION, MAP_IGNORE>;
class ExtWarn<string str>   : Diagnostic<str, CLASS_EXTENSION, MAP_WARNING>;
class Note<string str>      : Diagnostic<str, CLASS_NOTE, MAP_FATAL/*ignored*/>;
