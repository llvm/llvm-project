# Clang Plugins

Clang Plugins make it possible to run extra user defined actions during a
compilation. This document will provide a basic walkthrough of how to write and
run a Clang Plugin.

## Introduction

Clang Plugins run FrontendActions over code. See the {doc}`FrontendAction
tutorial <RAVFrontendAction>` on how to write a `FrontendAction` using the
`RecursiveASTVisitor`. In this tutorial, we'll demonstrate how to write a
simple clang plugin.

## Writing a `PluginASTAction`

The main difference from writing normal `FrontendActions` is that you can
handle plugin command line options. The `PluginASTAction` base class declares
a `ParseArgs` method which you have to implement in your plugin.

```c++
bool ParseArgs(const CompilerInstance &CI,
               const std::vector<std::string>& args) {
  for (unsigned i = 0, e = args.size(); i != e; ++i) {
    if (args[i] == "-some-arg") {
      // Handle the command line argument.
    }
  }
  return true;
}
```

## Registering a plugin

A plugin is loaded from a dynamic library at runtime by the compiler. To
register a plugin in a library, use `FrontendPluginRegistry::Add<>`:

```c++
static FrontendPluginRegistry::Add<MyPlugin> X("my-plugin-name", "my plugin description");
```

## Defining pragmas

Plugins can also define pragmas by declaring a `PragmaHandler` and
registering it using `PragmaHandlerRegistry::Add<>`:

```c++
// Define a pragma handler for #pragma example_pragma
class ExamplePragmaHandler : public PragmaHandler {
public:
  ExamplePragmaHandler() : PragmaHandler("example_pragma") { }
  void HandlePragma(Preprocessor &PP, PragmaIntroducer Introducer,
                    Token &PragmaTok) {
    // Handle the pragma
  }
};

static PragmaHandlerRegistry::Add<ExamplePragmaHandler> Y("example_pragma","example pragma description");
```

## Defining attributes

Plugins can define attributes by declaring a `ParsedAttrInfo` and registering
it using `ParsedAttrInfoRegister::Add<>`:

```c++
class ExampleAttrInfo : public ParsedAttrInfo {
public:
  ExampleAttrInfo() {
    Spellings.push_back({ParsedAttr::AS_GNU,"example"});
  }
  AttrHandling handleDeclAttribute(Sema &S, Decl *D,
                                   const ParsedAttr &Attr) const override {
    // Handle the attribute
    return AttributeApplied;
  }
};

static ParsedAttrInfoRegistry::Add<ExampleAttrInfo> Z("example_attr","example attribute description");
```

The members of `ParsedAttrInfo` that a plugin attribute must define are:

- `Spellings`, which must be populated with every [Spelling](https://clang.llvm.org/doxygen/structclang_1_1ParsedAttrInfo_1_1Spelling.html) of the
  attribute, each of which consists of an attribute syntax and how the
  attribute name is spelled for that syntax. If the syntax allows a scope then
  the spelling must be "scope::attr" if a scope is present or "::attr" if not.

The members of `ParsedAttrInfo` that may need to be defined, depending on the
attribute, are:

- `NumArgs` and `OptArgs`, which set the number of required and optional
  arguments to the attribute.
- `diagAppertainsToDecl`, which checks if the attribute has been used on the
  right kind of declaration and issues a diagnostic if not.
- `handleDeclAttribute`, which is the function that applies the attribute to
  a declaration. It is responsible for checking that the attribute's arguments
  are valid, and typically applies the attribute by adding an `Attr` to the
  `Decl`. It returns either `AttributeApplied`, to indicate that the
  attribute was successfully applied, or `AttributeNotApplied` if it wasn't.
- `diagAppertainsToStmt`, which checks if the attribute has been used on the
  right kind of statement and issues a diagnostic if not.
- `handleStmtAttribute`, which is the function that applies the attribute to
  a statement. It is responsible for checking that the attribute's arguments
  are valid, and typically applies the attribute by adding an `Attr` to the
  `Stmt`. It returns either `AttributeApplied`, to indicate that the
  attribute was successfully applied, or `AttributeNotApplied` if it wasn't.
- `diagLangOpts`, which checks if the attribute is permitted for the current
  language mode and issues a diagnostic if not.
- `existsInTarget`, which checks if the attribute is permitted for the given
  target.

To see a working example of an attribute plugin, see [the Attribute.cpp example](https://github.com/llvm/llvm-project/blob/main/clang/examples/Attribute/Attribute.cpp).

## Emitting diagnostics

A plugin emits diagnostics through the `DiagnosticsEngine` obtained from the
`CompilerInstance`. Calling `getCustomPluginDiagID` with the plugin's own name
places the diagnostic in that plugin's warning group, so users can control it
with `-W` flags exactly like a built-in warning. The group is named
`<plugin>-plugin` (for the plugin registered under `<plugin>`); clang registers
that group for every loaded plugin. Deriving the group from the plugin's name
keeps every plugin in its own namespace, so two plugins can never collide on a
group and a plugin diagnostic is never left in a name no `-W` flag reaches.

```c++
  DiagnosticsEngine &D = CI.getDiagnostics();
  unsigned ID = D.getCustomPluginDiagID(
      DiagnosticsEngine::Warning, "my plugin found something odd about '%0'",
      "print-fns");
  D.Report(Loc, ID) << Name;
```

The warning is on by default, prints `[-Wprint-fns-plugin]` so users can see how
to control it, and can be:

- silenced with `-Wno-print-fns-plugin`, with the `-Wno-plugin` umbrella that
  covers every loaded plugin, or with `-Wno-user-defined-warnings` which is the
  root over every runtime group (it also covers `diagnose_if`), and
- turned into an error with `-Werror=print-fns-plugin`.

The same grouping applies to remarks and errors. A remark
(`DiagnosticsEngine::Remark`) placed in a group is off by default, like a
built-in remark, and is opted into with `-R<group>`; `-W` flags do not affect
it. An error keeps its severity regardless of any group flag, so a
`-Wno-<group>` can never silence a grouped error.

A plugin may further split its diagnostics into subgroups by passing a subgroup
name as the final argument to `getCustomPluginDiagID`, producing
`<plugin>-plugin-<subgroup>`; each subgroup is controlled by its own name, by the
`<plugin>-plugin` group, and by the `-Wplugin` umbrella, with the most specific
flag winning. A `-W<plugin>-plugin` flag that no loaded plugin claims is reported
as an unknown warning option once all plugins have loaded, so misspellings are
still caught.

By convention plugin diagnostics live in their own `<plugin>-plugin` namespace
under `-Wplugin`, which in turn nests under `-Wuser-defined-warnings`.
`getCustomPluginDiagID` is a thin convenience over the general primitive

```c++
  unsigned getCustomDiagID(Level, StringRef Message, StringRef Group);
```

which places a diagnostic in any warning group named by `Group`. `Group` may
also be an existing built-in group, so a plugin that deliberately wants to
extend, say, `-Wdeprecated` can do so; the diagnostic is then controlled by that
group's flag like any other member. Prefer the `<plugin>-plugin` convention
unless there is a specific reason to join a built-in group.

A backend (IR-layer) plugin controls its diagnostics the same way. An
`llvm::DiagnosticInfo` that overrides `getWarningGroup()` to name a group is
routed by clang through the same mechanism, so a pass plugin naming
`<plugin>-plugin` gets the same `-W` control and nests under `-Wplugin` and
`-Wuser-defined-warnings`. A backend diagnostic that names no group falls under
`-Wbackend-plugin` as before.

## Organizing diagnostics like the compiler does

A plugin with more than a handful of diagnostics can organize them the way Clang
organizes its own: as a table, with a stable name per diagnostic that doubles as
its SARIF `ruleId`. Clang's diagnostics are defined as TableGen records in `.td`
files, from which `clang-tblgen -gen-clang-diags-defs` generates a table of
`DIAG(...)` rows. A plugin can do the same and register the generated table at
runtime with `getCustomPluginDiagIDs`:

```c++
  static const DiagnosticsEngine::PluginDiagnostic Table[] = {
    {"suspicious_decl", DiagnosticsEngine::Warning, "suspicious %0", ""},
    {"forbidden_decl",  DiagnosticsEngine::Error,   "forbidden %0",  ""},
  };
  llvm::SmallVector<unsigned> IDs =
      CI.getDiagnostics().getCustomPluginDiagIDs("my-plugin", Table);
```

Each entry lands in the plugin's `my-plugin-plugin` group and gets a stable
`ruleId` `my_plugin_<record>` (both derived from the names, so the plugin spells
neither). The returned IDs are in table order, so a parallel enumeration
generated from the same table gives type-safe call sites, exactly like Clang's
`diag::warn_*`. The `PrintFunctionNames` example shows this end to end,
hand-writing the table with an X-macro that a real plugin would instead generate
from a `.td`. Moving a plugin's diagnostics under its own TableGen this way is a
drop-in: the group names, the `-W` control, and the SARIF `ruleId` all stay the
same, so users and tooling see no churn.

## Putting it all together

Let's look at an example plugin that prints top-level function names. This
example is checked into the clang repository; please take a look at
the [latest version of PrintFunctionNames.cpp](https://github.com/llvm/llvm-project/blob/main/clang/examples/PrintFunctionNames/PrintFunctionNames.cpp).

## Running the plugin

### Using the compiler driver

The Clang driver accepts the `-fplugin` option to load a plugin.
Clang plugins can receive arguments from the compiler driver command
line via the `fplugin-arg-<plugin name>-<argument>` option. Using this
method, the plugin name cannot contain dashes itself, but the argument
passed to the plugin can.

```console
$ export BD=/path/to/build/directory
$ make -C $BD CallSuperAttr
$ clang++ -fplugin=$BD/lib/CallSuperAttr.so \
          -fplugin-arg-call_super_plugin-help \
          test.cpp
```

If your plugin name contains dashes, either rename the plugin or use the
cc1 command line options listed below.

### Using the cc1 command line

To run a plugin, the dynamic library containing the plugin registry must be
loaded via the `-load` command line option. This will load all plugins
that are registered, and you can select the plugins to run by specifying the
`-plugin` option. Additional parameters for the plugins can be passed with
`-plugin-arg-<plugin-name>`.

Note that those options must reach clang's cc1 process. There are two
ways to do so:

- Directly call the parsing process by using the `-cc1` option; this
  has the downside of not configuring the default header search paths, so
  you'll need to specify the full system path configuration on the command
  line.
- Use clang as usual, but prefix all arguments to the cc1 process with
  `-Xclang`.

For example, to run the `print-function-names` plugin over a source file in
clang, first build the plugin, and then call clang with the plugin from the
source tree:

```console
$ export BD=/path/to/build/directory
$ (cd $BD && make PrintFunctionNames )
$ clang++ -D_GNU_SOURCE -D_DEBUG -D__STDC_CONSTANT_MACROS \
          -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS -D_GNU_SOURCE \
          -I$BD/tools/clang/include -Itools/clang/include -I$BD/include -Iinclude \
          tools/clang/tools/clang-check/ClangCheck.cpp -fsyntax-only \
          -Xclang -load -Xclang $BD/lib/PrintFunctionNames.so -Xclang \
          -plugin -Xclang print-fns
```

Also see the print-function-name plugin example's
[README](https://github.com/llvm/llvm-project/blob/main/clang/examples/PrintFunctionNames/README.txt)

### Using the clang command line

Using `-fplugin=plugin` on the clang command line passes the plugin
through as an argument to `-load` on the cc1 command line. If the plugin
class implements the `getActionType` method then the plugin is run
automatically. For example, to run the plugin automatically after the main AST
action (i.e. the same as using `-add-plugin`):

```c++
// Automatically run the plugin after the main AST action
PluginASTAction::ActionType getActionType() override {
  return AddAfterMainAction;
}
```

### Interaction with `-clear-ast-before-backend`

To reduce peak memory usage of the compiler, plugins are recommended to run
*before* the main action, which is usually code generation. This is because
having any plugins that run after the codegen action automatically turns off
`-clear-ast-before-backend`. `-clear-ast-before-backend` reduces peak
memory by clearing the Clang AST after generating IR and before running IR
optimizations. Use `CmdlineBeforeMainAction` or `AddBeforeMainAction` as
`getActionType` to run plugins while still benefitting from
`-clear-ast-before-backend`. Plugins must make sure not to modify the AST,
otherwise they should run after the main action.
