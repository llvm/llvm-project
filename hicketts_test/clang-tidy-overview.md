# How clang-tidy Works

## Entry Point

`clang-tools-extra/clang-tidy/tool/ClangTidyMain.cpp` — parses CLI args (`--checks`, `--fix`, etc.), builds an options provider, and calls `runClangTidy()`.

## Module & Check Registration

Uses LLVM's static registry pattern:

1. **Modules** (e.g. `BugproneModule`, `ModernizeModule`) subclass `ClangTidyModule` and register themselves statically via `ClangTidyModuleRegistry::Add<T>`.
2. Each module's `addCheckFactories()` registers individual checks by name:
   ```cpp
   Factories.registerCheck<UseNullptrCheck>("modernize-use-nullptr");
   ```
3. `ClangTidyForceLinker.h` has linker anchors ensuring all modules get linked in.

## Per-TU Processing

For each translation unit, `ClangTidyASTConsumerFactory::createASTConsumer()`:

1. Loads merged options for the file (CLI + `.clang-tidy` hierarchy)
2. Instantiates only the **enabled** checks
3. Each check calls `registerMatchers(MatchFinder*)` to register AST matchers
4. Optionally registers preprocessor callbacks (for `#include`/`#define` analysis)
5. `MatchFinder` traverses the AST, calling `check(MatchResult&)` on matches

## Check Lifecycle

Every check extends `ClangTidyCheck` (`ClangTidyCheck.h`):

- **`registerMatchers()`** — declare what AST patterns to match
- **`check()`** — called on each match; emit diagnostics via `diag()` with optional `FixItHint`s
- **`storeOptions()`** — serialize check-specific options (read via `Options.get()`)
- **`isLanguageVersionSupported()`** — skip checks for wrong language modes

## Configuration

`ClangTidyOptions.h/cpp` handles `.clang-tidy` YAML files. Options merge with priority: defaults < parent directory configs < local config < CLI flags. Each option carries a `Priority` value to resolve conflicts.

## Diagnostic Flow

```
check->diag(loc, "message")
  -> ClangTidyContext::diag()
    -> DiagnosticsEngine
      -> ClangTidyDiagnosticConsumer
        -> Creates ClangTidyError (SourceManager-independent)
        -> Converts FixItHints to tooling::Replacement
        -> Filters by header regex, NOLINT comments
        -> Applies fixes if --fix was passed
```

## Key Design

- **Pluggable**: new checks just subclass `ClangTidyCheck` and register in a module — no core changes needed
- **AST Matcher-based**: most checks are declarative pattern matches over the Clang AST
- **Hierarchical config**: `.clang-tidy` files cascade up the directory tree, merged by priority


Kay TODO next
- To implement new attributes
SemaDeclAttr.cpp - define the attribute so clang can parse it [clang/lib/Sema/]
every FlowSensitive class would need updating to check if that module is meant to be use for a partuclar pass (we can just do this for UncheckedOptional for now) [clang/lib/Analysis/FlowSens
  itive/Models/]
Have a look at attributes that are currently implemented in Attr.td [clang/include/Basic]