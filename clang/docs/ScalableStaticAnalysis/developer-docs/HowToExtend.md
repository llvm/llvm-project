# How to Extend the Framework

:::{WARNING}
The framework is rapidly evolving.
The documentation might be out-of-sync with the implementation.
The purpose of this documentation is to give context for upcoming reviews.
:::

SSAF is designed to be extensible with new **summary extractors** and **serialization formats**.
Extensions can be added in three ways:

1. **Statically, in-tree** — built as part of the upstream LLVM/Clang tree.
2. **Statically, out-of-tree (downstream)** — built in a downstream fork or project that links `clangScalableStaticAnalysisCore` as a static library.
3. **Dynamically, via plugins** — loaded at runtime as shared objects.

All three approaches use the same `llvm::Registry`-based registration mechanism.
The key difference is how the linker sees the registration:
static libraries need {doc}`force-linker anchors <ForceLinkerHeaders>` to prevent dead-stripping, while shared libraries do not.

## Adding a summary extractor

A summary extractor is an `ASTConsumer` that inspects the AST and populates a `TUSummary` via the `TUSummaryBuilder` interface.

### Step 1: Implement the extractor

```c++
//--- MyExtractor.h
#include "clang/ScalableStaticAnalysis/Core/TUSummary/TUSummaryExtractor.h"

namespace clang::ssaf {

class MyExtractor : public TUSummaryExtractor {
public:
  using TUSummaryExtractor::TUSummaryExtractor;

  // Override HandleTranslationUnit or any other virtual functions of an ASTConsumer...
  // Use the SummaryBuilder to populate the summary while walking the AST.
};

} // namespace clang::ssaf
```

### Step 2: Register the extractor

```c++
//--- MyExtractor.cpp
#include "MyExtractor.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/ExtractorRegistry.h"

using namespace clang::ssaf;

namespace clang::ssaf {
// NOLINTNEXTLINE(misc-use-internal-linkage)
volatile int MyExtractorAnchorSource = 0;
} // namespace clang::ssaf

static TUSummaryExtractorRegistry::Add<MyExtractor>
    RegisterExtractor("MyExtractor", "My awesome summary extractor");
```

The `"MyExtractor"` string is the name users pass to `--ssaf-extract-summaries=MyExtractor`.

### Step 3: Add the force-linker anchor

See {doc}`ForceLinkerHeaders` for a full explanation of why this is needed.

For **in-tree** additions, add one line to
`clang/include/clang/ScalableStaticAnalysis/BuiltinAnchorSources.def`
(in alphabetical order):

```c++
ANCHOR(MyExtractorAnchorSource)
```

`SSAFBuiltinForceLinker.h` includes this `.def` file automatically — no
need to edit it directly.

For **downstream** additions, see [Out-of-tree (downstream) extensions](#out-of-tree-downstream-extensions) below.

## Adding a serialization format

A serialization format controls how the `TUSummary` is written to (and read from) disk.
This involves more boilerplate than an extractor because each format has a per-analysis `FormatInfo` sub-registry.

### Step 1: Define the format class

Your format class must inherit from `SerializationFormat` and define a `FormatInfo` type alias:

```c++
//--- MyFormat.h
#include "clang/ScalableStaticAnalysis/Core/Serialization/SerializationFormat.h"
#include "clang/Support/Compiler.h"
#include "llvm/Support/Registry.h"

namespace clang::ssaf {

class MyFormat : public SerializationFormat {
public:
  // Define the type aliases: SerializerFn, DeserializerFn
  using FormatInfo = FormatInfoEntry<SerializerFn, DeserializerFn>;

  // Override readTUSummaryEncoding, writeTUSummary, etc.
};

} // namespace clang::ssaf

LLVM_DECLARE_REGISTRY(llvm::Registry<MyFormat::FormatInfo>)
```

### Step 2: Register the format

```c++
//--- MyFormat.cpp
#include "MyFormat.h"
#include "clang/ScalableStaticAnalysis/Core/Serialization/SerializationFormatRegistry.h"

using namespace clang::ssaf;

namespace clang::ssaf {
// NOLINTNEXTLINE(misc-use-internal-linkage)
volatile int MyFormatAnchorSource = 0;
} // namespace clang::ssaf

static SerializationFormatRegistry::Add<MyFormat>
    RegisterFormat("myformat", "My awesome serialization format");

LLVM_DEFINE_REGISTRY(llvm::Registry<MyFormat::FormatInfo>)
```

The format name (`"myformat"`) is matched against the file extension in `--ssaf-tu-summary-file=output.myformat`.

### Step 3: Register per-analysis FormatInfo entries

For each analysis that should be serializable in your format, register a `FormatInfo` entry.
`FormatInfo` must be implemented for any of the summaries that wants to support `myformat`:

```c++
namespace {
using FormatInfo = MyFormat::FormatInfo;
struct MyAnalysisFormatInfo final : FormatInfo {
  MyAnalysisFormatInfo() : FormatInfo{
              SummaryName("MyAnalysis"),
              serializeMyAnalysis,
              deserializeMyAnalysis,
          } {}
};
} // namespace

static llvm::Registry<FormatInfo>::Add<MyAnalysisFormatInfo>
    RegisterFormatInfo("MyAnalysisFormatInfo",
                       "MyFormat format info for MyAnalysis");
```

### Step 4: Add the force-linker anchor

Same pattern as for extractors — add the anchor to `BuiltinAnchorSources.def`
(in alphabetical order). See [Adding a summary extractor](#adding-a-summary-extractor) Step 3,
and {doc}`ForceLinkerHeaders`.

## Static extensibility

### In-tree extensions

For extensions that are part of the upstream LLVM/Clang tree:

1. Add the anchor to `clang/include/clang/ScalableStaticAnalysis/BuiltinAnchorSources.def` (in alphabetical order).
2. Add the source files to the `clangScalableStaticAnalysisCore` CMake library target.
3. That's it — the `SSAFForceLinker.h` umbrella includes `SSAFBuiltinForceLinker.h`
   transitively, so any binary that includes the umbrella will pull in the registration.

### Out-of-tree (downstream) extensions

Downstream projects that maintain a fork can add their own extensions without
modifying upstream files — reducing the risk of merge-conflicts:

1. Create a downstream force-linker header, e.g. `SSAFDownstreamForceLinker.h`,
   containing the anchor references for downstream-only extractors and formats.

2. Include it from `SSAFForceLinker.h` (the umbrella):

   ```c++
   // In SSAFForceLinker.h
   #include "SSAFBuiltinForceLinker.h"        // IWYU pragma: keep
   #include "SSAFDownstreamForceLinker.h"     // IWYU pragma: keep
   ```

   This is a single-line addition per downstream project, minimizing conflicts with upstream changes.
   Upstream will try to avoid modifying this umbrella header, making it a stable static extension point.

3. Add the downstream source files to the build system as usual.

## Dynamic extensibility (plugins)

Shared libraries loaded at runtime — via `dlopen` / `LoadLibrary` or the
Clang plugin mechanism — do **not** need force-linker anchors, but having them also does not hurt.

When a shared object (`.so` / `.dylib`) is loaded, the dynamic linker runs all global constructors in that library unconditionally.
This means the `llvm::Registry::Add<>` objects execute their constructors and register themselves automatically.

To use a plugin:

1. Build your extractor or format as a shared library.
2. Load it with the Clang plugin mechanism (`-fplugin=` or `-load`).
3. Pass the extractor name to `--ssaf-extract-summaries=` as usual.

No changes to any force-linker header are required.
The `llvm::Registry` infrastructure handles everything once the shared object is loaded.
