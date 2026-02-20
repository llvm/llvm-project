# ClangIR ABI Lowering - Design Document

## 1. Introduction

This design describes calling convention lowering that builds on the LLVM ABI
Lowering Library in `llvm/lib/ABI/`: we use its `abi::Type*` and target ABI
logic and add an MLIR integration layer (ABITypeMapper, ABI lowering pass, and
dialect rewriters).  The framework relies on the LLVM ABI library as the single
source of truth for ABI classification.  MLIR dialects use it via an adapter
layer.  The design provides a way to perform ABI-compliant calling convention
lowering that can be used by any MLIR dialect that implements the necessary
interfaces.  Inputs are high-level function signatures in CIR, FIR, or other
MLIR dialect.  Outputs are ABI-lowered signatures and call sites.  Lowering
runs as an MLIR pass in the compilation pipeline.

### 1.1 Design Goals

Building on the LLVM ABI library and adding an MLIR integration layer avoids
duplicating complex ABI logic across MLIR dialects, reduces maintenance, and
keeps a single source of ABI compliance in `llvm/lib/ABI/`.  The separation
between the ABI library (classification) and dialect-specific ABIRewriteContext
(rewriting) enables clearer testing and a straightforward migration path from
the CIR incubator by porting useful algorithms into the ABI library where
appropriate.

A central goal is that generated code be call-compatible with Classic Clang
CodeGen and other compilers.  Parity is with Classic Clang CodeGen output,
not only with the incubator.  Success means CIR correctly lowers x86_64 and
AArch64 calling conventions with full ABI compliance using the LLVM ABI library
and MLIR integration layer; FIR can adopt the same infrastructure with minimal
dialect-specific adaptation (e.g.  cdecl when calling C from Fortran).  ABI
compliance will be validated through differential testing against Classic Clang
CodeGen, and performance overhead should remain under 5% compared to a direct,
dialect-specific implementation.  Initial scope focuses on fixed-argument
functions; variadic support (varargs) is deferred.

## 2. Background and Context

### 2.1 What is Calling Convention Lowering?

Calling convention lowering transforms high-level function signatures to match
target ABI (Application Binary Interface) requirements.  When a function is
declared at the source level with convenient, language-level types, these types
must be translated into the specific register assignments, memory layouts, and
calling sequences that the target architecture expects.  For example, on x86_64
System V ABI, a struct containing two 64-bit integers might be "expanded" into
two separate arguments passed in registers, rather than being passed as a single
aggregate:

```
// High-level CIR
func @foo(i32, struct<i64, i64>) -> i32

// After ABI lowering
func @foo(i32 %arg0, i64 %arg1, i64 %arg2) -> i32
//        ^       ^            ^        ^
//        |       |            +--------+ struct expanded into fields
//        |       +---- first field passed in register
//        +---- small integer passed in register
```

Calling convention lowering is complex for several reasons: it is highly
target-specific (each architecture has different rules for registers vs.
memory), type-dependent (rules differ for integers, floats, structs, unions,
arrays), and context-sensitive (varargs, virtual calls, conventions like
vectorcall or preserve_most).  The same target may have multiple ABI variants
(e.g.  x86_64 System V vs.  Windows x64), adding further complexity.

### 2.2 Existing Implementations

#### Classic Clang CodeGen

Classic Clang CodeGen (located in `clang/lib/CodeGen/`) transforms calling
conventions during the AST-to-LLVM-IR lowering process.  This implementation is
mature and well-tested, handling all supported targets with comprehensive ABI
coverage.  However, it's tightly coupled to both Clang's AST representation and
LLVM IR, making it difficult to reuse for MLIR-based frontends.

#### CIR Incubator

The CIR incubator includes a calling convention lowering pass in
`clang/lib/CIR/Dialect/Transforms/TargetLowering/` that transforms CIR
operations into ABI-lowered CIR operations as an MLIR pass.  This implementation
successfully adapted logic from Classic Clang CodeGen to work within the MLIR
framework.  However, it relies on CIR-specific types and operations, preventing
reuse by other MLIR dialects.

#### LLVM ABI Lowering Library

A 2025 Google Summer of Code project produced [PR
#140112](https://github.com/llvm/llvm-project/pull/140112), which proposes
extracting Clang's ABI logic into a reusable library in `llvm/lib/ABI/`.  The
design centers on a shadow type system (`abi::Type*`) separate from both Clang's
AST types and LLVM IR types, enabling the ABI classification algorithms to work
independently of any specific frontend representation.  The library includes
abstract `ABIInfo` base classes and target-specific implementations (e.g.
x86_64, BPF) and provides QualTypeMapper for Clang to map `QualType` to
`abi::Type*`.

Our approach is to complete and extend this library and use it as the single
source of truth for ABI classification.  One implementation in one place reduces
duplication, simplifies bug fixes, and creates a path for Classic Clang CodeGen
to use the same logic in the future.  MLIR dialects (CIR, FIR, and others) will
use the library via an adapter layer rather than reimplementing ABI logic.

**Current state.** The x86_64 implementation is largely complete and under
review.  AArch64 and some other targets are not yet implemented; there is no
MLIR integration today.  The work is being upstreamed in smaller parts (e.g.
[PR 158329](https://github.com/llvm/llvm-project/pull/158329)); progress is
limited by reviewer bandwidth.  The overhead of the shadow type system
(converting to and from `abi::Type*`) has been measured at under 0.1% for clang
-O0, so it is negligible for CIR.  Our approach therefore depends on the ABI
library being merged upstream or our contributions to it being accepted.

**Our approach.** The approach is to complete and extend the ABI library (e.g.
AArch64, review feedback, tests) and add an **MLIR integration layer** so that
MLIR dialects can use it:

- **ABITypeMapper**: maps `mlir::Type` to `abi::Type*`, analogous to
  QualTypeMapper for Clang.

- **MLIR ABI lowering pass**: uses the library's `ABIInfo` for classification,
  then performs dialect-specific rewriting via `ABIRewriteContext` for CIR, FIR,
  and other dialects.

The CIR incubator serves as a **reference only** (e.g. for AArch64 algorithms).
We do not upstream the incubator's CIR-specific ABI implementation as the
long-term solution; we port useful algorithms into the ABI library where
appropriate.

### 2.3 Requirements for MLIR Dialects

CIR needs to lower C/C++ calling conventions correctly, with initial support for
x86_64 and AArch64 targets.  It must handle structs, unions, and complex types,
as well as support instance methods and virtual calls.  FIR's initial need is
**cdecl for calling C from Fortran** (C interop); that is in scope.
Fortran-specific ABI semantics (e.g.  CHARACTER hidden length parameters, array
descriptors) are out of initial scope; full Fortran ABI lowering is a broader
goal.  Both dialects share common requirements: strict target ABI compliance,
efficient lowering with minimal overhead, extensibility for adding new target
architectures, and comprehensive testability and validation capabilities.

## 3. Proposed Solution

**Core.** The LLVM ABI library in `llvm/lib/ABI/` performs ABI classification on
`abi::Type*`.  It provides `ABIInfo` and target-specific implementations
(x86_64, BPF, and eventually AArch64 and others).  This is the single place
where ABI rules are implemented.

**MLIR side.** To use this library from MLIR dialects we add an integration
layer: (1) **ABITypeMapper** maps `mlir::Type` to `abi::Type*` (analogous to
QualTypeMapper for Clang).  (2) A **generic ABI lowering pass** invokes the
library's `ABIInfo` for classification, then (3) performs **dialect-specific
rewriting** via the `ABIRewriteContext` interface—each dialect (CIR, FIR, etc.)
implements only the glue to create its own operations (e.g. `cir.call`,
`fir.call`).  Classification logic is shared; operation creation is
dialect-specific.

The following diagram shows the layering.  At the top, the ABI library holds
the ABI logic.  In the middle, adapters connect frontends to it: Classic Clang
CodeGen uses QualTypeMapper; MLIR uses ABITypeMapper and the ABI lowering pass.
At the bottom, each dialect implements `ABIRewriteContext` only; FIR is shown as
a consumer for cdecl/C interop (e.g. calling C from Fortran).

```
┌─────────────────────────────────────────────────────────────────┐
│  LLVM ABI Library (llvm/lib/ABI/)                               │
│  ABIInfo, abi::Type*, target implementations (X86, AArch64,…)   │
└─────────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┴─────────────────┐
            │                                   │
            ▼                                   ▼
┌───────────────────────┐         ┌───────────────────────────────┐
│  Classic CodeGen      │         │  MLIR adapter                 │
│  QualTypeMapper       │         │  ABITypeMapper + ABI pass     │
└───────────────────────┘         └───────────────────────────────┘
                                                │
                               ┌────────────────┼────────────────┐
                               │                │                │
                               ▼                ▼                ▼
                         ┌────────────┐   ┌────────────┐   ┌────────────┐
                         │ CIR        │   │ FIR        │   │ Future     │
                         │ ABIRewrite │   │ (cdecl/C   │   │ Dialects   │
                         │ Context    │   │  interop)  │   │            │
                         └────────────┘   └────────────┘   └────────────┘
```

## 4. Design Overview

### 4.1 Architecture Diagram

The following diagram shows how the design builds on the ABI library (Section
3).  At the top, the ABI library holds the classification logic.  The middle
layer adapts MLIR to the ABI library: ABITypeMapper converts `mlir::Type` to
`abi::Type*`, and the MLIR ABI lowering pass invokes the library's `ABIInfo` and
uses the classification
to drive rewriting.  At the bottom, each dialect implements only
`ABIRewriteContext` for operation creation; there is no separate type
abstraction layer in MLIR for classification—that lives in the ABI library.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  LLVM ABI Library (llvm/lib/ABI/) — single source of truth              │
│  abi::Type*, ABIInfo, target implementations (X86_64, AArch64, …)       │
│  Input: abi::Type*  →  Output: classification (ABIArgInfo, etc.)        │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  MLIR adapter                                                           │
│  ABITypeMapper (mlir::Type → abi::Type*)  +  MLIR ABI lowering pass     │
│  (1) Map types  (2) Call ABIInfo  (3) Drive rewriting from              │
│  classification result                                                  │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
              ┌────────────┐    ┌────────────┐    ┌────────────┐
              │ CIR        │    │ FIR        │    │ Future     │
              │ ABIRewrite │    │ ABIRewrite │    │ Dialects   │
              │ Context    │    │ Context    │    │            │
              └────────────┘    └────────────┘    └────────────┘
              Dialect-specific operation creation only (no type
              abstraction for classification in MLIR)
```

### 4.2 ABI Library, Adapter, and Dialect Layers

The architecture has three parts.  **The ABI library** (`llvm/lib/ABI/`) is the
single source of truth for ABI classification: it operates on `abi::Type*` and
produces classification results (e.g.  ABIArgInfo, ABIFunctionInfo).
Target-specific `ABIInfo` implementations (X86_64, AArch64, etc.) live there.
The **adapter layer** is MLIR-specific: ABITypeMapper maps `mlir::Type` to
`abi::Type*`, and the MLIR ABI lowering pass (1) maps types, (2) calls the
library's ABIInfo, and (3) uses the classification to drive rewriting.  The
**dialect layer** is only ABIRewriteContext: each dialect (CIR, FIR) implements
operation creation (createFunction, createCall, createExtractValue, etc.).
There is no type abstraction layer in MLIR for classification; type queries for
ABI are performed on `abi::Type*` inside the ABI library.

### 4.3 Key Components

The framework is built from the following components.  **The ABI library**
(`llvm/lib/ABI/`) provides the single source of truth for ABI classification:
the `abi::Type*` type system, the `ABIInfo` base and target-specific
implementations (e.g.  X86_64, AArch64), and the classification result types
(e.g.  ABIArgInfo, ABIFunctionInfo).  **ABITypeMapper** maps `mlir::Type` to
`abi::Type*` so that MLIR dialect types can be classified by the ABI library.
Dialects with custom types do not need a new interface for this: the mapper
relies on existing MLIR type interfaces (e.g.  `DataLayoutTypeInterface`) for
size and alignment, and pattern-matches on standard type categories (integers,
floats, pointers, structs, arrays, vectors) to build `abi::Type*`.
The **MLIR ABI lowering pass** orchestrates the flow: it uses ABITypeMapper,
calls the library's ABIInfo, and drives rewriting from the classification
result.  **ABIRewriteContext** is the dialect-specific interface for operation
creation (each dialect implements it to produce e.g.  cir.call, fir.call).  A
**target registry** (or equivalent) is used to select the appropriate ABIInfo
for the compilation target.  There is no ABITypeInterface or separate "ABIInfo
in MLIR"; classification lives entirely in the ABI library.

### 4.4 ABI Lowering Flow: How the Pieces Fit Together

This section describes the end-to-end flow of ABI lowering, showing how all
interfaces and components work together.

#### Step 1: Function Signature Analysis

The ABI lowering pass begins by analyzing the function signature.  Function
operations are identified via MLIR's `FunctionOpInterface`, which provides
access to the function type, argument types, and return types.  The pass
extracts the parameter types and return type to prepare them for
classification.  At this stage, the types are still in their
high-level, dialect-specific form (e.g., `!cir.struct` for CIR, or `!fir.type`
for FIR).  The pass collects these types into a list that will be fed to the
classification logic in the next step.

```
Input: func @foo(%arg0: !cir.int<u, 32>,
       %arg1: !cir.struct<{!cir.int<u, 64>,
                            !cir.int<u, 64>}>) -> !cir.int<u, 32>
```

#### Step 2: Type Mapping via ABITypeMapper

For each argument and the return type, the pass maps `mlir::Type` to
`abi::Type*` using ABITypeMapper.  The mapper produces the representation that
the library's ABIInfo expects; optionally, it can map back to MLIR types for coercion
types when needed.

```cpp
// Map dialect types to the library's type system
ABITypeMapper abiTypeMapper(module.getDataLayout());
abi::Type *arg0Abi = abiTypeMapper.map(arg0Type);   // i32 -> IntegerType
abi::Type *arg1Abi = abiTypeMapper.map(arg1Type);   // struct -> RecordType
abi::Type *retAbi = abiTypeMapper.map(returnType);
```

**Key Point**: Classification runs in the ABI library on `abi::Type*`; ABITypeMapper is
the only bridge from dialect types to that representation.

#### Step 3: ABI Classification

The library's target-specific `ABIInfo` (e.g.  X86_64) performs classification on
`abi::Type*` and produces the library's classification result (e.g.  ABIFunctionInfo
and ABIArgInfo as defined in `llvm/lib/ABI/`):

```cpp
// The MLIR ABI lowering pass obtains the ABIInfo from the target
// registry based on the module's target triple (see Section 5.2).
llvm::abi::ABIInfo *abiInfo = getABIInfo();  // e.g. X86_64
llvm::abi::ABIFunctionInfo abiFI;
abiInfo->computeInfo(abiFI, arg0Abi, arg1Abi, retAbi);
// For struct<i64,i64> on x86_64: produces Expand (two i64 args)
```

Output: the library's classification (e.g.  ABIFunctionInfo) for all arguments and
return:
- `%arg0 (i32)` → Direct (pass as-is)
- `%arg1 (struct)` → Expand (split into two i64 fields)
- Return type → Direct

#### Step 4: Function Signature Rewriting

After the library's classification is complete, the pass rewrites the function to match
the ABI requirements using the dialect's `ABIRewriteContext`.  The
classification result (from the ABI library) describes the lowered signature; the rewrite
context creates the actual dialect operations.  For example, if a struct is
classified as "Expand", the new function signature will have multiple scalar
parameters instead of the single struct parameter.

```cpp
ABIRewriteContext &ctx = getDialectRewriteContext();

// Create new function with lowered signature
FunctionType newType = ...; // (i32, i64, i64) -> i32
Operation *newFunc = ctx.createFunction(loc, "foo", newType);
```

**Key Point**: The original function had signature `(i32, struct) -> i32`, but
the ABI-lowered function has signature `(i32, i64, i64) -> i32` with the struct
expanded into its constituent fields.

#### Step 5: Argument Expansion

With the function signature rewritten, the pass updates all call sites to match
the new signature, using the classification from the ABI library to drive rewriting via
`ABIRewriteContext`.  For arguments classified as "Expand", the pass breaks down
the aggregate into its constituent parts (e.g.  struct into two i64 values).
The rewrite context provides operations to extract fields and construct the new
call with the expanded argument list.

```cpp
// Original call: call @foo(%val0, %structVal)
// Need to extract struct fields:

Value field0 = ctx.createExtractValue(loc, structVal, {0}); // extract 1st i64
Value field1 = ctx.createExtractValue(loc, structVal, {1}); // extract 2nd i64

// New call with expanded arguments
ctx.createCall(loc, newFunc, {resultType}, {val0, field0, field1});
```

**Key Point**: `ABIRewriteContext` abstracts the dialect-specific operation
creation, so the lowering logic doesn't need to know about CIR operations.

#### Step 6: Return Value Handling

For functions returning large structs (indirect return):

```cpp
// If return type is classified as Indirect:
Value sretPtr = ctx.createAlloca(loc, retType, alignment);
ctx.createCall(loc, func, {}, {sretPtr, ...otherArgs});
Value result = ctx.createLoad(loc, sretPtr);
```

#### Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ Input: High-Level Function (CIR/FIR/other dialect)              │
│         func @foo(%arg0: i32, %arg1: struct<i64,i64>) -> i32    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: Extract Types                                           │
│   For each parameter: mlir::Type                                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 2: Map Types (ABITypeMapper → abi::Type*)                  │
│   abiTypeMapper.map(argType) → abi::Type*                       │
│   └─> Dialect types converted for ABI library                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 3: Classify (ABIInfo)                                      │
│   abiInfo->computeInfo(abiFI, ...) on abi::Type*                │
│   Applies target rules (e.g. x86_64 System V)                   │
│   └─> Produces: ABIFunctionInfo / ABIArgInfo                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 4: Rewrite Function (ABIRewriteContext)                    │
│   Use ABI classification to build lowered signature             │
│   └─> ctx.createFunction(loc, name, newType); (i32, i64, i64)   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 5: Rewrite Call Sites (ABIRewriteContext)                  │
│   ctx.createExtractValue() - expand struct; ctx.createCall()    │
│   └─> Dialect-specific operation creation                       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Output: ABI-Lowered Function                                    │
│         func @foo(%arg0: i32, %arg1: i64, %arg2: i64) -> i32    │
└─────────────────────────────────────────────────────────────────┘
```

#### Key Interactions Between Components

Classification lives in the ABI library: `ABIInfo` operates on `abi::Type*` and produces
classification results (e.g.  ABIArgInfo, ABIFunctionInfo).  MLIR types reach
the ABI library only via ABITypeMapper, which converts `mlir::Type` to `abi::Type*`.  The
lowering pass (1) maps types with ABITypeMapper, (2) calls the library's ABIInfo to
get classification, and (3) uses that result to drive rewriting through the
dialect's ABIRewriteContext.

ABIRewriteContext consumes the classification (e.g.  "Expand" for a struct) and
performs the actual IR changes: createFunction with the lowered signature,
createExtractValue and createCall at call sites.  Each dialect implements
ABIRewriteContext to produce its own operations (e.g.  cir.call, fir.call).
This keeps classification in one place (the ABI library) and limits dialect code to
operation creation.

## 5. ABIRewriteContext and Target Registry

### 5.1 ABIRewriteContext Interface

ABIRewriteContext is the only dialect-specific layer: CIR and FIR each
implement it to create their own dialect operations (e.g.  cir.call, fir.call).
In a module with mixed dialect content, the pass selects the appropriate
ABIRewriteContext for each function based on the dialect of its operations.  Classification is
performed by the library's ABIInfo and produces the library's result (e.g.  ABIFunctionInfo,
ABIArgInfo); ABIRewriteContext consumes that classification to perform the
actual IR rewriting.  The interface defines the operations needed for lowering
(createFunction, createCall, createExtractValue, createLoad, etc.); each dialect
implements these to produce its own operations.  ABIRewriteContext is also
responsible for updating ABI-related attributes (e.g.  sret, byval, signext,
zeroext, inreg) on the rewritten function signatures and call sites as
indicated by the classification result.

The interface defines approximately 15-20 methods covering function operations
(`createFunction`, `createCall`), value manipulation (`createCast`,
`createLoad`, `createStore`, `createAlloca`), type coercion (`createBitcast`,
`createTrunc`, `createZExt`, `createSExt`), aggregate operations
(`createExtractValue`, `createInsertValue`, `createGEP`), and housekeeping
(`createFunctionType`, `replaceOp`).  This set was chosen based on analyzing the
operations actually needed by existing ABI lowering code: struct expansion
requires extract/insert operations, indirect passing requires alloca and pointer
operations, and coercion requires bitcasts and truncations.

Each dialect implementing ABI lowering must provide a concrete
`ABIRewriteContext` subclass—estimated at 800-1000 lines of implementation code
that wraps the dialect's builder API.  This is a significant but one-time cost:
CIR implements `CIRABIRewriteContext`, FIR implements `FIRABIRewriteContext`,
and any future dialect reuses the shared classification infrastructure by
providing its own context implementation.  The alternative—reimplementing the
entire ABI classification logic per dialect—would require 8,000-15,000 lines per
dialect (the combined size of x86_64 and AArch64 classification code plus all
supporting infrastructure), introduce divergent behavior across dialects, and
create a maintenance burden where ABI bug fixes must be propagated to every
dialect independently.

### 5.2 Target Registry

We use the library's target selection or registry to obtain the appropriate ABIInfo for
the compilation target (e.g.  X86_64, AArch64).  We do not introduce a separate
MLIR TargetRegistry unless the MLIR ABI pass needs it for pass options or
configuration.  The dependency direction is: the MLIR ABI pass depends on
`llvm/lib/ABI`; there is no reverse dependency from the ABI library to MLIR dialects.

## 6. Open Questions

The following items are open for discussion.  This section may be revised,
shortened, or removed before final merge.

### 6.1 How to Handle clang::TargetInfo Dependency in MLIR?

The CIR incubator currently uses `clang::TargetInfo` to query target-specific
properties needed for ABI decisions, such as pointer width, alignment,
endianness, and calling convention availability.  Moving this functionality to
MLIR dialect-agnostic infrastructure raises an architectural question: should
MLIR code depend on a Clang library, or should it use MLIR-based mechanisms?

Three approaches are under consideration.

1.  Continue using `clang::TargetInfo` directly, accepting an MLIR→Clang
   dependency for this target-specific infrastructure.  This approach requires
   no additional implementation since it already works in the CIR incubator,
   and `clang::TargetInfo` provides comprehensive, battle-tested coverage of
   all target properties.  However, it creates a dependency relationship that
   may violate MLIR's architectural principle of being a peer to Clang rather
   than dependent on it.

2.  Combine `llvm::Triple` with MLIR's `DataLayoutInterface`, supplemented by
   module-level attributes for ABI-specific properties not covered by the data
   layout.  This approach maintains clean layering with no Clang dependency and
   follows MLIR patterns, but requires defining approximately 10-15 additional
   attributes and some upfront design work.

3.  Create a new `mlir::target::TargetInfo` abstraction with minimal methods
   tailored specifically for ABI needs (approximately 15-20 methods).  This
   provides clean layering without Clang dependency but requires implementing
   and maintaining target-specific code that duplicates some knowledge from
   `clang::TargetInfo`.

Option 2 is recommended as the preferred approach.  It maintains MLIR's
independence from Clang, which is important for MLIR's mission to be reusable by
non-Clang frontends like Rust, Julia, and Swift.  Target information is input
metadata rather than an output format, so it should be expressible through
MLIR's existing mechanisms rather than requiring external dependencies.  Option
3 serves as an acceptable fallback if Option 2 proves insufficient during
prototyping, while Option 1 is not recommended due to the architectural concerns
around MLIR depending on Clang.

### 6.2 Scope: C Calling Convention vs.  Arbitrary Calling Conventions

This design focuses on the **C calling convention layer** (e.g. cdecl, System V,
AAPCS).  C++ ABI concerns such as non-trivial copy constructors or destructors
are largely handled elsewhere in the compilation pipeline; the ABI library and
MLIR integration layer address how arguments and return values are passed at the
C ABI boundary.  An open question is whether the design should remain explicitly
scoped to C calling conventions only, or be general enough to support arbitrary
calling conventions (e.g. vectorcall, preserve_most) via extensible interfaces.
Clarifying this scope will guide the design of the LLVM ABI library integration
and the MLIR pass.
