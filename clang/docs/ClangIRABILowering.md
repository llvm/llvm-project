# ClangIR ABI Lowering - Design Document

## 1. Introduction

This document proposes a comprehensive design for creating an MLIR-agnostic calling convention lowering framework. The framework will enable CIR to perform ABI-compliant calling convention lowering, be reusable by other MLIR dialects (particularly FIR), achieve parity with the CIR incubator implementation for x86_64 and AArch64, and integrate with or inform the GSoC ABI Lowering Library project.

### 1.1 Problem Statement

Calling convention lowering is currently implemented separately for each MLIR dialect that needs it. The CIR incubator has a partial implementation, but it's tightly coupled to CIR-specific types and operations, making it unsuitable for reuse by other dialects. This means that FIR (Fortran IR) and future MLIR dialects would need to duplicate this complex logic. While classic Clang codegen contains mature ABI lowering code, it cannot be reused directly because it's tightly coupled to Clang's AST representation and LLVM IR generation.

### 1.2 Proposed Solution

This design proposes a shared MLIR ABI lowering infrastructure that multiple dialects can leverage. The framework sits at the top, providing common interfaces and target-specific ABI classification logic. Each MLIR dialect (CIR, FIR, and future dialects) implements a small amount of dialect-specific glue code to connect to this infrastructure. At the bottom, target-specific implementations handle the complex ABI rules for architectures like x86_64 and AArch64. This approach enables code reuse while maintaining the flexibility for each dialect to handle its own operation creation patterns.

```
┌──────────────────────────────────────────────────────────────┐
│         MLIR ABI Lowering Infrastructure                     │
│         mlir/include/mlir/Interfaces/ABI/                    │
└──────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            │                 │                 │
            ▼                 ▼                 ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │ CIR Dialect  │  │ FIR Dialect  │  │   Future     │
    │              │  │              │  │   Dialects   │
    └──────────────┘  └──────────────┘  └──────────────┘
         │                 │                 │
         └─────────────────┴─────────────────┘
                           │
                           ▼
               ┌───────────────────────┐
               │  Target ABI Logic     │
               │  X86, AArch64, etc.   │
               └───────────────────────┘
```

### 1.3 Key Benefits

This architecture avoids duplicating complex ABI logic across MLIR dialects, reducing the maintenance burden and risk of inconsistencies. It maintains correct ABI compliance for all targets by reusing proven classification algorithms. The clear separation of concerns enables easier testing and validation, as each layer can be tested independently. Additionally, the design provides a straightforward migration path from the existing CIR incubator implementation.

### 1.4 Success Criteria

The framework will be considered successful when CIR can correctly lower x86_64 and AArch64 calling conventions with full ABI compliance. FIR should be able to adopt the same infrastructure with minimal dialect-specific adaptation. ABI compliance will be validated through differential testing, comparing output against classic Clang codegen to ensure correct calling convention implementation. Finally, the performance overhead should remain under 5% compared to a direct, dialect-specific implementation.

## 2. Background and Context

### 2.1 What is Calling Convention Lowering?

Calling convention lowering transforms high-level function signatures to match target ABI (Application Binary Interface) requirements. When a function is declared at the source level with convenient, language-level types, these types must be translated into the specific register assignments, memory layouts, and calling sequences that the target architecture expects. For example, on x86_64 System V ABI, a struct containing two 64-bit integers might be "expanded" into two separate arguments passed in registers, rather than being passed as a single aggregate:

```
// High-level CIR
func @foo(i32, struct<i64, i64>) -> i32

// After ABI lowering
func @foo(i32 %arg0, i64 %arg1, i64 %arg2) -> i32
//        ^       ^            ^        ^
//        |       |            +--------+---- struct expanded into fields
//        |       +---- first field passed in register
//        +---- small integer passed in register
```

### 2.2 Why It's Complex

Calling convention lowering is complex for several reasons. First, it's highly target-specific: each architecture (x86_64, AArch64, RISC-V, etc.) has different rules for how arguments are passed in registers versus memory. Second, it's type-dependent: the rules differ significantly for integers, floating-point values, structs, unions, and arrays. Third, it's context-sensitive: special handling is required for varargs functions, virtual method calls, and alternative calling conventions like vectorcall or preserve_most. Finally, the same target may have multiple ABI variants (e.g., x86_64 System V vs. Windows x64), adding another dimension of complexity.

### 2.3 Existing Implementations

#### Classic Clang CodeGen

Classic Clang codegen (located in `clang/lib/CodeGen/`) transforms calling conventions during the AST-to-LLVM-IR lowering process. This implementation is mature and well-tested, handling all supported targets with comprehensive ABI coverage. However, it's tightly coupled to both Clang's AST representation and LLVM IR, making it difficult to reuse for MLIR-based frontends.

#### CIR Incubator

The CIR incubator includes a calling convention lowering pass in `clang/lib/CIR/Dialect/Transforms/TargetLowering/` that transforms CIR operations into ABI-lowered CIR operations as an MLIR pass. This implementation successfully adapted logic from classic codegen to work within the MLIR framework. However, it relies on CIR-specific types and operations, preventing reuse by other MLIR dialects.

#### GSoC ABI Lowering Library (WIP)

The Google Summer of Code project (PR #140112, not yet merged) proposes an independent ABI type system extracted from Clang's codegen. This library aims to be frontend-agnostic and reusable across different language frontends. While promising, it's still under development and currently focuses on Clang and LLVM IR rather than MLIR abstractions.

### 2.4 Requirements for MLIR Dialects

CIR needs to lower C/C++ calling conventions correctly, with initial support for x86_64 and AArch64 targets. It must handle structs, unions, and complex types, as well as support instance methods and virtual calls. FIR will have similar but distinct requirements in the future, needing to lower Fortran calling conventions with Fortran-specific types like complex numbers and derived types, while supporting Fortran's unique calling semantics. Both dialects share common requirements: strict target ABI compliance, efficient lowering with minimal overhead, extensibility for adding new target architectures, and comprehensive testability and validation capabilities.

## 3. Design Overview

### 3.1 Architecture Diagram

The following diagram provides a detailed view of the three-layer architecture introduced in Section 1.2. At the top (Layer 3), each dialect provides its own rewrite context for creating dialect-specific operations. In the middle (Layer 2), the `ABITypeInterface` provides a dialect-agnostic way to query type properties, allowing the classification logic below to work with any dialect's types. At the bottom (Layer 1), target-specific `ABIInfo` implementations (e.g., X86_64, AArch64) perform the actual ABI classification using only the abstract type information from Layer 2. Data flows downward for classification, then back upward for operation rewriting.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MLIR-Agnostic ABI Lowering                       │
│                         (Three-Layer Design)                        │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ Layer 3: Dialect-Specific Operation Rewriting                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │
│  │ CIR Rewrite │  │ FIR Rewrite │  │ Other       │                  │
│  │ Context     │  │ Context     │  │ Dialects    │                  │
│  └──────┬──────┘  └───────┬─────┘  └────────┬────┘                  │
│         │                 │                 │                       │
│         └─────────────────┼─────────────────┘                       │
│                           │                                         │
└───────────────────────────┼─────────────────────────────────────────┘
                            │ ABIRewriteContext Interface
┌───────────────────────────┼─────────────────────────────────────────┐
│ Layer 2: Interface-Based Type Abstractions                          │
│                           │                                         │
│  ┌────────────────────────▼──────────────────────────┐              │
│  │ ABITypeInterface (TypeInterface)                  │              │
│  │  - isRecord(), isInteger(), isFloatingPoint()     │              │
│  │  - getNumFields(), getFieldType()                 │              │
│  │  - getAlignof(), getSizeof()                      │              │
│  └────────────────────────┬──────────────────────────┘              │
│                           │                                         │
└───────────────────────────┼─────────────────────────────────────────┘
                            │ Abstract Type Queries
┌───────────────────────────┼─────────────────────────────────────────┐
│ Layer 1: Pure ABI Classification Logic (Dialect-Agnostic)           │
│                           │                                         │
│  ┌────────────────────────▼──────────────────────────┐              │
│  │ ABIInfo (Target-Specific)                         │              │
│  │  - classifyArgumentType()                         │              │
│  │  - classifyReturnType()                           │              │
│  └──────┬─────────────────────────┬──────────────┬───┘              │
│         │                         │              │                  │
│  ┌──────▼──────┐  ┌───────────────▼───┐  ┌───────▼──────┐           │
│  │ X86_64      │  │ AArch64           │  │ Other        │           │
│  │ ABIInfo     │  │ ABIInfo           │  │ Targets      │           │
│  └─────────────┘  └───────────────────┘  └──────────────┘           │
│                                                                     │
│  Output: LowerFunctionInfo + ABIArgInfo                             │
└─────────────────────────────────────────────────────────────────────┘

```

### 3.2 Three-Layer Design

The architecture is organized into three distinct layers, each with clear responsibilities. Layer 1 performs pure ABI classification, taking an `mlir::Type` and metadata as input and producing `ABIArgInfo` that describes how to pass the value. This layer has no dialect knowledge and implements target-specific algorithms. Layer 2 provides type and layout abstraction through the `ABITypeInterface` for querying type properties, leveraging MLIR's standard `DataLayoutInterface`, and using shared data structures like `ABIArgInfo` and `LowerFunctionInfo` to capture classification results. Layer 3 handles dialect-specific rewriting through the `ABIRewriteContext` interface, where each dialect implements its own operation creation logic, pass infrastructure, value coercion, and temporary allocation strategies.

### 3.3 Key Components

The framework consists of six key components that work together to perform ABI lowering. `ABIArgInfo` captures the classification result, indicating whether an argument should be passed directly, indirectly, expanded, or handled through other strategies. `LowerFunctionInfo` represents a fully classified function signature, aggregating the `ABIArgInfo` results for all parameters and the return value. `ABITypeInterface` provides the type query mechanism that enables ABI classification logic to inspect type properties without coupling to specific dialects. `ABIInfo` implements the target-specific classification algorithms (e.g., x86_64 System V, AArch64 PCS). `ABIRewriteContext` defines the interface for dialect-specific operation creation and rewriting. Finally, `TargetRegistry` maps target triples to their corresponding ABI implementations, enabling runtime selection of the appropriate target-specific logic.

### 3.4 ABI Lowering Flow: How the Pieces Fit Together

This section describes the end-to-end flow of ABI lowering, showing how all interfaces and components work together.

#### Step 1: Function Signature Analysis

The ABI lowering pass begins by analyzing the function signature. When it encounters a function operation, it extracts the parameter types and return type to prepare them for classification. At this stage, the types are still in their high-level, dialect-specific form (e.g., `!cir.struct` for CIR, or `!fir.type` for FIR). The pass collects these types into a list that will be fed to the classification logic in the next step.

```
Input: func @foo(%arg0: !cir.int<u, 32>, %arg1: !cir.struct<{!cir.int<u, 64>, !cir.int<u, 64>}>) -> !cir.int<u, 32>
```

#### Step 2: Type Classification via ABITypeInterface

For each argument and the return type, the target-specific `ABIInfo` queries type properties through `ABITypeInterface`:

```cpp
// For %arg1 (struct type)
ABITypeInterface typeIface = arg1Type.cast<ABITypeInterface>();
bool isRecord = typeIface.isRecord();           // true
unsigned numFields = typeIface.getNumFields();  // 2
Type field0 = typeIface.getFieldType(0);        // i64
Type field1 = typeIface.getFieldType(1);        // i64
```

**Key Point**: `ABITypeInterface` allows the `ABIInfo` to inspect types without knowing about CIR-specific type classes.

#### Step 3: ABI Classification

The target `ABIInfo` (e.g., `X86_64ABIInfo`) applies platform-specific rules:

```cpp
X86_64ABIInfo::classifyArgumentType(mlir::Type argType, LowerFunctionInfo &FI) {
  // For struct<i64, i64>:
  // - Check size: 16 bytes (fits in 2 registers)
  // - Classify: INTEGER (x86_64 System V ABI rule)
  // - Result: Expand into two i64 arguments
  return ABIArgInfo::getExpand();
}
```

Output: `LowerFunctionInfo` containing classification for all arguments:
- `%arg0 (i32)` → `ABIArgInfo::Direct` (pass as-is)
- `%arg1 (struct)` → `ABIArgInfo::Expand` (split into two i64 fields)
- Return type → `ABIArgInfo::Direct`

#### Step 4: Function Signature Rewriting

After classification is complete, the pass must rewrite the function to match the ABI requirements. This involves creating a new function with a transformed signature that reflects how arguments will actually be passed at the machine level. For example, if a struct is classified as "Expand", the new function signature will have multiple scalar parameters instead of the single struct parameter. The `ABIRewriteContext` provides the dialect-specific hooks to create this new function operation while preserving the dialect's semantics.

```cpp
ABIRewriteContext &ctx = getDialectRewriteContext();

// Create new function with lowered signature
FunctionType newType = ...; // (i32, i64, i64) -> i32
Operation *newFunc = ctx.createFunction(loc, "foo", newType);
```

**Key Point**: The original function had signature `(i32, struct) -> i32`, but the ABI-lowered function has signature `(i32, i64, i64) -> i32` with the struct expanded into its constituent fields.

#### Step 5: Argument Expansion

With the function signature rewritten, the pass must now update all call sites to match the new signature. For arguments that were classified as "Expand", the pass needs to break down the aggregate value into its constituent parts. In our example, the struct argument must be split into two separate i64 values. The `ABIRewriteContext` provides operations to extract fields from aggregates and construct the new call with the expanded argument list.

```cpp
// Original call: call @foo(%val0, %structVal)
// Need to extract struct fields:

Value field0 = ctx.createExtractValue(loc, structVal, {0}); // extract first i64
Value field1 = ctx.createExtractValue(loc, structVal, {1}); // extract second i64

// New call with expanded arguments
ctx.createCall(loc, newFunc, {resultType}, {val0, field0, field1});
```

**Key Point**: `ABIRewriteContext` abstracts the dialect-specific operation creation, so the lowering logic doesn't need to know about CIR operations.

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
│ Step 2: Query Type Properties (ABITypeInterface)                │
│   typeIface.isRecord(), getNumFields(), getFieldType()          │
│   └─> Type-agnostic inspection (no CIR/FIR knowledge)           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 3: Classify (Target ABIInfo)                               │
│   X86_64ABIInfo::classifyArgumentType(type, functionInfo)       │
│   Applies x86_64 System V rules                                 │
│   └─> Produces: ABIArgInfo (Direct, Indirect, Expand, etc.)     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 4: Build LowerFunctionInfo                                 │
│   Aggregate all ABIArgInfo results                              │
│   └─> Complete calling convention specification                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 5: Rewrite Function (ABIRewriteContext)                    │
│   ctx.createFunction(loc, name, newType)                        │
│   New signature: (i32, i64, i64) -> i32                         │
│   └─> Dialect-specific operation creation                       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 6: Rewrite Call Sites (ABIRewriteContext)                  │
│   ctx.createExtractValue() - expand struct                      │
│   ctx.createCall() - call with expanded args                    │
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

The framework's power comes from how these components interact with clear separation of concerns. The `ABITypeInterface` and `ABIInfo` interaction is foundational: `ABIInfo` calls `ABITypeInterface` methods like `isRecord()`, `getNumFields()`, and `getFieldType()` to inspect types, which enables the target logic to work with any dialect's types without coupling to specific type implementations.

The `ABIInfo` produces `ABIArgInfo` structures for each argument, where `ABIArgInfo` remains completely dialect-agnostic by only describing "how to pass" the value (Direct, Indirect, Expand, etc.). These classification results are stored in `LowerFunctionInfo`, creating a complete specification of the function's calling convention.

The lowering pass then reads `LowerFunctionInfo` to understand what transformations to apply, and calls `ABIRewriteContext` methods to perform the actual IR rewriting. For example, if an argument has `ABIArgInfo::Expand`, the pass will call `createExtractValue()` for each field that needs to be separated.

Finally, the dialect implements the `ABIRewriteContext` interface to return dialect-specific operations (like `cir.call` for CIR or `fir.call` for FIR). This ensures that the ABI lowering logic never directly creates dialect operations, maintaining clean separation.

This layered separation enables three key benefits: target logic reuse where `ABIInfo` works with any dialect via interfaces, dialect flexibility where each dialect controls its own operation creation patterns, and testability where ABIInfo classification can be tested independently of dialect operations.

## 4. Detailed Component Design

### 4.1 ABIArgInfo

The `ABIArgInfo` class captures the result of ABI classification for a single argument or return value. When a target-specific `ABIInfo` implementation analyzes a type (such as a struct or primitive), it produces an `ABIArgInfo` describing whether the value should be passed directly in registers, indirectly through memory, expanded into multiple arguments, or handled through other specialized mechanisms. This separation between classification (producing `ABIArgInfo`) and rewriting (consuming `ABIArgInfo`) is fundamental to achieving dialect independence: the classification logic operates purely on type metadata and doesn't need to know about specific MLIR operations like `cir.call` or `fir.call`.

The classification is captured through a `Kind` enum with variants like `Direct` (pass value as-is, potentially with type coercion), `Indirect` (pass via hidden pointer), `Expand` (split aggregate into individual field arguments), and several others for edge cases like sign/zero extension or Windows-specific calling conventions. Each kind may carry additional information such as coercion types (for example, passing `{float, float}` as `<2 x float>` on x86_64) or padding requirements. This design is adapted directly from Clang's existing `ABIArgInfo` in `clang/lib/CodeGen/CGCall.h`, which has proven robust and comprehensive across years of production ABI implementation work spanning dozens of targets.

This component already exists in the CIR incubator codebase and is dialect-agnostic by design—it describes "how to pass a value" without prescribing "how to create operations." The only work required is moving it from `clang/lib/CIR/` to `mlir/include/mlir/Interfaces/ABI/ABIArgInfo.h` to make it available to all MLIR dialects.

### 4.2 LowerFunctionInfo

The `LowerFunctionInfo` class represents a complete, ABI-classified function signature. It associates each argument and the return value with both its original high-level type (e.g., `!cir.struct<"Point", !s32, !s32>`) and the `ABIArgInfo` describing how it should be lowered (e.g., `Direct` with coercion to `i64`). This pairing of original type and ABI classification is essential because the dialect-specific rewriter needs both pieces of information: the original type tells it which operations to rewrite, while the `ABIArgInfo` tells it how to perform the transformation.

The class also captures metadata like the calling convention (C, fastcc, etc.) and whether the function accepts variable arguments, which affect classification rules. The internal storage treats the return value as argument index 0, followed by actual arguments—a convention inherited from Clang's implementation that simplifies iteration over all classified values. This design choice means that a function with N parameters contains N+1 entries in the classification vector, and helper methods like `getReturnInfo()` and `getArgInfo(i)` provide convenient access with proper index translation.

Like `ABIArgInfo`, this component already exists in CIR and requires only minor adaptations to be fully dialect-agnostic. The primary change is ensuring it doesn't directly reference CIR-specific types in its implementation, instead relying on the generic `mlir::Type` interface. Once moved to `mlir/include/mlir/Interfaces/ABI/`, it becomes available for use by any MLIR dialect that needs ABI lowering.

### 4.3 ABITypeInterface

The `ABITypeInterface` is an MLIR `TypeInterface` that defines the contract for exposing ABI-relevant type metadata. Target-specific ABI classification algorithms need to answer questions like "Is this an integer type?", "How large is this struct?", "What are the types and offsets of its fields?", and "Does this C++ class have a non-trivial destructor?" Without a common interface, the classification code would need to perform dialect-specific type casting (e.g., `dyn_cast<cir::StructType>` vs `dyn_cast<fir::RecordType>`), making it impossible to share the complex ABI logic across dialects. This interface solves that problem by requiring each dialect's types to implement a standard set of query methods.

The interface defines 15-25 methods covering basic type classification (`isInteger()`, `isRecord()`, `isPointer()`), type navigation (`getPointeeType()`, `getFieldType(unsigned index)`), size and alignment queries (`getSizeInBits()`, `getABIAlignmentInBits()`), and specialized predicates for edge cases like `__int128`, `_BitInt(N)`, and C++ non-trivial lifecycle operations. The exact method list will be finalized during Phase 1 Week 1 by auditing the existing x86_64 and AArch64 classification code to identify every type query used in practice. The TableGen-based interface definition ensures compile-time enforcement: if a type advertises `ABITypeInterface::Trait`, the compiler verifies that all required methods are implemented.

Each dialect must implement this interface for its types once. For CIR, this means adding the interface methods to types like `cir::IntType`, `cir::StructType`, and `cir::PointerType`. For FIR, it means implementing them for `fir::IntType`, `fir::RecordType`, and so on. The implementation cost is approximately 200-300 lines per dialect—a manageable one-time investment that enables reuse of thousands of lines of ABI classification logic.

### 4.4 ABIInfo Base Class

The `ABIInfo` abstract base class defines the interface for target-specific ABI classification. Each supported target (x86_64, AArch64, ARM, etc.) provides a concrete subclass that encodes that platform's calling convention rules. The core responsibility is implementing the `computeInfo()` method, which takes a `LowerFunctionInfo` object and populates it with `ABIArgInfo` classifications for each argument and the return value. This architecture allows the complexity of each ABI—which can span thousands of lines for targets like x86_64 with its intricate struct classification rules—to be isolated in dedicated implementation files.

The `computeInfo()` implementation queries type metadata through the `ABITypeInterface` methods defined in Section 4.3, enabling classification logic to work across dialects. When analyzing a function argument, the code calls methods like `type.isRecord()`, `type.getNumFields()`, and `type.getFieldType(i)` to understand the type's structure without knowing whether it's a `cir::StructType`, `fir::RecordType`, or some other dialect's representation. This interface-based approach is what makes the entire classification infrastructure dialect-agnostic.

The base class also provides common utility methods that are frequently needed across multiple targets, such as `getNaturalAlignIndirect()` for creating indirect-passing descriptors or `isPromotableIntegerTypeForABI()` for integer promotion checks. These helpers reduce code duplication and ensure consistent behavior for common patterns. The class takes a `clang::TargetInfo` reference at construction, which provides access to target-specific data like pointer size, register sizes, and platform conventions.

### 4.5 Target-Specific ABIInfo Implementations

The framework targets x86_64 System V and AArch64 PCS as initial platforms. These two targets provide valuable design validation: x86_64's chunk-based struct classification and AArch64's homogeneous aggregate detection represent fundamentally different ABI strategies, confirming that the ABITypeInterface abstraction can accommodate diverse classification approaches. Both target implementations are complete in the CIR incubator repository.

Concrete `ABIInfo` subclasses implement the classification rules for specific platforms. The `X86_64ABIInfo` class, for example, implements the x86-64 System V ABI's complex struct classification algorithm, which assigns each 8-byte chunk of a struct to register classes (Integer, SSE, X87, etc.) and then merges those classifications to determine whether the struct can be passed in registers or must go to memory. The `AArch64ABIInfo` class similarly implements the ARM Architecture Procedure Call Standard (AAPCS64), which has different rules for homogeneous floating-point aggregates and different register usage conventions.

These implementations represent thousands of lines of battle-tested code with extensive edge case handling. The x86_64 implementation alone handles over 20 distinct scenarios in its struct classification logic, covering cases like `__int128` (which passes in two integer registers), `_BitInt(N)` (which may pass indirectly depending on bit width), complex numbers (where `_Complex double` may pass in two SSE registers or via memory depending on surrounding struct members), and C++ objects with non-trivial lifecycle operations (which typically pass indirectly to enable proper copy construction and destruction). Rather than rewriting this complexity from scratch, the proposal reuses CIR's existing implementations—originally ported from Clang's `CodeGen/TargetInfo.cpp`—with targeted refactoring to replace CIR-specific type operations with `ABITypeInterface` queries.

The practical adaptation work involves identifying type casting sites (estimated at 100-200 locations across both targets) and replacing them with interface calls. For example, code that currently checks `if (auto ST = dyn_cast<cir::StructType>(Ty))` becomes `if (Ty.isa<ABITypeInterface>() && Ty.cast<ABITypeInterface>().isRecord())`. This transformation maintains the classification algorithms' correctness while making them callable from any MLIR dialect.

### 4.6 ABIRewriteContext Interface

The `ABIRewriteContext` interface is where dialect-specific code generation occurs. While the classification phase (handled by `ABIInfo`) operates purely on type metadata and is dialect-agnostic, the rewriting phase must create concrete MLIR operations—and operation creation is inherently dialect-specific. A CIR dialect needs to emit `cir.call`, `cir.cast`, and `cir.load` operations, while FIR needs `fir.call`, `fir.convert`, and `fir.load`. The `ABIRewriteContext` abstracts these differences through virtual methods for common operation patterns.

The interface defines approximately 15-20 methods covering function operations (`createFunction`, `createCall`), value manipulation (`createCast`, `createLoad`, `createStore`, `createAlloca`), type coercion (`createBitcast`, `createTrunc`, `createZExt`, `createSExt`), aggregate operations (`createExtractValue`, `createInsertValue`, `createGEP`), and housekeeping (`createFunctionType`, `replaceOp`). This set was chosen based on analyzing the operations actually needed by existing ABI lowering code: struct expansion requires extract/insert operations, indirect passing requires alloca and pointer operations, and coercion requires bitcasts and truncations.

Each dialect implementing ABI lowering must provide a concrete `ABIRewriteContext` subclass—estimated at 800-1000 lines of implementation code that wraps the dialect's builder API. This is a significant but one-time cost: CIR implements `CIRABIRewriteContext`, FIR implements `FIRABIRewriteContext`, and any future dialect reuses the shared classification infrastructure by providing its own context implementation. The alternative—reimplementing the entire ABI classification logic per dialect—would require 8,000-15,000 lines per dialect (the combined size of x86_64 and AArch64 classification code plus all supporting infrastructure), introduce divergent behavior across dialects, and create a maintenance burden where ABI bug fixes must be propagated to every dialect independently.

### 4.7 Target Registry

The `TargetABIRegistry` provides a simple factory mechanism for instantiating the correct target-specific `ABIInfo` implementation based on the target triple (e.g., `x86_64-unknown-linux-gnu` or `aarch64-apple-darwin`). When a dialect needs to perform ABI lowering, it queries the registry with the compilation target, and the registry returns the appropriate `X86_64ABIInfo`, `AArch64ABIInfo`, or other implementation. This design mirrors LLVM's existing target registry patterns and ensures that adding support for new architectures doesn't require changes to the core infrastructure or to dialect-specific code—it only requires implementing a new `ABIInfo` subclass and registering it.

The implementation is straightforward: a `createABIInfo()` method switches on the target architecture enum and constructs the corresponding concrete class. For unsupported targets, it returns `nullptr`, allowing graceful handling of architectures that haven't yet been ported. This extensibility is important for a shared infrastructure that may eventually support ARM32, RISC-V, PowerPC, and other platforms beyond the initial x86_64 and AArch64 focus.

## 5. Open Questions and Risks

### 5.1 Open Questions

1. **Should we use TypeInterface or helper class for type queries?**
   - TypeInterface is more MLIR-idiomatic but requires modifying type definitions
   - Helper class is more flexible but adds indirection
   - **Recommendation**: TypeInterface for better integration

2. **How to handle clang::TargetInfo dependency in MLIR?** ⚠️ **CRITICAL DECISION REQUIRED**

**Background**: The CIR incubator currently uses `clang::TargetInfo` (from `clang/include/clang/Basic/TargetInfo.h`) to query target-specific properties (pointer width, alignment, endianness, etc.) needed for ABI decisions. Moving this to MLIR-agnostic infrastructure raises the question: should MLIR code depend on a Clang library?

**The Issue**: 
- `clang::TargetInfo` lives in `clangBasic` library
- Creating dependency: `mlir/lib/Target/ABI/` → `clang/include/clang/Basic/`
- MLIR policy generally avoids depending on Clang (peer relationship, not hierarchical)
- However, this is target-specific infrastructure, not core MLIR

**What TargetInfo Provides** (~20-30 methods used by ABI code):
- Pointer size and alignment
- Integer/float type sizes
- Maximum alignment
- Endianness
- Calling conventions available for target
- Target triple information
- ABI-specific flags (e.g., passes objects in registers)

---

**Option A: Use llvm::Triple + MLIR DataLayoutInterface**

**Approach**: Combine existing LLVM/MLIR infrastructure:
```cpp
// Instead of clang::TargetInfo, use:
llvm::Triple triple;                      // From LLVM (arch/OS/vendor)
mlir::DataLayoutSpecInterface layout;     // From MLIR (sizes/alignments)
mlir::ModuleOp attributes;                // Target-specific properties

// Example queries:
unsigned ptrWidth = layout.getTypeSizeInBits(ptrType);
bool isLittleEndian = triple.isLittleEndian();
```

**Pros**:
- ✅ No Clang dependency (clean layering)
- ✅ Uses existing MLIR patterns (DataLayoutInterface)
- ✅ MLIR-idiomatic approach
- ✅ Works with any MLIR dialect

**Cons**:
- ⚠️ Need to define module-level attributes for ~10-15 ABI properties
- ⚠️ Upfront design work (2-3 days)
- ⚠️ Less comprehensive than TargetInfo (may need to add properties later)

**Effort**: ~3-5 days design + implementation

---

**Option B: Keep Using clang::TargetInfo**

**Approach**: Accept MLIR→Clang dependency for target-specific code:
```cpp
// Continue using what works:
const clang::TargetInfo &Target;
unsigned ptrWidth = Target.getPointerWidth(0);
bool isLittleEndian = Target.isLittleEndian();
```

**Pros**:
- ✅ Zero implementation time (already done)
- ✅ Mature, comprehensive (500+ lines of target properties)
- ✅ Battle-tested across all Clang targets
- ✅ No duplication of knowledge
- ✅ Actually target-agnostic despite the name/location

**Cons**:
- ❌ Creates MLIR→Clang dependency (architectural concern)
- ❌ May be rejected by MLIR maintainers
- ⚠️ Lives in `clang/Basic/` (naming suggests Clang-specific)

**Risk**: If rejected during review, need to pivot to Option A or C (adds 1-3 weeks delay)

---

**Option C: Minimal MLIR-Native TargetInfo**

**Approach**: Create lightweight `mlir::target::TargetInfo` abstraction:
```cpp
// mlir/include/mlir/Target/TargetInfo.h
namespace mlir::target {
class TargetInfo {
public:
  static std::unique_ptr<TargetInfo> create(llvm::Triple, DataLayoutSpec);
  
  virtual unsigned getPointerWidth(unsigned AddrSpace) const = 0;
  virtual unsigned getMaxAlignment() const = 0;
  virtual bool isLittleEndian() const = 0;
  // ... ~15-20 methods total for ABI needs
};

// Per-target implementations
class X86_64TargetInfo : public TargetInfo { ... };
class AArch64TargetInfo : public TargetInfo { ... };
}
```

**Pros**:
- ✅ No Clang dependency (clean layering)
- ✅ Tailored specifically for ABI lowering needs
- ✅ Can evolve independently

**Cons**:
- ❌ Duplicates information from clang::TargetInfo (~200 lines per target)
- ❌ More code to maintain
- ❌ Implementation effort: ~200 lines × 2 targets = 400 lines
- ⚠️ May need to sync with Clang when targets evolve

**Effort**: ~1-2 weeks implementation + testing

---

**Recommendation**: **Option A (Triple + DataLayoutInterface)** - VERIFY FEASIBILITY, then commit

**Priority Order**:
1. **Option A** (PREFERRED) - MLIR-native, architecturally correct
2. **Option C** (FALLBACK) - If Option A insufficient, create minimal MLIR TargetInfo
3. **Option B** (NOT RECOMMENDED) - MLIR→Clang dependency violates MLIR architecture principles

**Rationale**:

**Why Option A is Preferred**:
- ✅ **MLIR Independence**: Maintains MLIR as peer to Clang, not dependent
- ✅ **Architectural Correctness**: TargetInfo is input/metadata, should be expressible in MLIR
- ✅ **Reasonable Effort**: 3-5 days with clear path forward
- ✅ **MLIR-Idiomatic**: Uses DataLayoutInterface and module attributes (standard patterns)
- ✅ **Upstream Acceptance**: MLIR maintainers will approve this approach

**Why Option B is NOT Recommended**:
- ❌ **Breaks MLIR Independence**: MLIR is peer to Clang, not dependent (architectural principle)
- ❌ **Upstream Rejection Risk**: MLIR maintainers will likely request MLIR-native approach
- ❌ **Wrong Precedent**: `mlir/lib/Target/` dependencies should be for output formats (LLVM IR, SPIR-V), not input metadata
- ⚠️ **False Economy**: Zero implementation time now, but redesign later if rejected

**Why Option C is Acceptable Fallback**:
- ✅ **Architecturally Sound**: MLIR-native, clean layering
- ✅ **Tailored for ABI**: Only ~15-20 methods needed (not 500+ like clang::TargetInfo)
- ✅ **Upstream Acceptable**: MLIR maintainers will approve
- ⚠️ **Higher Effort**: 1-2 weeks vs 3-5 days for Option A
- ⚠️ **Duplication**: Some overlap with clang::TargetInfo knowledge

**MLIR Architect Perspective**:
> "MLIR's mission is to be reusable by Rust, Julia, Swift, etc. without requiring Clang. TargetInfo is metadata/input (not an output format like LLVM IR), so it should be expressible in MLIR. Option B breaks this principle. I would request changes in upstream review."

**Decision Timeline**:
- **Weeks 1-2 (Validation Phase - Days 1-10)**: Complete all audits and prototype
  - Audit actual TargetInfo usage in CIR incubator
  - Generate concrete list of methods/properties needed
  - Identify which are covered by DataLayout vs need attributes
  - Design Option A with concrete module attributes
  - Define exact attribute schema (names, types, defaults)
  - Prototype with x86_64 ABI queries
  - Validate DataLayoutInterface provides what we need
- **End of Week 2 (Day 10)**: Go/No-Go Decision
  - ✅ **If Option A is sufficient** → Commit to Option A, proceed to Phase 1
  - ❌ **If Option A has gaps** → Assess: can we add attributes? Or need Option C?
  - 🔴 **If Option C required AND adds >2 weeks** → Pivot to Strategy 1 (graduate with current impl)

**Weeks 1-2 Exit Criteria (Validation Phase)**:
```
[ ] Complete audit of TargetInfo usage (concrete method list)
[ ] Audit CIR coupling depth (count dyn_cast<cir::Type> sites)
[ ] Audit ABITypeInterface requirements (list exact methods needed)
[ ] Audit ABIRewriteContext requirements (list exact methods needed)
[ ] Draft module attribute schema for Option A
[ ] Prototype Option A with 1 target (x86_64) proving feasibility
[ ] Ask Andy: Is varargs required for graduation?
[ ] Decision: A (commit) or C (fallback) or Strategy 1 (pivot)
[ ] Apply Weeks 1-2 Pivot Thresholds (Green/Yellow/Red)
```

**Weeks 1-2 Pivot Thresholds** (Go/No-Go Decision):

**🟢 GREEN (Proceed with Strategy 2)**:
- TargetInfo usage: ≤30 methods → Option A feasible
- CIR coupling: ≤250 type cast sites → Phase 3 on schedule
- Interface complexity: ≤20 methods per interface → Phase 2 on schedule
- Varargs: Deferred (confirmed by Andy)
- **Total Additional Risk**: ≤2 weeks → 15-17 week timeline acceptable → **PROCEED**

**🟡 YELLOW (Proceed with Caution)**:
- TargetInfo usage: 31-40 methods → Option A challenging, might need Option C
- CIR coupling: 251-350 sites → Phase 3 +1 week
- Interface complexity: 21-25 methods → Phase 2 +0.5 weeks
- Varargs: Required for graduation (likely)
- **Total Additional Risk**: 2.5-4 weeks → 17-19 week timeline → **PROCEED WITH BUFFER**

**🔴 RED (Pivot to Strategy 1)**:
- TargetInfo usage: >40 methods → Option C required (+2 weeks)
- CIR coupling: >350 sites → Phase 3 +2 weeks
- Interface complexity: >25 methods → Phase 2 +1 week
- Multiple blockers simultaneously
- **Total Additional Risk**: >4 weeks → 19-21 week timeline → **PIVOT TO STRATEGY 1**

**Strategy 1 Pivot**: Graduate with current CIR-specific implementation, refactor upstream later

**Fallback Strategy**:
If Option A requires Option C, and Option C adds >2 weeks to timeline (total >3 weeks for TargetInfo resolution), consider graduating with current CIR-specific implementation and refactoring upstream (Strategy 1 pivot).

3. **Where should code be located?**

**ABITypeInterface**:
- **Location**: `mlir/include/mlir/Interfaces/ABI/ABITypeInterface.td`
- **Rationale**: Cross-dialect interface, follows MLIR convention

**ABIArgInfo, LowerFunctionInfo, ABIRewriteContext** (shared structures):
- **Location**: `mlir/include/mlir/Interfaces/ABI/`
- **Rationale**: Shared data structures and interfaces used by all dialects

**ABIInfo, Target Implementations**:
- **Location**: `mlir/include/mlir/Target/ABI/`
- **Rationale**: Target-specific classification logic, matches MLIR precedent
- **Precedent**: `mlir/include/mlir/Target/LLVMIR/`, `mlir/include/mlir/Target/SPIRV/`
- **MLIR Convention**: `Interfaces/` is for cross-dialect, `Target/` is for target-specific

**Recommendation**: This split follows MLIR conventions correctly

4. **ABIRewriteContext vs OpBuilder + Interfaces?** ⚠️ **TO BE VALIDATED IN WEEK 1**

**Current Design**: Custom `ABIRewriteContext` interface for dialect-specific operations

**MLIR Architect Concern**: MLIR already has operation abstractions (`OpBuilder`, `FunctionOpInterface`, `CallOpInterface`)

**Alternative Approach**:
```cpp
// Instead of custom ABIRewriteContext:
// Use existing MLIR interfaces + OpBuilder directly
template<typename FuncOpT, typename CallOpT>
  requires FunctionOpInterface<FuncOpT> && CallOpInterface<CallOpT>
class ABILowering {
  OpBuilder &builder;
  // No virtual calls, use concrete types
};
```

**Week 1 Task**: Prototype both approaches
- **Option 1**: Custom ABIRewriteContext (current design)
- **Option 2**: OpBuilder + existing interfaces (template-based)
- **Decision Criteria**: Code clarity, maintainability, performance

**Not a Blocker**: Both approaches work. Choose based on prototype results.

5. **How to coordinate with FIR team?**
   - When to engage them?
   - Who owns the shared infrastructure?
   - **Recommendation**: Build CIR-first, engage FIR team at Phase 7 (after CIR proven)

### 5.2 Risks

**Risk 1: TargetInfo Dependency Rejected** ⚠️ **CRITICAL**
- **Impact**: High (could add 1-3 weeks to timeline)
- **Probability**: Medium (30-40%)
- **Description**: MLIR maintainers may reject `clang::TargetInfo` dependency, requiring MLIR-native implementation
- **Mitigation**: 
  - Weeks 1-2 (Validation Phase): Design MLIR-native alternative (Option A)
  - Get early feedback from Andy and MLIR maintainers
  - Audit actual TargetInfo usage to minimize required functionality
  - Have fallback implementation ready
- **Fallback**: If adds >2 weeks, pivot to Strategy 1 (graduate with current implementation)

**Risk 2: GSoC Library Divergence**
- **Impact**: Medium
- **Probability**: Medium
- **Description**: Parallel development with GSoC project could create incompatible approaches
- **Mitigation**: Stay in contact with GSoC author, plan integration path, share design early

**Risk 3: Performance Overhead**
- **Impact**: High (if > 10% overhead)
- **Probability**: Low
- **Description**: Abstraction layers could introduce unacceptable compile-time overhead
- **Mitigation**: Profile early, optimize hot paths, consider caching, benchmark against classic codegen

**Risk 4: Incomplete Target Support Blocks Graduation** ⚠️ **HIGH PROBABILITY**
- **Impact**: High (blocks graduation)
- **Probability**: **High (70-80%)** - varargs likely required
- **Description**: Missing features (varargs, complex types) may be required for graduation
- **Specific Issue**: CIR incubator has many `NYI` for varargs; ~40% of C code uses printf/scanf
- **Mitigation**: 
  - **Week 1**: Ask Andy explicitly: "Is varargs required for graduation?"
  - Budget 17 weeks (not 15) to account for likely varargs requirement
  - Have varargs implementation plan ready (2-3 weeks)
  - Focus on x86_64/AArch64 Linux (80% of use cases)
  - Document limitations clearly if varargs deferred

**Risk 5: Breaking Changes in MLIR**
- **Impact**: Medium
- **Probability**: Low
- **Description**: MLIR interface changes could break our implementation
- **Mitigation**: Follow MLIR development, use stable interfaces, engage with MLIR community

**Risk 6: Complexity Underestimation**
- **Impact**: High (timeline slip)
- **Probability**: Medium
- **Description**: Edge cases and corner cases in ABI handling are complex
- **Mitigation**: Incremental development, frequent validation against classic codegen, comprehensive testing

## 6. Appendices

### A. Glossary

- **ABI**: Application Binary Interface
- **CC**: Calling Convention
- **CIR**: Clang Intermediate Representation (MLIR-based)
- **FIR**: Fortran Intermediate Representation (MLIR-based)
- **HFA**: Homogeneous Floating-point Aggregate (ARM term)
- **HVA**: Homogeneous Short-Vector Aggregate (ARM term)
- **NYI**: Not Yet Implemented
- **PCS**: Procedure Call Standard (ARM term)
- **RVO**: Return Value Optimization

### B. File Structure Summary

```
mlir/
├── include/mlir/Interfaces/ABI/
│   ├── ABITypeInterface.td
│   ├── ABIArgInfo.h
│   ├── LowerFunctionInfo.h
│   └── ABIRewriteContext.h
├── include/mlir/Target/ABI/
│   ├── ABIInfo.h
│   └── TargetRegistry.h
├── lib/Interfaces/ABI/
│   ├── ABIArgInfo.cpp
│   ├── LowerFunctionInfo.cpp
│   └── CMakeLists.txt
└── lib/Target/ABI/
    ├── ABIInfo.cpp
    ├── TargetRegistry.cpp
    ├── X86/
    │   ├── X86_64ABIInfo.h/cpp
    │   └── CMakeLists.txt
    ├── AArch64/
    │   ├── AArch64ABIInfo.h/cpp
    │   └── CMakeLists.txt
    └── CMakeLists.txt

clang/lib/CIR/Dialect/Transforms/TargetLowering/
├── CallConvLowering.cpp         # CIR-specific pass
├── CIRABIRewriteContext.h/cpp   # CIR operation rewriting
└── CMakeLists.txt
```

### C. Implementation Checklist

**Phase 1: Infrastructure**
- [ ] Create directory structure
- [ ] Move ABIArgInfo
- [ ] Define ABITypeInterface
- [ ] Define ABIRewriteContext
- [ ] Setup build system

**Phase 2: CIR Integration**
- [ ] Implement ABITypeInterface for CIR types
- [ ] Implement CIRABIRewriteContext
- [ ] Add type query tests

**Phase 3: Target ABI**
- [ ] Extract X86_64ABIInfo
- [ ] Extract AArch64ABIInfo
- [ ] Create TargetRegistry
- [ ] Add classification tests

**Phase 4: Lowering Pass**
- [ ] Create CallConvLowering pass
- [ ] Function signature rewriting
- [ ] Call site rewriting
- [ ] Value coercion
- [ ] Integration tests

**Phase 5: Testing**
- [ ] Port CIR tests
- [ ] ABI compliance tests
- [ ] Performance benchmarks
- [ ] Bug fixes

**Phase 6: Varargs**
- [ ] x86_64 varargs implementation
- [ ] AArch64 varargs implementation
- [ ] Varargs tests (60+)

**Phase 7: Documentation**
- [ ] API documentation
- [ ] User guide
- [ ] Target guide
- [ ] Design document

---

**Contact**: Adam Smith (CIR Team)  
**Last Updated**: January 2026
