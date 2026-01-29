# ClangIR ABI Lowering - Design Document

**Version**: 1.0  
**Date**: January 2026  
**Authors**: Adam Smith (CIR Team)  
**Status**: Complete Specification - Ready for Implementation  
**Target**: x86_64 and AArch64 (primary), extensible to other targets

---

## Quick Start: How to Read This Document

**If you have 5 minutes**: Read Section I (Executive Summary)  
**If you have 30 minutes**: Read Section I (Executive Summary) + Section V (Implementation Phases)  
**If you have 2 hours**: Read the entire document  
**If you're implementing**: Focus on Section IV (Architecture) and Section V (Phases)  
**If you're reviewing for approval**: Focus on Section X (Open Questions) and Section XI (Success Metrics)  
**If you're new to MLIR**: Read Section II (Background) first

---

## Document Purpose

This document proposes a comprehensive design for creating an MLIR-agnostic calling convention lowering framework. The framework will:
1. Enable CIR to perform ABI-compliant calling convention lowering
2. Be reusable by other MLIR dialects (FIR, future dialects)
3. Achieve parity with CIR incubator implementation for x86_64 and AArch64
4. Integrate with or inform the GSoC ABI Lowering Library project

## I. Executive Summary

### 1.1 Problem Statement
- Calling convention lowering is currently duplicated per-dialect
- CIR incubator has partial implementation but CIR-specific
- FIR and future dialects need similar functionality
- Classic Clang codegen can't be reused directly (AST/LLVM IR specific)

### 1.2 Proposed Solution
Three-layer architecture:
1. **Layer 1 (Dialect-Agnostic)**: Pure ABI classification logic
2. **Layer 2 (Interface-Based)**: Type and layout abstractions
3. **Layer 3 (Dialect-Specific)**: Operation rewriting per dialect

### 1.3 Key Benefits
- Avoids duplicating complex ABI logic across dialects
- Maintains correct ABI compliance for all targets
- Enables easier testing and validation
- Provides migration path from CIR incubator

### 1.4 Success Criteria
- CIR can lower x86_64 and AArch64 calling conventions correctly
- FIR can adopt the same infrastructure
- Test suite validates ABI compliance
- Performance overhead < 5% vs direct implementation

## II. Background and Context

### 2.1 What is Calling Convention Lowering?

**Definition**: Transform high-level function signatures to match target ABI requirements.

**Example** (x86_64 System V ABI):
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

- **Target-specific**: Each architecture has different rules
- **Type-dependent**: Rules differ for integers, floats, structs, unions, etc.
- **Context-sensitive**: Varargs, virtual calls, special calling conventions
- **ABI versions**: Same target may have multiple ABI variants

### 2.3 Existing Implementations

#### Classic Clang CodeGen
- **Location**: `clang/lib/CodeGen/`
- **Approach**: AST → LLVM IR during codegen
- **Pros**: Mature, handles all targets, well-tested
- **Cons**: Tightly coupled to Clang AST and LLVM IR

#### CIR Incubator
- **Location**: `clang/lib/CIR/Dialect/Transforms/TargetLowering/`
- **Approach**: CIR ops → ABI-lowered CIR ops (MLIR pass)
- **Pros**: Works with MLIR, adapted classic logic
- **Cons**: CIR-specific types and operations

#### GSoC ABI Lowering Library (WIP)
- **Status**: PR #140112, not yet merged
- **Approach**: Independent ABI type system, extracted from Clang
- **Pros**: Frontend-agnostic, reusable
- **Cons**: Still in development, Clang/LLVM IR focused

### 2.4 Requirements for MLIR Dialects

**CIR Needs**:
- Lower C/C++ calling conventions correctly
- Support x86_64 and AArch64 initially
- Handle structs, unions, complex types
- Support instance methods, virtual calls

**FIR Needs** (future):
- Lower Fortran calling conventions
- Handle Fortran-specific types (complex, derived types)
- Support Fortran calling semantics

**Common Needs**:
- Target ABI compliance
- Efficient lowering (minimal overhead)
- Extensibility for new targets
- Testability and validation

### 2.4.1 Fortran-Specific Considerations (FIR)

**Context**: FIR team (NVIDIA Fortran frontend) will be a major consumer of this infrastructure. Fortran has unique type system features and ABI semantics that differ from C/C++.

**Fortran Types**:

1. **Derived Types** (Fortran's version of structs):
   ```fortran
   type :: MyType
     integer :: field1
     real :: field2
     type(OtherType) :: field3  ! Nested derived type
   end type
   ```
   - **Handling**: Similar to C structs; ABITypeInterface `getNumFields()`, `getFieldType()`, `getFieldOffsetInBits()` should work
   - **Status**: ✅ Covered by existing design

2. **COMPLEX Types**:
   ```fortran
   complex :: z  ! 2 floats (real part + imaginary part)
   ```
   - **Handling**: Struct of 2 floats; ABITypeInterface includes `isComplexType()` + `getComplexElementType()` methods
   - **Status**: ✅ Added in interface design

3. **CHARACTER Types** (with hidden length parameter):
   ```fortran
   subroutine foo(str)
     character(len=*) :: str  ! str is passed + hidden length parameter
   end subroutine
   ```
   - **Fortran ABI Quirk**: Character strings are passed with TWO arguments:
     1. Pointer to string data (explicit)
     2. Hidden length parameter (integer, passed AFTER all explicit args)
   - **Example**: `foo(x, str, y)` → lowered to `foo(x, str_data, y, str_len)`
   - **Challenge**: ABIRewriteContext must support hidden argument insertion at arbitrary positions
   - **Status**: ⚠️ **Week 4 FIR check-in will design solution**

4. **Arrays** (descriptor-based, not C-style):
   ```fortran
   real, dimension(:,:) :: matrix  ! Allocatable, rank-2
   ```
   - **Fortran Reality**: Arrays have **descriptors** (hidden metadata: bounds, strides, pointer to data)
   - Descriptor is passed, not the array itself
   - **Challenge**: How to represent descriptor in ABITypeInterface?
   - **Options**: 
     - A) Add descriptor-specific methods (`isDescriptorType()`, `getDescriptorElementType()`)
     - B) Treat as opaque struct (don't expose internals to ABI classification)
   - **Status**: ⚠️ **Week 4 FIR check-in will decide approach**

**Fortran ABI Semantics**:

1. **Default Pass-by-Reference**:
   - C/C++: Small types passed by value, large types by pointer
   - **Fortran**: EVERYTHING passed by reference (except `INTENT(IN) VALUE`)
   ```fortran
   subroutine foo(x)
     integer :: x  ! Passed by REFERENCE (pointer to integer)
   end subroutine
   ```
   - **Handling**: ABIArgInfo `Indirect` kind (already exists)
   - **Status**: ✅ Should work (FIR classifies everything as `Indirect` by default)

2. **CHARACTER Hidden Length Argument Reordering**:
   - gfortran ABI: CHARACTER lengths passed AFTER all explicit args
   - Requires non-trivial argument reordering
   - **Requires**: ABIRewriteContext extension for hidden arguments
   - **Status**: ⚠️ **Design TBD in Week 4**

**FIR Integration Estimate**:
- **Per-Dialect Cost**: 1,000-1,200 lines (vs 800-1,000 for dialects without hidden args)
- **Why Higher**: CHARACTER + descriptor handling, type-bound procedures
- **FIR Types to Implement**: 8-10 types (IntegerType, RealType, LogicalType, ComplexType, CharacterType, RecordType, SequenceType, BoxType, PointerType, ReferenceType)

**Testing Challenges**:
- **No "Classic Fortran Codegen" Baseline**: Unlike CIR (compare with classic Clang), FIR has no equivalent
- **Validation Approach**: Differential testing against `gfortran` or `ifort`
- **Test Coverage**: 50-100 Fortran-specific test cases (CHARACTER, arrays, derived types, COMPLEX, interop with C)

**Week 4 Validation Will Determine**:
- Feasibility of CHARACTER hidden length mechanism
- Array descriptor representation approach
- Whether ABITypeInterface/ABIRewriteContext need Fortran-specific extensions

## III. Design Overview

### 3.1 Architecture Diagram

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

### 3.2 Three-Layer Design

**Layer 1: Pure ABI Classification**
- Input: mlir::Type + metadata
- Output: ABIArgInfo (how to pass)
- No dialect knowledge
- Target-specific algorithms

**Layer 2: Type/Layout Abstraction**
- ABITypeInterface for type queries
- DataLayoutInterface (MLIR standard)
- ABIArgInfo, LowerFunctionInfo data structures
- Target info access

**Layer 3: Dialect-Specific Rewriting**
- ABIRewriteContext interface
- Dialect implements operation creation
- Pass infrastructure per dialect
- Value coercion, temporary allocation

### 3.3 Key Components

1. **ABIArgInfo**: Classification result (Direct, Indirect, Expand, etc.)
2. **LowerFunctionInfo**: Classified function signature
3. **ABITypeInterface**: Type queries for ABI decisions
4. **ABIInfo**: Target-specific classification logic
5. **ABIRewriteContext**: Dialect-specific operation rewriting
6. **TargetRegistry**: Maps target triple to ABI implementation

## IV. Detailed Component Design

### 4.1 ABIArgInfo (Already Exists in CIR)

**Location**: `mlir/include/mlir/Interfaces/ABI/ABIArgInfo.h`

**Purpose**: Describes how a single argument or return value should be passed.

**Structure**:
```cpp
class ABIArgInfo {
  enum Kind {
    Direct,          // Pass directly (possibly coerced)
    Extend,          // Pass with sign/zero extension
    Indirect,        // Pass via hidden pointer
    IndirectAliased, // Pass indirectly, may alias
    Ignore,          // Ignore (empty struct/void)
    Expand,          // Expand into constituent fields
    CoerceAndExpand, // Coerce and expand
    InAlloca         // Windows inalloca
  };
  
  mlir::Type CoerceToType;  // Target type for coercion
  mlir::Type PaddingType;   // Padding type if needed
  // Flags: InReg, CanBeFlattened, SignExt, etc.
};
```

**Status**: ✅ Exists in CIR, already dialect-agnostic, just needs to be moved.

### 4.2 LowerFunctionInfo

**Location**: `mlir/include/mlir/Interfaces/ABI/LowerFunctionInfo.h`

**Purpose**: Represents function signature with ABI classification for each argument/return.

**Structure**:
```cpp
class LowerFunctionInfo {
  struct ArgInfo {
    mlir::Type originalType;
    ABIArgInfo abiInfo;
  };
  
  unsigned CallingConvention;
  unsigned EffectiveCallingConvention;
  RequiredArgs Required;  // For varargs
  
  // Return type at index 0, args follow
  SmallVector<ArgInfo> Args;
};
```

**Methods**:
```cpp
ABIArgInfo &getReturnInfo();
mlir::Type getReturnType();
unsigned getNumArgs();
ABIArgInfo &getArgInfo(unsigned i);
mlir::Type getArgType(unsigned i);
```

**Status**: 🔄 Exists in CIR, needs minor adaptation for MLIR-agnostic use.

### 4.3 ABITypeInterface

**Location**: `mlir/include/mlir/Interfaces/ABI/ABITypeInterface.td`

**Purpose**: Provides type queries needed for ABI classification.

**Interface Definition** (TableGen):

> **TableGen Syntax Note**: `InterfaceMethod<description, return_type, method_name, parameters>` defines a polymorphic method that types can implement. `(ins)` means no parameters. This generates C++ virtual methods that each type overrides.

```cpp
def ABITypeInterface : TypeInterface<"ABITypeInterface"> {
  let methods = [
    // Basic type queries
    InterfaceMethod<"Check if type is an integer",
      "bool", "isInteger", (ins)>,
    InterfaceMethod<"Check if type is a record (struct/class)",
      "bool", "isRecord", (ins)>,
    InterfaceMethod<"Check if type is a pointer",
      "bool", "isPointer", (ins)>,
    InterfaceMethod<"Check if type is floating point",
      "bool", "isFloatingPoint", (ins)>,
    InterfaceMethod<"Check if type is an array",
      "bool", "isArray", (ins)>,
      
    // Type navigation
    InterfaceMethod<"Get pointee type for pointers",
      "mlir::Type", "getPointeeType", (ins)>,
    InterfaceMethod<"Get element type for arrays",
      "mlir::Type", "getElementType", (ins)>,
      
    // Size and alignment queries
    InterfaceMethod<"Get type size in bits",
      "uint64_t", "getSizeInBits", (ins "mlir::DataLayout", "$layout")>,
    InterfaceMethod<"Get ABI alignment in bits",
      "uint32_t", "getABIAlignmentInBits", (ins "mlir::DataLayout", "$layout")>,
    InterfaceMethod<"Get preferred alignment in bits",
      "uint32_t", "getPreferredAlignmentInBits", (ins "mlir::DataLayout", "$layout")>,
      
    // Record (struct/class) queries - CRITICAL FOR ABI CLASSIFICATION
    InterfaceMethod<"Get number of fields in record",
      "unsigned", "getNumFields", (ins)>,
    InterfaceMethod<"Get field type by index",
      "mlir::Type", "getFieldType", (ins "unsigned", "$index")>,
    InterfaceMethod<"Get field offset in bits",
      "uint64_t", "getFieldOffsetInBits", 
      (ins "unsigned", "$index", "mlir::DataLayout", "$layout")>,
    InterfaceMethod<"Check if record is empty (no fields)",
      "bool", "isEmpty", (ins)>,
      
    // Additional methods for ABI decisions
    InterfaceMethod<"Check if integer type is signed",
      "bool", "isSignedInteger", (ins)>,
    InterfaceMethod<"Get integer width in bits",
      "unsigned", "getIntegerBitWidth", (ins)>,
    
    // Additional methods that may be needed for edge cases (15-25 total)
    InterfaceMethod<"Check if type is a union",
      "bool", "isUnion", (ins)>,
    InterfaceMethod<"Check if type is complex",
      "bool", "isComplexType", (ins)>,
    InterfaceMethod<"Get complex element type",
      "mlir::Type", "getComplexElementType", (ins)>,
    
    // x86_64-specific edge cases (CRITICAL for ABI correctness)
    InterfaceMethod<"Check if type is __int128",
      "bool", "isInt128", (ins)>,
    InterfaceMethod<"Check if type is _BitInt(N)",
      "bool", "isBitInt", (ins)>,
    InterfaceMethod<"Get _BitInt width",
      "unsigned", "getBitIntWidth", (ins)>,
    
    // C++ ABI support (required if targeting C++)
    InterfaceMethod<"Has non-trivial copy constructor",
      "bool", "hasNonTrivialCopyCtor", (ins)>,
    InterfaceMethod<"Has non-trivial destructor",
      "bool", "hasNonTrivialDtor", (ins)>,
    InterfaceMethod<"Check if type is trivially copyable",
      "bool", "isTriviallyCopyable", (ins)>,
    InterfaceMethod<"Check if type is vector",
      "bool", "isVectorType", (ins)>,
    InterfaceMethod<"Get vector element count",
      "unsigned", "getVectorNumElements", (ins)>,
  ];
  
  let description = [{
    Interface for types to provide ABI-relevant information.
    
    Key Design Notes:
    - Field iteration (getNumFields, getFieldType, getFieldOffsetInBits) is 
      CRITICAL for struct classification in x86_64 and AArch64 ABIs
    - DataLayout is passed to size/alignment queries to support target-specific layouts
    - Not all types implement all methods (e.g., integers don't have fields)
    
    **Method Count**: 15-20 methods shown, potentially 20-25 with edge cases
    
    **Additional Methods That May Be Needed**:
    - Union handling (isUnion, getActiveUnionMember)
    - Complex types (isComplexType, getComplexElementType) - shown above
    - Vector types (isVectorType, getVectorNumElements) - shown above
    - Flexible array members (isVariablySized)
    - Padding queries (hasPaddingBetweenFields)
    
    **Week 1 Task**: Audit x86_64/AArch64 classification code to determine exact method list
  }];
}
```

**Dialects Implement**:
```cpp
// CIR
class IntType : public Type<IntType, ..., ABITypeInterface::Trait> {
  bool isInteger() { return true; }
  bool isRecord() { return false; }
  // ...
};

// FIR
class fir::IntType : public Type<fir::IntType, ..., ABITypeInterface::Trait> {
  bool isInteger() { return true; }
  // ...
};
```

**Status**: ✨ New, needs to be created.

### 4.4 ABIInfo Base Class

**Location**: `mlir/lib/Target/ABI/ABIInfo.h`

**Purpose**: Abstract base for target-specific ABI classification.

**Structure**:
```cpp
class ABIInfo {
protected:
  const clang::TargetInfo &Target;
  
public:
  explicit ABIInfo(const clang::TargetInfo &Target);
  virtual ~ABIInfo();
  
  // Pure virtual - must implement per target
  virtual void computeInfo(LowerFunctionInfo &FI) const = 0;
  
  // Helpers
  ABIArgInfo getNaturalAlignIndirect(mlir::Type Ty, mlir::DataLayout &DL);
  bool isPromotableIntegerTypeForABI(mlir::Type Ty);
};
```

**Status**: 🔄 Exists in CIR, needs adaptation to remove CIR-specific dependencies.

### 4.5 Target-Specific ABIInfo Implementations

**Location**: `mlir/lib/Target/ABI/X86/`, `mlir/lib/Target/ABI/AArch64/`

**Example: X86_64ABIInfo**:
```cpp
class X86_64ABIInfo : public ABIInfo {
  enum Class { Integer, SSE, SSEUp, X87, X87Up, NoClass, Memory };
  
  void classify(mlir::Type Ty, uint64_t offset, Class &Lo, Class &Hi);
  Class merge(Class A, Class B);
  
public:
  ABIArgInfo classifyReturnType(mlir::Type Ty);
  ABIArgInfo classifyArgumentType(mlir::Type Ty, ...);
  
  void computeInfo(LowerFunctionInfo &FI) const override;
};
```

**Status**: 🔄 Exists in CIR, needs minor adaptation (remove CIR type casts, use ABITypeInterface).

### 4.6 ABIRewriteContext Interface

**Location**: `mlir/include/mlir/Interfaces/ABI/ABIRewriteContext.h`

**Purpose**: Dialect-specific callbacks for operation rewriting.

**Interface**:
```cpp
class ABIRewriteContext {
public:
  virtual ~ABIRewriteContext() = default;
  
  // Operation creation
  virtual Operation *createFunction(
      Location loc, StringRef name, FunctionType type) = 0;
  
  virtual Operation *createCall(
      Location loc, Value callee, TypeRange results, ValueRange args) = 0;
  
  virtual Value createCast(
      Location loc, Value value, Type targetType) = 0;
  
  virtual Value createLoad(Location loc, Value ptr) = 0;
  virtual void createStore(Location loc, Value value, Value ptr) = 0;
  
  virtual Value createAlloca(Location loc, Type type, unsigned align) = 0;
  
  // Value coercion (CRITICAL for ABI lowering)
  virtual Value createBitcast(
      Location loc, Value value, Type targetType) = 0;
  
  virtual Value createTrunc(
      Location loc, Value value, Type targetType) = 0;
  
  virtual Value createZExt(
      Location loc, Value value, Type targetType) = 0;
  
  virtual Value createSExt(
      Location loc, Value value, Type targetType) = 0;
  
  // Aggregate operations (CRITICAL for struct expansion)
  virtual Value createExtractValue(
      Location loc, Value aggregate, ArrayRef<unsigned> indices) = 0;
  
  virtual Value createInsertValue(
      Location loc, Value aggregate, Value element, 
      ArrayRef<unsigned> indices) = 0;
  
  virtual Value createGEP(
      Location loc, Value ptr, ArrayRef<Value> indices) = 0;
  
  // Type conversion
  virtual FunctionType createFunctionType(
      ArrayRef<Type> inputs, ArrayRef<Type> results) = 0;
  
  // Operation replacement
  virtual void replaceOp(Operation *old, Operation *new_op) = 0;
};
```

**Implementation Complexity**: **HIGH**
- 15-20 methods total (not just 5-6 shown in original design)
- Each dialect must implement all methods
- Per-dialect cost: ~800-1000 lines (revised from 500)

**Dialect Implements**:
```cpp
class CIRABIRewriteContext : public ABIRewriteContext {
  OpBuilder &builder;
  
  Operation *createFunction(...) override {
    return builder.create<cir::FuncOp>(...);
  }
  // ... other CIR-specific implementations
};
```

**Status**: ✨ New, needs to be created.

### 4.7 Target Registry

**Location**: `mlir/lib/Target/ABI/TargetRegistry.h`

**Purpose**: Map target triple to ABIInfo implementation.

**Interface**:
```cpp
class TargetABIRegistry {
public:
  static std::unique_ptr<ABIInfo> createABIInfo(
      const llvm::Triple &triple,
      const clang::TargetInfo &targetInfo);
  
private:
  // Factory functions
  static std::unique_ptr<ABIInfo> createX86_64ABIInfo(...);
  static std::unique_ptr<ABIInfo> createAArch64ABIInfo(...);
};
```

**Implementation**:
```cpp
std::unique_ptr<ABIInfo> TargetABIRegistry::createABIInfo(
    const llvm::Triple &triple,
    const clang::TargetInfo &targetInfo) {
  
  switch (triple.getArch()) {
  case llvm::Triple::x86_64:
    return createX86_64ABIInfo(targetInfo);
  case llvm::Triple::aarch64:
    return createAArch64ABIInfo(targetInfo);
  default:
    return nullptr;  // Unsupported target
  }
}
```

**Status**: ✨ New, straightforward to create.

## V. Implementation Phases

### Implementation Timeline & Risk Assessment

**Baseline Timeline**: 13 weeks (aggressive)  
**Realistic Timeline**: 15 weeks (with contingency)  
**With Varargs**: 17 weeks (if required for graduation)

**Risk Factors**:
1. CIR coupling depth: 100-200 type cast sites expected, could be 300-400 (+0.5-1 week)
2. ABITypeInterface complexity: 15-20 methods with field iteration (+0.5 week)
3. ABIRewriteContext complexity: 15-20 methods needed vs 5-6 shown (+0.5 week)
4. Testing infrastructure: Differential testing setup takes time (+1 week)

**Contingency Recommendation**: Budget 15-16 weeks (20% buffer over 13 week baseline)

---

### Phase 1: Infrastructure Setup (Weeks 1-2)
1. Create directory structure in `mlir/include/mlir/Interfaces/ABI/` and `mlir/include/mlir/Target/ABI/`
2. Move ABIArgInfo from CIR to shared location
3. Adapt LowerFunctionInfo for MLIR-agnostic use
4. Define ABITypeInterface in TableGen
5. Create ABIRewriteContext interface
6. Set up build system (CMakeLists.txt)

**Deliverable**: Compiling but empty infrastructure

### Phase 2: CIR Integration - Type Interface (Weeks 3-4)
1. Implement ABITypeInterface for CIR types
   - cir::IntType, cir::BoolType
   - cir::RecordType
   - cir::PointerType
   - cir::ArrayType
   - cir::FuncType
   - cir::FloatType, cir::DoubleType
2. Test type queries
3. Implement CIRABIRewriteContext

**Deliverable**: CIR types implement ABITypeInterface

**Implementation Notes**:
- Must implement 15-20 methods per type (not just basic queries)
- Field iteration for RecordType is critical and potentially complex
- Estimated 1.5-2 weeks (upper end of range due to interface complexity)

### Phase 3: Extract Target ABI Logic (Weeks 5-7)
1. Move X86_64ABIInfo from CIR to `mlir/lib/Target/ABI/X86/`
2. Replace CIR type casts with ABITypeInterface queries
3. Move AArch64ABIInfo similarly
4. Create TargetABIRegistry
5. Add unit tests for classification

**Deliverable**: Target ABI logic is MLIR-agnostic

**Implementation Notes**:
- Expected: 100-200 `dyn_cast<cir::Type>` replacement sites
- Risk: Could be 300-400 sites if coupling deeper than expected
- Each site must be refactored to use ABITypeInterface
- Estimated 3-3.5 weeks (upper end if coupling is deeper)

### Phase 4: CIR Calling Convention Pass (Weeks 8-10)
1. Create new CallConvLowering pass using shared infrastructure
2. Implement function signature rewriting
3. Implement call site rewriting
4. Handle value coercion (direct, indirect, expand)
5. Add integration tests

**Deliverable**: CIR can lower calling conventions using shared infrastructure

### Phase 5: Testing and Validation (Weeks 11-12)

**Duration**: 2-3 weeks

**Testing Strategy Definition**:

1. **Differential Testing** (1 week setup + ongoing):
   - Create harness to compare CIR output with classic Clang codegen
   - Assembly-level comparison for ABI compliance
   - Automated regression detection

2. **ABI Compliance Tests** (1 week):
   - Port existing ABI test suites (x86_64 System V, AArch64 PCS)
   - Create **500+ systematic test cases** covering:
     - **x86_64 System V** (250+ tests):
       - Basic types: int, float, pointer, __int128, _BitInt(20 tests)
       - Structs: 1-byte, 2-byte, 4-byte, 8-byte, 9-byte, 16-byte (varying sizes/alignments) (100 tests)
       - Unions: FP+integer, multiple FP, nested unions (30 tests)
       - Arrays: Fixed-size, multi-dimensional (20 tests)
       - Edge cases: empty structs, __int128 vs _BitInt, bitfields, over-aligned (50 tests)
       - Varargs: printf/scanf edge cases (30 tests, if varargs implemented)
     - **AArch64 PCS** (250+ tests):
       - Basic types (20 tests)
       - HFA/HVA detection: 1-5 fields, nested, mixed types (80 tests - CRITICAL)
       - Structs: various sizes and alignments (80 tests)
       - Over-alignment: 16, 32, 64-byte aligned structs (30 tests)
       - Edge cases: empty structs, padding (40 tests)
   - **Differential Tests** (100+ tests):
     - Real-world struct layouts from open-source projects
     - Compare assembly output with classic Clang
   - **Interop Tests** (50+ tests):
     - Actual C→CIR→C function calls
     - Runtime binary compatibility verification

3. **Performance Benchmarks** (3-5 days):
   - Compilation time overhead measurement
   - Generated code quality comparison
   - 10-20 representative benchmarks

4. **C++ Non-Trivial Types Testing** (Phase 2 only, 20 tests):
   - Copy constructors (passed by value → call copy constructor)
   - Destructors (temporary destruction)
   - Deleted copy constructors (must pass by reference)
   - Move-only types (std::unique_ptr, etc.)
   - Note: Phase 1 is C-only; this testing applies to Phase 2 C++ support

5. **Bug Fixing & Iteration** (1-2 weeks):
   - Fix issues discovered by tests
   - Handle edge cases
   - Performance optimization if needed

**Deliverable**: Production-ready CIR calling convention lowering

**Implementation Notes**:
- Testing infrastructure setup (differential testing harness) takes significant time (~1 week)
- If infrastructure setup exceeds 1 week, may extend Phase 5 duration
- Estimated 2-3 weeks (upper end due to testing infrastructure complexity)

### Phase 6: Varargs Support (Conditional - If Required for Graduation)

**Duration**: 3-4 weeks (not currently in baseline)

**Probability Required**: **70-80%** (most C programs use `printf`/`scanf`)

**Rationale**:
- CIR incubator has many `NYI` assertions for varargs
- Real-world C code heavily uses varargs (printf, scanf, logging)
- ~40% of C code would be unusable without varargs support
- Graduation reviewers may block without varargs
- Complex state management (GP vs FP register tracking, register save area, 30+ tests per target)

**Work Required**:

1. **x86_64 System V Varargs** (1.5-2 weeks):
   - Implement `va_list` type lowering
   - Implement `va_start` (initialize va_list from register save area)
   - Implement `va_arg` (extract next argument, handle types)
   - Implement `va_end` (cleanup)
   - Handle register save area allocation (176 bytes: 6 GP * 8 + 8 FP * 16)
   - Track GP registers (RDI, RSI, RDX, RCX, R8, R9) vs FP registers (XMM0-XMM7) separately
   - Handle overflow to stack for arguments beyond 6+8 registers
   - Test with printf/scanf (30+ tests)

2. **AArch64 PCS Varargs** (1.5-2 weeks):
   - Different `va_list` structure (5 fields: gp_offset, fp_offset, overflow_arg_area, reg_save_area, etc.)
   - Stack-based varargs with register overflow area
   - Implement va_start/va_arg/va_end/va_copy
   - Handle alignment requirements (8-byte GP, 16-byte FP)
   - Register save area is stack-based (not pre-allocated)
   - Test with printf/scanf (30+ tests)

3. **Testing & Edge Cases** (3-5 days):
   - Test varargs calling conventions (60+ tests total)
   - Handle va_copy edge cases
   - Validate against classic codegen
   - Mixed GP/FP argument scenarios

**Decision Point**: **Week 1** - ask Andy if varargs is graduation blocker (don't wait for Week 2)

**Impact on Timeline**:
- **If Required**: 15 weeks → 17-19 weeks total
- **If Deferred**: Stay on 13-15 week timeline, add varargs post-graduation

**Recommendation**: **Assume varargs IS required** and budget 17-19 weeks, not 15 weeks

### Phase 7: Documentation (Week 19)

1. API documentation
2. User guide for adding new dialects
3. Target implementation guide
4. Design rationale document

**Deliverable**: Comprehensive documentation

### Phase 8: FIR Prototype (Future)

1. Work with FIR team on requirements
2. Implement ABITypeInterface for FIR types
3. Implement FIRABIRewriteContext
4. Create FIR calling convention pass
5. Validate with Fortran test cases

**Deliverable**: Proof of concept for FIR

**Note**: This phase is post-graduation and not included in the 17-19 week timeline.

## VI. Target-Specific Details

### 6.1 x86_64 System V ABI

**Reference**: [System V AMD64 ABI](https://refspecs.linuxbase.org/elf/x86_64-abi-0.99.pdf)

**Key Rules**:
- Integer arguments in registers: RDI, RSI, RDX, RCX, R8, R9
- FP arguments in XMM0-XMM7
- Return in RAX/RDX (integer) or XMM0/XMM1 (FP)
- Structs classified by 8-byte chunks
- Memory arguments passed on stack

**Classification Algorithm**:
1. Divide type into 8-byte chunks
2. Classify each chunk (Integer, SSE, X87, Memory, NoClass)
3. Merge adjacent chunks
4. Post-merge cleanup
5. Map to registers or memory

**Edge Case: `__int128` vs `_BitInt(128)`**

These types have the same size (16 bytes) but **different ABI classification**:
- `__int128`: **INTEGER** class → passed in RDI + RSI (return: RAX + RDX)
- `_BitInt(128)`: **MEMORY** class → passed indirectly via hidden pointer
- `_BitInt(64)`: **INTEGER** class → passed in single register RDI

**Why This Matters**: Same size, different calling convention. Implementation must use ABITypeInterface methods `isInt128()` and `isBitInt()` to distinguish these types correctly.

**Implementation Status**: ✅ Already implemented in CIR incubator

**Migration Effort**: Low - mainly replacing CIR type checks

### 6.2 AArch64 Procedure Call Standard

**Reference**: [ARM AArch64 ABI](https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst)

**Key Rules**:
- Integer arguments in X0-X7
- FP arguments in V0-V7
- Return in X0/X1 (integer) or V0/V1 (FP)
- Homogeneous Floating-point Aggregates (HFA) in FP registers
- Homogeneous Short-Vector Aggregates (HVA) in vector registers

**Classification**:
1. Check if type is HFA/HVA
2. If aggregate, check if fits in registers
3. Otherwise, pass indirectly

**Implementation Status**: ✅ Already implemented in CIR incubator

**Migration Effort**: Low - similar to x86_64

### 6.3 Future Targets

**Candidates** (if time permits):
- ARM32 (for embedded systems)
- RISC-V (emerging importance)
- WebAssembly (for WASM backends)
- PowerPC (for HPC systems)

**Not Priority**: MIPS, Sparc, Hexagon, etc. (less common)

## VII. Testing Strategy

### 7.1 Unit Tests

**Type Interface Tests**:
```cpp
TEST(ABITypeInterface, IntegerQueries) {
  MLIRContext ctx;
  Type intTy = cir::IntType::get(&ctx, 32, true);
  auto abiTy = dyn_cast<ABITypeInterface>(intTy);
  EXPECT_TRUE(abiTy.isInteger());
  EXPECT_FALSE(abiTy.isRecord());
}
```

**Classification Tests**:
```cpp
TEST(X86_64ABI, SimpleIntReturn) {
  // Setup
  MLIRContext ctx;
  X86_64ABIInfo abi(...);
  Type i32 = IntegerType::get(&ctx, 32);
  
  // Classify
  ABIArgInfo info = abi.classifyReturnType(i32);
  
  // Verify
  EXPECT_TRUE(info.isDirect());
  EXPECT_FALSE(info.isIndirect());
}
```

**Lowering Tests**:
```cpp
TEST(CIRCallConv, FunctionRewrite) {
  // Create function with struct argument
  // Run CallConvLowering pass
  // Verify function signature changed correctly
  // Verify call sites updated
}
```

### 7.2 Integration Tests

**ABI Compliance Tests**:
- Generate test cases using Clang classic codegen
- Lower same functions with CIR
- Compare LLVM IR output after lowering to LLVM
- Ensure calling conventions match

**Cross-Dialect Tests** (future):
- CIR function calling FIR function
- FIR function calling CIR function
- Verify ABI compatibility

### 7.3 Performance Tests

**Compilation Time**:
- Measure time to run CallConvLowering pass
- Compare with CIR incubator implementation
- Target: < 5% overhead

**Generated Code Quality**:
- Compare with classic codegen output
- Check for unnecessary copies or spills
- Verify register allocation is similar

## VIII. Migration from CIR Incubator

### 8.1 Migration Steps

1. **Parallel Implementation**:
   - Build new MLIR-agnostic infrastructure
   - Keep CIR incubator code working
   - Test new infrastructure alongside old

2. **Incremental Switchover**:
   - Replace one component at a time
   - ABIArgInfo first (easiest)
   - Then LowerFunctionInfo
   - Then target implementations
   - Finally, pass structure

3. **Validation**:
   - Run both old and new implementations
   - Compare results
   - Fix discrepancies

4. **Upstream Submission**:
   - Submit shared infrastructure to MLIR
   - Submit CIR adaptations to CIR upstream
   - Deprecate incubator implementation

### 8.2 Compatibility Considerations

**Source Compatibility**:
- New ABIArgInfo API should match old API where possible
- Minimize changes to target implementations
- Provide migration utilities if API changes

**Binary Compatibility**:
- Not a concern (no ABI for internal compiler structures)

**Test Migration**:
- Port existing CIR tests to new infrastructure
- Ensure all test cases still pass
- Add new tests for edge cases

### 8.3 Deprecation Plan

Once new implementation is stable:
1. Mark CIR incubator implementation as deprecated (Month 1)
2. Update documentation to point to new implementation (Month 1)
3. Keep old code for 1-2 releases for safety (Months 1-6)
4. Remove old implementation (Month 6+)

## IX. Future Work

### 9.1 Additional Targets

- RISC-V (emerging ISA, growing importance)
- WebAssembly (for web-based backends)
- ARM32 (for embedded systems)
- PowerPC (for HPC)

### 9.2 Advanced Features

**Varargs Support**:
- Currently marked NYI in CIR
- Need to handle variable argument lowering
- Different per target (va_list representation varies)

**Microsoft ABI**:
- Windows calling conventions
- MSVC C++ ABI
- Different from Itanium C++ ABI

**Swift Calling Convention**:
- Swift-specific argument passing
- Error handling conventions
- Async conventions

**Vector ABI**:
- SIMD type passing
- SVE (ARM Scalable Vector Extension)
- AVX-512 considerations

### 9.3 Optimization Opportunities

**Return Value Optimization (RVO)**:
- Avoid copies for returned aggregates
- Requires coordination with frontend

**Tail Call Optimization**:
- Recognize tail call patterns
- Lower to tail call convention

**Inlining-Aware Lowering**:
- Delay ABI lowering until after inlining
- Can avoid unnecessary marshalling

### 9.4 GSoC Integration

**Monitor GSoC Progress**:
- Track PR #140112 development
- Assess fit with MLIR needs
- Plan integration if beneficial

**Potential Integration**:
- Use GSoC's ABI type system
- Wrap GSoC ABIInfo implementations
- Share test cases and validation

**Timeline**:
- Short term (Q1 2026): Implement MLIR-native solution
- Medium term (Q2-Q3 2026): Evaluate GSoC library
- Long term (Q4 2026+): Potentially refactor to use GSoC

## X. Open Questions and Risks

### 10.1 Open Questions

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

### 10.2 Risks

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

## XI. Success Metrics

### 11.1 Functional Metrics

- ✅ CIR can lower x86_64 calling conventions correctly (100% test pass rate)
- ✅ CIR can lower AArch64 calling conventions correctly (100% test pass rate)
- ✅ ABI output matches classic Clang codegen (validated by comparison tests)
- ✅ All CIR incubator tests pass with new implementation

### 11.2 Quality Metrics

- ✅ Code coverage > 90% for ABI classification logic
- ✅ Zero known ABI compliance bugs
- ✅ Documentation complete (API, user guide, design rationale)

### 11.3 Performance Metrics

- ✅ CallConvLowering pass overhead < 5% compilation time
  - **Context**: This refers to **compile-time overhead**, not runtime performance
  - **Baseline**: Classic Clang ABI lowering adds ~1-2% to compile time
  - **Target**: MLIR-agnostic version should be ≤2.5× classic overhead (5% total)
  - **Measurement**: Profile on LLVM test-suite, measure time in ABI classification
  - **Optimization Strategies**: Cache ABITypeInterface queries, fast-path for primitives
- ✅ No degradation in generated code quality vs direct implementation
  - **Runtime performance unchanged**: ABI lowering is compile-time only

### 11.4 Reusability Metrics

- ✅ FIR can adopt infrastructure with < 2 weeks integration effort
- ✅ New target can be added with < 1 week effort (given ABI spec)
- ✅ ABITypeInterface requires < 10 methods implementation per dialect

## XII. References

### 12.1 ABI Specifications

- [System V AMD64 ABI](https://refspecs.linuxbase.org/elf/x86_64-abi-0.99.pdf)
- [ARM AArch64 PCS](https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst)
- [Itanium C++ ABI](https://itanium-cxx-abi.github.io/cxx-abi/abi.html)

### 12.2 LLVM/MLIR Documentation

- [MLIR Interfaces](https://mlir.llvm.org/docs/Interfaces/)
- [MLIR Type System](https://mlir.llvm.org/docs/DefiningDialects/AttributesAndTypes/)
- [MLIR Pass Infrastructure](https://mlir.llvm.org/docs/PassManagement/)

### 12.3 Related Projects

- [GSoC ABI Lowering RFC](https://discourse.llvm.org/t/rfc-an-abi-lowering-library-for-llvm/84495)
- [GSoC PR #140112](https://github.com/llvm/llvm-project/pull/140112)
- [CIR Project](https://github.com/llvm/clangir)

### 12.4 Related Implementation

- Clang CodeGen: `clang/lib/CodeGen/`
- CIR Incubator: `clang/lib/CIR/Dialect/Transforms/TargetLowering/`
- SPIR-V ABI: `mlir/lib/Dialect/SPIRV/IR/TargetAndABI.cpp`

## XIII. Appendices

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
