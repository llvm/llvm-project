//===-- aiir-c/IR.h - C API to Core AIIR IR classes ---------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares the C interface to AIIR core IR classes.
//
// Many exotic languages can interoperate with C code but have a harder time
// with C++ due to name mangling. So in addition to C, this interface enables
// tools written in such languages.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_C_IR_H
#define AIIR_C_IR_H

#include <stdbool.h>
#include <stdint.h>

#include "aiir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
/// Opaque type declarations.
///
/// Types are exposed to C bindings as structs containing opaque pointers. They
/// are not supposed to be inspected from C. This allows the underlying
/// representation to change without affecting the API users. The use of structs
/// instead of typedefs enables some type safety as structs are not implicitly
/// convertible to each other.
///
/// Instances of these types may or may not own the underlying object (most
/// often only point to an IR fragment without owning it). The ownership
/// semantics is defined by how an instance of the type was obtained.

//===----------------------------------------------------------------------===//

#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name

DEFINE_C_API_STRUCT(AiirAsmState, void);
DEFINE_C_API_STRUCT(AiirBytecodeWriterConfig, void);
DEFINE_C_API_STRUCT(AiirContext, void);
DEFINE_C_API_STRUCT(AiirDialect, void);
DEFINE_C_API_STRUCT(AiirDialectRegistry, void);
DEFINE_C_API_STRUCT(AiirOperation, void);
DEFINE_C_API_STRUCT(AiirOpOperand, void);
DEFINE_C_API_STRUCT(AiirOpPrintingFlags, void);
DEFINE_C_API_STRUCT(AiirBlock, void);
DEFINE_C_API_STRUCT(AiirRegion, void);
DEFINE_C_API_STRUCT(AiirSymbolTable, void);

DEFINE_C_API_STRUCT(AiirAttribute, const void);
DEFINE_C_API_STRUCT(AiirIdentifier, const void);
DEFINE_C_API_STRUCT(AiirLocation, const void);
DEFINE_C_API_STRUCT(AiirModule, const void);
DEFINE_C_API_STRUCT(AiirType, const void);
DEFINE_C_API_STRUCT(AiirValue, const void);

#undef DEFINE_C_API_STRUCT

/// Named AIIR attribute.
///
/// A named attribute is essentially a (name, attribute) pair where the name is
/// a string.
struct AiirNamedAttribute {
  AiirIdentifier name;
  AiirAttribute attribute;
};
typedef struct AiirNamedAttribute AiirNamedAttribute;

//===----------------------------------------------------------------------===//
// Context API.
//===----------------------------------------------------------------------===//

/// Creates an AIIR context and transfers its ownership to the caller.
/// This sets the default multithreading option (enabled).
AIIR_CAPI_EXPORTED AiirContext aiirContextCreate(void);

/// Creates an AIIR context with an explicit setting of the multithreading
/// setting and transfers its ownership to the caller.
AIIR_CAPI_EXPORTED AiirContext
aiirContextCreateWithThreading(bool threadingEnabled);

/// Creates an AIIR context, setting the multithreading setting explicitly and
/// pre-loading the dialects from the provided DialectRegistry.
AIIR_CAPI_EXPORTED AiirContext aiirContextCreateWithRegistry(
    AiirDialectRegistry registry, bool threadingEnabled);

/// Checks if two contexts are equal.
AIIR_CAPI_EXPORTED bool aiirContextEqual(AiirContext ctx1, AiirContext ctx2);

/// Checks whether a context is null.
static inline bool aiirContextIsNull(AiirContext context) {
  return !context.ptr;
}

/// Takes an AIIR context owned by the caller and destroys it.
AIIR_CAPI_EXPORTED void aiirContextDestroy(AiirContext context);

/// Sets whether unregistered dialects are allowed in this context.
AIIR_CAPI_EXPORTED void
aiirContextSetAllowUnregisteredDialects(AiirContext context, bool allow);

/// Returns whether the context allows unregistered dialects.
AIIR_CAPI_EXPORTED bool
aiirContextGetAllowUnregisteredDialects(AiirContext context);

/// Returns the number of dialects registered with the given context. A
/// registered dialect will be loaded if needed by the parser.
AIIR_CAPI_EXPORTED intptr_t
aiirContextGetNumRegisteredDialects(AiirContext context);

/// Append the contents of the given dialect registry to the registry associated
/// with the context.
AIIR_CAPI_EXPORTED void
aiirContextAppendDialectRegistry(AiirContext ctx, AiirDialectRegistry registry);

/// Returns the number of dialects loaded by the context.

AIIR_CAPI_EXPORTED intptr_t
aiirContextGetNumLoadedDialects(AiirContext context);

/// Gets the dialect instance owned by the given context using the dialect
/// namespace to identify it, loads (i.e., constructs the instance of) the
/// dialect if necessary. If the dialect is not registered with the context,
/// returns null. Use aiirContextLoad<Name>Dialect to load an unregistered
/// dialect.
AIIR_CAPI_EXPORTED AiirDialect aiirContextGetOrLoadDialect(AiirContext context,
                                                           AiirStringRef name);

/// Set threading mode (must be set to false to aiir-print-ir-after-all).
AIIR_CAPI_EXPORTED void aiirContextEnableMultithreading(AiirContext context,
                                                        bool enable);

/// Eagerly loads all available dialects registered with a context, making
/// them available for use for IR construction.
AIIR_CAPI_EXPORTED void
aiirContextLoadAllAvailableDialects(AiirContext context);

/// Returns whether the given fully-qualified operation (i.e.
/// 'dialect.operation') is registered with the context. This will return true
/// if the dialect is loaded and the operation is registered within the
/// dialect.
AIIR_CAPI_EXPORTED bool aiirContextIsRegisteredOperation(AiirContext context,
                                                         AiirStringRef name);

/// Sets the thread pool of the context explicitly, enabling multithreading in
/// the process. This API should be used to avoid re-creating thread pools in
/// long-running applications that perform multiple compilations, see
/// the C++ documentation for AIIRContext for details.
AIIR_CAPI_EXPORTED void aiirContextSetThreadPool(AiirContext context,
                                                 AiirLlvmThreadPool threadPool);

/// Gets the number of threads of the thread pool of the context when
/// multithreading is enabled. Returns 1 if no multithreading.
AIIR_CAPI_EXPORTED unsigned aiirContextGetNumThreads(AiirContext context);

/// Gets the thread pool of the context when enabled multithreading, otherwise
/// an assertion is raised.
AIIR_CAPI_EXPORTED AiirLlvmThreadPool
aiirContextGetThreadPool(AiirContext context);

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//

/// Returns the context that owns the dialect.
AIIR_CAPI_EXPORTED AiirContext aiirDialectGetContext(AiirDialect dialect);

/// Checks if the dialect is null.
static inline bool aiirDialectIsNull(AiirDialect dialect) {
  return !dialect.ptr;
}

/// Checks if two dialects that belong to the same context are equal. Dialects
/// from different contexts will not compare equal.
AIIR_CAPI_EXPORTED bool aiirDialectEqual(AiirDialect dialect1,
                                         AiirDialect dialect2);

/// Returns the namespace of the given dialect.
AIIR_CAPI_EXPORTED AiirStringRef aiirDialectGetNamespace(AiirDialect dialect);

//===----------------------------------------------------------------------===//
// DialectHandle API.
// Registration entry-points for each dialect are declared using the common
// AIIR_DECLARE_DIALECT_REGISTRATION_CAPI macro, which takes the dialect
// API name (i.e. "Func", "Tensor", "Linalg") and namespace (i.e. "func",
// "tensor", "linalg"). The following declarations are produced:
//
//   /// Gets the above hook methods in struct form for a dialect by namespace.
//   /// This is intended to facilitate dynamic lookup and registration of
//   /// dialects via a plugin facility based on shared library symbol lookup.
//   const AiirDialectHandle *aiirGetDialectHandle__{NAMESPACE}__();
//
// This is done via a common macro to facilitate future expansion to
// registration schemes.
//===----------------------------------------------------------------------===//

struct AiirDialectHandle {
  const void *ptr;
};
typedef struct AiirDialectHandle AiirDialectHandle;

#define AIIR_DECLARE_CAPI_DIALECT_REGISTRATION(Name, Namespace)                \
  AIIR_CAPI_EXPORTED AiirDialectHandle aiirGetDialectHandle__##Namespace##__(  \
      void)

/// Returns the namespace associated with the provided dialect handle.
AIIR_CAPI_EXPORTED
AiirStringRef aiirDialectHandleGetNamespace(AiirDialectHandle);

/// Inserts the dialect associated with the provided dialect handle into the
/// provided dialect registry
AIIR_CAPI_EXPORTED void aiirDialectHandleInsertDialect(AiirDialectHandle,
                                                       AiirDialectRegistry);

/// Registers the dialect associated with the provided dialect handle.
AIIR_CAPI_EXPORTED void aiirDialectHandleRegisterDialect(AiirDialectHandle,
                                                         AiirContext);

/// Loads the dialect associated with the provided dialect handle.
AIIR_CAPI_EXPORTED AiirDialect aiirDialectHandleLoadDialect(AiirDialectHandle,
                                                            AiirContext);

//===----------------------------------------------------------------------===//
// DialectRegistry API.
//===----------------------------------------------------------------------===//

/// Creates a dialect registry and transfers its ownership to the caller.
AIIR_CAPI_EXPORTED AiirDialectRegistry aiirDialectRegistryCreate(void);

/// Checks if the dialect registry is null.
static inline bool aiirDialectRegistryIsNull(AiirDialectRegistry registry) {
  return !registry.ptr;
}

/// Takes a dialect registry owned by the caller and destroys it.
AIIR_CAPI_EXPORTED void
aiirDialectRegistryDestroy(AiirDialectRegistry registry);

//===----------------------------------------------------------------------===//
// Location API.
//===----------------------------------------------------------------------===//

/// Returns the underlying location attribute of this location.
AIIR_CAPI_EXPORTED AiirAttribute
aiirLocationGetAttribute(AiirLocation location);

/// Creates a location from a location attribute.
AIIR_CAPI_EXPORTED AiirLocation
aiirLocationFromAttribute(AiirAttribute attribute);

/// Creates an File/Line/Column location owned by the given context.
AIIR_CAPI_EXPORTED AiirLocation aiirLocationFileLineColGet(
    AiirContext context, AiirStringRef filename, unsigned line, unsigned col);

/// Creates an File/Line/Column range location owned by the given context.
AIIR_CAPI_EXPORTED AiirLocation aiirLocationFileLineColRangeGet(
    AiirContext context, AiirStringRef filename, unsigned start_line,
    unsigned start_col, unsigned end_line, unsigned end_col);

/// Getter for filename of FileLineColRange.
AIIR_CAPI_EXPORTED AiirIdentifier
aiirLocationFileLineColRangeGetFilename(AiirLocation location);

/// Getter for start_line of FileLineColRange.
AIIR_CAPI_EXPORTED int
aiirLocationFileLineColRangeGetStartLine(AiirLocation location);

/// Getter for start_column of FileLineColRange.
AIIR_CAPI_EXPORTED int
aiirLocationFileLineColRangeGetStartColumn(AiirLocation location);

/// Getter for end_line of FileLineColRange.
AIIR_CAPI_EXPORTED int
aiirLocationFileLineColRangeGetEndLine(AiirLocation location);

/// Getter for end_column of FileLineColRange.
AIIR_CAPI_EXPORTED int
aiirLocationFileLineColRangeGetEndColumn(AiirLocation location);

/// TypeID Getter for FileLineColRange.
AIIR_CAPI_EXPORTED AiirTypeID aiirLocationFileLineColRangeGetTypeID(void);

/// Checks whether the given location is an FileLineColRange.
AIIR_CAPI_EXPORTED bool aiirLocationIsAFileLineColRange(AiirLocation location);

/// Creates a call site location with a callee and a caller.
AIIR_CAPI_EXPORTED AiirLocation aiirLocationCallSiteGet(AiirLocation callee,
                                                        AiirLocation caller);

/// Getter for callee of CallSite.
AIIR_CAPI_EXPORTED AiirLocation
aiirLocationCallSiteGetCallee(AiirLocation location);

/// Getter for caller of CallSite.
AIIR_CAPI_EXPORTED AiirLocation
aiirLocationCallSiteGetCaller(AiirLocation location);

/// TypeID Getter for CallSite.
AIIR_CAPI_EXPORTED AiirTypeID aiirLocationCallSiteGetTypeID(void);

/// Checks whether the given location is an CallSite.
AIIR_CAPI_EXPORTED bool aiirLocationIsACallSite(AiirLocation location);

/// Creates a fused location with an array of locations and metadata.
AIIR_CAPI_EXPORTED AiirLocation
aiirLocationFusedGet(AiirContext ctx, intptr_t nLocations,
                     AiirLocation const *locations, AiirAttribute metadata);

/// Getter for number of locations fused together.
AIIR_CAPI_EXPORTED unsigned
aiirLocationFusedGetNumLocations(AiirLocation location);

/// Getter for locations of Fused. Requires pre-allocated memory of
/// #fusedLocations X sizeof(AiirLocation).
AIIR_CAPI_EXPORTED void
aiirLocationFusedGetLocations(AiirLocation location,
                              AiirLocation *locationsCPtr);

/// Getter for metadata of Fused.
AIIR_CAPI_EXPORTED AiirAttribute
aiirLocationFusedGetMetadata(AiirLocation location);

/// TypeID Getter for Fused.
AIIR_CAPI_EXPORTED AiirTypeID aiirLocationFusedGetTypeID(void);

/// Checks whether the given location is an Fused.
AIIR_CAPI_EXPORTED bool aiirLocationIsAFused(AiirLocation location);

/// Creates a name location owned by the given context. Providing null location
/// for childLoc is allowed and if childLoc is null location, then the behavior
/// is the same as having unknown child location.
AIIR_CAPI_EXPORTED AiirLocation aiirLocationNameGet(AiirContext context,
                                                    AiirStringRef name,
                                                    AiirLocation childLoc);

/// Getter for name of Name.
AIIR_CAPI_EXPORTED AiirIdentifier
aiirLocationNameGetName(AiirLocation location);

/// Getter for childLoc of Name.
AIIR_CAPI_EXPORTED AiirLocation
aiirLocationNameGetChildLoc(AiirLocation location);

/// TypeID Getter for Name.
AIIR_CAPI_EXPORTED AiirTypeID aiirLocationNameGetTypeID(void);

/// Checks whether the given location is an Name.
AIIR_CAPI_EXPORTED bool aiirLocationIsAName(AiirLocation location);

/// Creates a location with unknown position owned by the given context.
AIIR_CAPI_EXPORTED AiirLocation aiirLocationUnknownGet(AiirContext context);

/// Gets the context that a location was created with.
AIIR_CAPI_EXPORTED AiirContext aiirLocationGetContext(AiirLocation location);

/// Checks if the location is null.
static inline bool aiirLocationIsNull(AiirLocation location) {
  return !location.ptr;
}

/// Checks if two locations are equal.
AIIR_CAPI_EXPORTED bool aiirLocationEqual(AiirLocation l1, AiirLocation l2);

/// Prints a location by sending chunks of the string representation and
/// forwarding `userData to `callback`. Note that the callback may be called
/// several times with consecutive chunks of the string.
AIIR_CAPI_EXPORTED void aiirLocationPrint(AiirLocation location,
                                          AiirStringCallback callback,
                                          void *userData);

//===----------------------------------------------------------------------===//
// Module API.
//===----------------------------------------------------------------------===//

/// Creates a new, empty module and transfers ownership to the caller.
AIIR_CAPI_EXPORTED AiirModule aiirModuleCreateEmpty(AiirLocation location);

/// Parses a module from the string and transfers ownership to the caller.
AIIR_CAPI_EXPORTED AiirModule aiirModuleCreateParse(AiirContext context,
                                                    AiirStringRef module);

/// Parses a module from file and transfers ownership to the caller.
AIIR_CAPI_EXPORTED AiirModule
aiirModuleCreateParseFromFile(AiirContext context, AiirStringRef fileName);

/// Gets the context that a module was created with.
AIIR_CAPI_EXPORTED AiirContext aiirModuleGetContext(AiirModule module);

/// Gets the body of the module, i.e. the only block it contains.
AIIR_CAPI_EXPORTED AiirBlock aiirModuleGetBody(AiirModule module);

/// Checks whether a module is null.
static inline bool aiirModuleIsNull(AiirModule module) { return !module.ptr; }

/// Takes a module owned by the caller and deletes it.
AIIR_CAPI_EXPORTED void aiirModuleDestroy(AiirModule module);

/// Views the module as a generic operation.
AIIR_CAPI_EXPORTED AiirOperation aiirModuleGetOperation(AiirModule module);

/// Views the generic operation as a module.
/// The returned module is null when the input operation was not a ModuleOp.
AIIR_CAPI_EXPORTED AiirModule aiirModuleFromOperation(AiirOperation op);

/// Checks if two modules are equal.
AIIR_CAPI_EXPORTED bool aiirModuleEqual(AiirModule lhs, AiirModule rhs);

/// Compute a hash for the given module.
AIIR_CAPI_EXPORTED size_t aiirModuleHashValue(AiirModule mod);

//===----------------------------------------------------------------------===//
// Operation state.
//===----------------------------------------------------------------------===//

/// An auxiliary class for constructing operations.
///
/// This class contains all the information necessary to construct the
/// operation. It owns the AiirRegions it has pointers to and does not own
/// anything else. By default, the state can be constructed from a name and
/// location, the latter being also used to access the context, and has no other
/// components. These components can be added progressively until the operation
/// is constructed. Users are not expected to rely on the internals of this
/// class and should use aiirOperationState* functions instead.

struct AiirOperationState {
  AiirStringRef name;
  AiirLocation location;
  intptr_t nResults;
  AiirType *results;
  intptr_t nOperands;
  AiirValue *operands;
  intptr_t nRegions;
  AiirRegion *regions;
  intptr_t nSuccessors;
  AiirBlock *successors;
  intptr_t nAttributes;
  AiirNamedAttribute *attributes;
  bool enableResultTypeInference;
};
typedef struct AiirOperationState AiirOperationState;

/// Constructs an operation state from a name and a location.
AIIR_CAPI_EXPORTED AiirOperationState aiirOperationStateGet(AiirStringRef name,
                                                            AiirLocation loc);

/// Adds a list of components to the operation state.
AIIR_CAPI_EXPORTED void aiirOperationStateAddResults(AiirOperationState *state,
                                                     intptr_t n,
                                                     AiirType const *results);
AIIR_CAPI_EXPORTED void
aiirOperationStateAddOperands(AiirOperationState *state, intptr_t n,
                              AiirValue const *operands);
AIIR_CAPI_EXPORTED void
aiirOperationStateAddOwnedRegions(AiirOperationState *state, intptr_t n,
                                  AiirRegion const *regions);
AIIR_CAPI_EXPORTED void
aiirOperationStateAddSuccessors(AiirOperationState *state, intptr_t n,
                                AiirBlock const *successors);
AIIR_CAPI_EXPORTED void
aiirOperationStateAddAttributes(AiirOperationState *state, intptr_t n,
                                AiirNamedAttribute const *attributes);

/// Enables result type inference for the operation under construction. If
/// enabled, then the caller must not have called
/// aiirOperationStateAddResults(). Note that if enabled, the
/// aiirOperationCreate() call is failable: it will return a null operation
/// on inference failure and will emit diagnostics.
AIIR_CAPI_EXPORTED void
aiirOperationStateEnableResultTypeInference(AiirOperationState *state);

//===----------------------------------------------------------------------===//
// AsmState API.
// While many of these are simple settings that could be represented in a
// struct, they are wrapped in a heap allocated object and accessed via
// functions to maximize the possibility of compatibility over time.
//===----------------------------------------------------------------------===//

/// Creates new AsmState, as with AsmState the IR should not be mutated
/// in-between using this state.
/// Must be freed with a call to aiirAsmStateDestroy().
// TODO: This should be expanded to handle location & resouce map.
AIIR_CAPI_EXPORTED AiirAsmState
aiirAsmStateCreateForOperation(AiirOperation op, AiirOpPrintingFlags flags);

/// Creates new AsmState from value.
/// Must be freed with a call to aiirAsmStateDestroy().
// TODO: This should be expanded to handle location & resouce map.
AIIR_CAPI_EXPORTED AiirAsmState
aiirAsmStateCreateForValue(AiirValue value, AiirOpPrintingFlags flags);

/// Destroys printing flags created with aiirAsmStateCreate.
AIIR_CAPI_EXPORTED void aiirAsmStateDestroy(AiirAsmState state);

//===----------------------------------------------------------------------===//
// Op Printing flags API.
// While many of these are simple settings that could be represented in a
// struct, they are wrapped in a heap allocated object and accessed via
// functions to maximize the possibility of compatibility over time.
//===----------------------------------------------------------------------===//

/// Creates new printing flags with defaults, intended for customization.
/// Must be freed with a call to aiirOpPrintingFlagsDestroy().
AIIR_CAPI_EXPORTED AiirOpPrintingFlags aiirOpPrintingFlagsCreate(void);

/// Destroys printing flags created with aiirOpPrintingFlagsCreate.
AIIR_CAPI_EXPORTED void aiirOpPrintingFlagsDestroy(AiirOpPrintingFlags flags);

/// Enables the elision of large elements attributes by printing a lexically
/// valid but otherwise meaningless form instead of the element data. The
/// `largeElementLimit` is used to configure what is considered to be a "large"
/// ElementsAttr by providing an upper limit to the number of elements.
AIIR_CAPI_EXPORTED void
aiirOpPrintingFlagsElideLargeElementsAttrs(AiirOpPrintingFlags flags,
                                           intptr_t largeElementLimit);

/// Enables the elision of large resources strings by omitting them from the
/// `dialect_resources` section. The `largeResourceLimit` is used to configure
/// what is considered to be a "large" resource by providing an upper limit to
/// the string size.
AIIR_CAPI_EXPORTED void
aiirOpPrintingFlagsElideLargeResourceString(AiirOpPrintingFlags flags,
                                            intptr_t largeResourceLimit);

/// Enable or disable printing of debug information (based on `enable`). If
/// 'prettyForm' is set to true, debug information is printed in a more readable
/// 'pretty' form. Note: The IR generated with 'prettyForm' is not parsable.
AIIR_CAPI_EXPORTED void
aiirOpPrintingFlagsEnableDebugInfo(AiirOpPrintingFlags flags, bool enable,
                                   bool prettyForm);

/// Always print operations in the generic form.
AIIR_CAPI_EXPORTED void
aiirOpPrintingFlagsPrintGenericOpForm(AiirOpPrintingFlags flags);

/// Print the name and location, if NamedLoc, as a prefix to the SSA ID.
AIIR_CAPI_EXPORTED void
aiirOpPrintingFlagsPrintNameLocAsPrefix(AiirOpPrintingFlags flags);

/// Use local scope when printing the operation. This allows for using the
/// printer in a more localized and thread-safe setting, but may not
/// necessarily be identical to what the IR will look like when dumping
/// the full module.
AIIR_CAPI_EXPORTED void
aiirOpPrintingFlagsUseLocalScope(AiirOpPrintingFlags flags);

/// Do not verify the operation when using custom operation printers.
AIIR_CAPI_EXPORTED void
aiirOpPrintingFlagsAssumeVerified(AiirOpPrintingFlags flags);

/// Skip printing regions.
AIIR_CAPI_EXPORTED void
aiirOpPrintingFlagsSkipRegions(AiirOpPrintingFlags flags);

//===----------------------------------------------------------------------===//
// Bytecode printing flags API.
//===----------------------------------------------------------------------===//

/// Creates new printing flags with defaults, intended for customization.
/// Must be freed with a call to aiirBytecodeWriterConfigDestroy().
AIIR_CAPI_EXPORTED AiirBytecodeWriterConfig
aiirBytecodeWriterConfigCreate(void);

/// Destroys printing flags created with aiirBytecodeWriterConfigCreate.
AIIR_CAPI_EXPORTED void
aiirBytecodeWriterConfigDestroy(AiirBytecodeWriterConfig config);

/// Sets the version to emit in the writer config.
AIIR_CAPI_EXPORTED void
aiirBytecodeWriterConfigDesiredEmitVersion(AiirBytecodeWriterConfig flags,
                                           int64_t version);

//===----------------------------------------------------------------------===//
// Operation API.
//===----------------------------------------------------------------------===//

/// Creates an operation and transfers ownership to the caller.
/// Note that caller owned child objects are transferred in this call and must
/// not be further used. Particularly, this applies to any regions added to
/// the state (the implementation may invalidate any such pointers).
///
/// This call can fail under the following conditions, in which case, it will
/// return a null operation and emit diagnostics:
///   - Result type inference is enabled and cannot be performed.
AIIR_CAPI_EXPORTED AiirOperation aiirOperationCreate(AiirOperationState *state);

/// Parses an operation, giving ownership to the caller. If parsing fails a null
/// operation will be returned, and an error diagnostic emitted.
///
/// `sourceStr` may be either the text assembly format, or binary bytecode
/// format. `sourceName` is used as the file name of the source; any IR without
/// locations will get a `FileLineColLoc` location with `sourceName` as the file
/// name.
AIIR_CAPI_EXPORTED AiirOperation aiirOperationCreateParse(
    AiirContext context, AiirStringRef sourceStr, AiirStringRef sourceName);

/// Creates a deep copy of an operation. The operation is not inserted and
/// ownership is transferred to the caller.
AIIR_CAPI_EXPORTED AiirOperation aiirOperationClone(AiirOperation op);

/// Takes an operation owned by the caller and destroys it.
AIIR_CAPI_EXPORTED void aiirOperationDestroy(AiirOperation op);

/// Removes the given operation from its parent block. The operation is not
/// destroyed. The ownership of the operation is transferred to the caller.
AIIR_CAPI_EXPORTED void aiirOperationRemoveFromParent(AiirOperation op);

/// Checks whether the underlying operation is null.
static inline bool aiirOperationIsNull(AiirOperation op) { return !op.ptr; }

/// Checks whether two operation handles point to the same operation. This does
/// not perform deep comparison.
AIIR_CAPI_EXPORTED bool aiirOperationEqual(AiirOperation op,
                                           AiirOperation other);

/// Compute a hash for the given operation.
AIIR_CAPI_EXPORTED size_t aiirOperationHashValue(AiirOperation op);

/// Gets the context this operation is associated with
AIIR_CAPI_EXPORTED AiirContext aiirOperationGetContext(AiirOperation op);

/// Checks if the operation name has a trait identified by the given type id.
AIIR_CAPI_EXPORTED bool aiirOperationNameHasTrait(AiirStringRef opName,
                                                  AiirTypeID traitTypeID,
                                                  AiirContext context);

/// Gets the location of the operation.
AIIR_CAPI_EXPORTED AiirLocation aiirOperationGetLocation(AiirOperation op);

/// Sets the location of the operation.
AIIR_CAPI_EXPORTED void aiirOperationSetLocation(AiirOperation op,
                                                 AiirLocation loc);

/// Gets the type id of the operation.
/// Returns null if the operation does not have a registered operation
/// description.
AIIR_CAPI_EXPORTED AiirTypeID aiirOperationGetTypeID(AiirOperation op);

/// Gets the name of the operation as an identifier.
AIIR_CAPI_EXPORTED AiirIdentifier aiirOperationGetName(AiirOperation op);

/// Gets the block that owns this operation, returning null if the operation is
/// not owned.
AIIR_CAPI_EXPORTED AiirBlock aiirOperationGetBlock(AiirOperation op);

/// Gets the operation that owns this operation, returning null if the operation
/// is not owned.
AIIR_CAPI_EXPORTED AiirOperation
aiirOperationGetParentOperation(AiirOperation op);

/// Returns the number of regions attached to the given operation.
AIIR_CAPI_EXPORTED intptr_t aiirOperationGetNumRegions(AiirOperation op);

/// Returns `pos`-th region attached to the operation.
AIIR_CAPI_EXPORTED AiirRegion aiirOperationGetRegion(AiirOperation op,
                                                     intptr_t pos);

/// Returns an operation immediately following the given operation it its
/// enclosing block.
AIIR_CAPI_EXPORTED AiirOperation aiirOperationGetNextInBlock(AiirOperation op);

/// Returns the number of operands of the operation.
AIIR_CAPI_EXPORTED intptr_t aiirOperationGetNumOperands(AiirOperation op);

/// Returns `pos`-th operand of the operation.
AIIR_CAPI_EXPORTED AiirValue aiirOperationGetOperand(AiirOperation op,
                                                     intptr_t pos);

/// Returns `pos`-th OpOperand of the operation.
AIIR_CAPI_EXPORTED AiirOpOperand aiirOperationGetOpOperand(AiirOperation op,
                                                           intptr_t pos);

/// Sets the `pos`-th operand of the operation.
AIIR_CAPI_EXPORTED void aiirOperationSetOperand(AiirOperation op, intptr_t pos,
                                                AiirValue newValue);

/// Replaces the operands of the operation.
AIIR_CAPI_EXPORTED void aiirOperationSetOperands(AiirOperation op,
                                                 intptr_t nOperands,
                                                 AiirValue const *operands);

/// Returns the number of results of the operation.
AIIR_CAPI_EXPORTED intptr_t aiirOperationGetNumResults(AiirOperation op);

/// Returns `pos`-th result of the operation.
AIIR_CAPI_EXPORTED AiirValue aiirOperationGetResult(AiirOperation op,
                                                    intptr_t pos);

/// Returns the number of successor blocks of the operation.
AIIR_CAPI_EXPORTED intptr_t aiirOperationGetNumSuccessors(AiirOperation op);

/// Returns `pos`-th successor of the operation.
AIIR_CAPI_EXPORTED AiirBlock aiirOperationGetSuccessor(AiirOperation op,
                                                       intptr_t pos);

/// Set `pos`-th successor of the operation.
AIIR_CAPI_EXPORTED void
aiirOperationSetSuccessor(AiirOperation op, intptr_t pos, AiirBlock block);

/// Returns true if this operation defines an inherent attribute with this name.
/// Note: the attribute can be optional, so
/// `aiirOperationGetInherentAttributeByName` can still return a null attribute.
AIIR_CAPI_EXPORTED bool
aiirOperationHasInherentAttributeByName(AiirOperation op, AiirStringRef name);

/// Returns an inherent attribute attached to the operation given its name.
AIIR_CAPI_EXPORTED AiirAttribute
aiirOperationGetInherentAttributeByName(AiirOperation op, AiirStringRef name);

/// Sets an inherent attribute by name, replacing the existing if it exists.
/// This has no effect if "name" does not match an inherent attribute.
AIIR_CAPI_EXPORTED void
aiirOperationSetInherentAttributeByName(AiirOperation op, AiirStringRef name,
                                        AiirAttribute attr);

/// Returns the number of discardable attributes attached to the operation.
AIIR_CAPI_EXPORTED intptr_t
aiirOperationGetNumDiscardableAttributes(AiirOperation op);

/// Return `pos`-th discardable attribute of the operation.
AIIR_CAPI_EXPORTED AiirNamedAttribute
aiirOperationGetDiscardableAttribute(AiirOperation op, intptr_t pos);

/// Returns a discardable attribute attached to the operation given its name.
AIIR_CAPI_EXPORTED AiirAttribute aiirOperationGetDiscardableAttributeByName(
    AiirOperation op, AiirStringRef name);

/// Sets a discardable attribute by name, replacing the existing if it exists or
/// adding a new one otherwise. The new `attr` Attribute is not allowed to be
/// null, use `aiirOperationRemoveDiscardableAttributeByName` to remove an
/// Attribute instead.
AIIR_CAPI_EXPORTED void
aiirOperationSetDiscardableAttributeByName(AiirOperation op, AiirStringRef name,
                                           AiirAttribute attr);

/// Removes a discardable attribute by name. Returns false if the attribute was
/// not found and true if removed.
AIIR_CAPI_EXPORTED bool
aiirOperationRemoveDiscardableAttributeByName(AiirOperation op,
                                              AiirStringRef name);

/// Returns the number of attributes attached to the operation.
/// Deprecated, please use `aiirOperationGetNumInherentAttributes` or
/// `aiirOperationGetNumDiscardableAttributes`.
AIIR_CAPI_EXPORTED intptr_t aiirOperationGetNumAttributes(AiirOperation op);

/// Return `pos`-th attribute of the operation.
/// Deprecated, please use `aiirOperationGetInherentAttribute` or
/// `aiirOperationGetDiscardableAttribute`.
AIIR_CAPI_EXPORTED AiirNamedAttribute
aiirOperationGetAttribute(AiirOperation op, intptr_t pos);

/// Returns an attribute attached to the operation given its name.
/// Deprecated, please use `aiirOperationGetInherentAttributeByName` or
/// `aiirOperationGetDiscardableAttributeByName`.
AIIR_CAPI_EXPORTED AiirAttribute
aiirOperationGetAttributeByName(AiirOperation op, AiirStringRef name);

/// Sets an attribute by name, replacing the existing if it exists or
/// adding a new one otherwise.
/// Deprecated, please use `aiirOperationSetInherentAttributeByName` or
/// `aiirOperationSetDiscardableAttributeByName`.
AIIR_CAPI_EXPORTED void aiirOperationSetAttributeByName(AiirOperation op,
                                                        AiirStringRef name,
                                                        AiirAttribute attr);

/// Removes an attribute by name. Returns false if the attribute was not found
/// and true if removed.
/// Deprecated, please use `aiirOperationRemoveInherentAttributeByName` or
/// `aiirOperationRemoveDiscardableAttributeByName`.
AIIR_CAPI_EXPORTED bool aiirOperationRemoveAttributeByName(AiirOperation op,
                                                           AiirStringRef name);

/// Prints an operation by sending chunks of the string representation and
/// forwarding `userData to `callback`. Note that the callback may be called
/// several times with consecutive chunks of the string.
AIIR_CAPI_EXPORTED void aiirOperationPrint(AiirOperation op,
                                           AiirStringCallback callback,
                                           void *userData);

/// Same as aiirOperationPrint but accepts flags controlling the printing
/// behavior.
AIIR_CAPI_EXPORTED void aiirOperationPrintWithFlags(AiirOperation op,
                                                    AiirOpPrintingFlags flags,
                                                    AiirStringCallback callback,
                                                    void *userData);

/// Same as aiirOperationPrint but accepts AsmState controlling the printing
/// behavior as well as caching computed names.
AIIR_CAPI_EXPORTED void aiirOperationPrintWithState(AiirOperation op,
                                                    AiirAsmState state,
                                                    AiirStringCallback callback,
                                                    void *userData);

/// Same as aiirOperationPrint but writing the bytecode format.
AIIR_CAPI_EXPORTED void aiirOperationWriteBytecode(AiirOperation op,
                                                   AiirStringCallback callback,
                                                   void *userData);

/// Same as aiirOperationWriteBytecode but with writer config and returns
/// failure only if desired bytecode could not be honored.
AIIR_CAPI_EXPORTED AiirLogicalResult aiirOperationWriteBytecodeWithConfig(
    AiirOperation op, AiirBytecodeWriterConfig config,
    AiirStringCallback callback, void *userData);

/// Prints an operation to stderr.
AIIR_CAPI_EXPORTED void aiirOperationDump(AiirOperation op);

/// Verify the operation and return true if it passes, false if it fails.
AIIR_CAPI_EXPORTED bool aiirOperationVerify(AiirOperation op);

/// Moves the given operation immediately after the other operation in its
/// parent block. The given operation may be owned by the caller or by its
/// current block. The other operation must belong to a block. In any case, the
/// ownership is transferred to the block of the other operation.
AIIR_CAPI_EXPORTED void aiirOperationMoveAfter(AiirOperation op,
                                               AiirOperation other);

/// Moves the given operation immediately before the other operation in its
/// parent block. The given operation may be owner by the caller or by its
/// current block. The other operation must belong to a block. In any case, the
/// ownership is transferred to the block of the other operation.
AIIR_CAPI_EXPORTED void aiirOperationMoveBefore(AiirOperation op,
                                                AiirOperation other);

/// Given an operation 'other' that is within the same parent block, return
/// whether the current operation is before 'other' in the operation list
/// of the parent block.
/// Note: This function has an average complexity of O(1), but worst case may
/// take O(N) where N is the number of operations within the parent block.
AIIR_CAPI_EXPORTED bool aiirOperationIsBeforeInBlock(AiirOperation op,
                                                     AiirOperation other);
/// Operation walk result.
typedef enum AiirWalkResult {
  AiirWalkResultAdvance,
  AiirWalkResultInterrupt,
  AiirWalkResultSkip
} AiirWalkResult;

/// Traversal order for operation walk.
typedef enum AiirWalkOrder {
  AiirWalkPreOrder,
  AiirWalkPostOrder
} AiirWalkOrder;

/// Operation walker type. The handler is passed an (opaque) reference to an
/// operation and a pointer to a `userData`.
typedef AiirWalkResult (*AiirOperationWalkCallback)(AiirOperation,
                                                    void *userData);

/// Walks operation `op` in `walkOrder` and calls `callback` on that operation.
/// `*userData` is passed to the callback as well and can be used to tunnel some
/// context or other data into the callback.
AIIR_CAPI_EXPORTED
void aiirOperationWalk(AiirOperation op, AiirOperationWalkCallback callback,
                       void *userData, AiirWalkOrder walkOrder);

/// Replace uses of 'of' value with the 'with' value inside the 'op' operation.
AIIR_CAPI_EXPORTED void
aiirOperationReplaceUsesOfWith(AiirOperation op, AiirValue of, AiirValue with);

//===----------------------------------------------------------------------===//
// Region API.
//===----------------------------------------------------------------------===//

/// Creates a new empty region and transfers ownership to the caller.
AIIR_CAPI_EXPORTED AiirRegion aiirRegionCreate(void);

/// Takes a region owned by the caller and destroys it.
AIIR_CAPI_EXPORTED void aiirRegionDestroy(AiirRegion region);

/// Checks whether a region is null.
static inline bool aiirRegionIsNull(AiirRegion region) { return !region.ptr; }

/// Checks whether two region handles point to the same region. This does not
/// perform deep comparison.
AIIR_CAPI_EXPORTED bool aiirRegionEqual(AiirRegion region, AiirRegion other);

/// Gets the first block in the region.
AIIR_CAPI_EXPORTED AiirBlock aiirRegionGetFirstBlock(AiirRegion region);

/// Takes a block owned by the caller and appends it to the given region.
AIIR_CAPI_EXPORTED void aiirRegionAppendOwnedBlock(AiirRegion region,
                                                   AiirBlock block);

/// Takes a block owned by the caller and inserts it at `pos` to the given
/// region. This is an expensive operation that linearly scans the region,
/// prefer insertAfter/Before instead.
AIIR_CAPI_EXPORTED void
aiirRegionInsertOwnedBlock(AiirRegion region, intptr_t pos, AiirBlock block);

/// Takes a block owned by the caller and inserts it after the (non-owned)
/// reference block in the given region. The reference block must belong to the
/// region. If the reference block is null, prepends the block to the region.
AIIR_CAPI_EXPORTED void aiirRegionInsertOwnedBlockAfter(AiirRegion region,
                                                        AiirBlock reference,
                                                        AiirBlock block);

/// Takes a block owned by the caller and inserts it before the (non-owned)
/// reference block in the given region. The reference block must belong to the
/// region. If the reference block is null, appends the block to the region.
AIIR_CAPI_EXPORTED void aiirRegionInsertOwnedBlockBefore(AiirRegion region,
                                                         AiirBlock reference,
                                                         AiirBlock block);

/// Returns first region attached to the operation.
AIIR_CAPI_EXPORTED AiirRegion aiirOperationGetFirstRegion(AiirOperation op);

/// Returns the region immediately following the given region in its parent
/// operation.
AIIR_CAPI_EXPORTED AiirRegion aiirRegionGetNextInOperation(AiirRegion region);

/// Moves the entire content of the source region to the target region.
AIIR_CAPI_EXPORTED void aiirRegionTakeBody(AiirRegion target,
                                           AiirRegion source);

//===----------------------------------------------------------------------===//
// Block API.
//===----------------------------------------------------------------------===//

/// Creates a new empty block with the given argument types and transfers
/// ownership to the caller.
AIIR_CAPI_EXPORTED AiirBlock aiirBlockCreate(intptr_t nArgs,
                                             AiirType const *args,
                                             AiirLocation const *locs);

/// Takes a block owned by the caller and destroys it.
AIIR_CAPI_EXPORTED void aiirBlockDestroy(AiirBlock block);

/// Detach a block from the owning region and assume ownership.
AIIR_CAPI_EXPORTED void aiirBlockDetach(AiirBlock block);

/// Checks whether a block is null.
static inline bool aiirBlockIsNull(AiirBlock block) { return !block.ptr; }

/// Checks whether two blocks handles point to the same block. This does not
/// perform deep comparison.
AIIR_CAPI_EXPORTED bool aiirBlockEqual(AiirBlock block, AiirBlock other);

/// Returns the closest surrounding operation that contains this block.
AIIR_CAPI_EXPORTED AiirOperation aiirBlockGetParentOperation(AiirBlock);

/// Returns the region that contains this block.
AIIR_CAPI_EXPORTED AiirRegion aiirBlockGetParentRegion(AiirBlock block);

/// Returns the block immediately following the given block in its parent
/// region.
AIIR_CAPI_EXPORTED AiirBlock aiirBlockGetNextInRegion(AiirBlock block);

/// Returns the first operation in the block.
AIIR_CAPI_EXPORTED AiirOperation aiirBlockGetFirstOperation(AiirBlock block);

/// Returns the terminator operation in the block or null if no terminator.
AIIR_CAPI_EXPORTED AiirOperation aiirBlockGetTerminator(AiirBlock block);

/// Takes an operation owned by the caller and appends it to the block.
AIIR_CAPI_EXPORTED void aiirBlockAppendOwnedOperation(AiirBlock block,
                                                      AiirOperation operation);

/// Takes an operation owned by the caller and inserts it as `pos` to the block.
/// This is an expensive operation that scans the block linearly, prefer
/// insertBefore/After instead.
AIIR_CAPI_EXPORTED void aiirBlockInsertOwnedOperation(AiirBlock block,
                                                      intptr_t pos,
                                                      AiirOperation operation);

/// Takes an operation owned by the caller and inserts it after the (non-owned)
/// reference operation in the given block. If the reference is null, prepends
/// the operation. Otherwise, the reference must belong to the block.
AIIR_CAPI_EXPORTED void
aiirBlockInsertOwnedOperationAfter(AiirBlock block, AiirOperation reference,
                                   AiirOperation operation);

/// Takes an operation owned by the caller and inserts it before the (non-owned)
/// reference operation in the given block. If the reference is null, appends
/// the operation. Otherwise, the reference must belong to the block.
AIIR_CAPI_EXPORTED void
aiirBlockInsertOwnedOperationBefore(AiirBlock block, AiirOperation reference,
                                    AiirOperation operation);

/// Returns the number of arguments of the block.
AIIR_CAPI_EXPORTED intptr_t aiirBlockGetNumArguments(AiirBlock block);

/// Appends an argument of the specified type to the block. Returns the newly
/// added argument.
AIIR_CAPI_EXPORTED AiirValue aiirBlockAddArgument(AiirBlock block,
                                                  AiirType type,
                                                  AiirLocation loc);

/// Erase the argument at 'index' and remove it from the argument list.
AIIR_CAPI_EXPORTED void aiirBlockEraseArgument(AiirBlock block, unsigned index);

/// Inserts an argument of the specified type at a specified index to the block.
/// Returns the newly added argument.
AIIR_CAPI_EXPORTED AiirValue aiirBlockInsertArgument(AiirBlock block,
                                                     intptr_t pos,
                                                     AiirType type,
                                                     AiirLocation loc);

/// Returns `pos`-th argument of the block.
AIIR_CAPI_EXPORTED AiirValue aiirBlockGetArgument(AiirBlock block,
                                                  intptr_t pos);

/// Prints a block by sending chunks of the string representation and
/// forwarding `userData to `callback`. Note that the callback may be called
/// several times with consecutive chunks of the string.
AIIR_CAPI_EXPORTED void
aiirBlockPrint(AiirBlock block, AiirStringCallback callback, void *userData);

/// Returns the number of successor blocks of the block.
AIIR_CAPI_EXPORTED intptr_t aiirBlockGetNumSuccessors(AiirBlock block);

/// Returns `pos`-th successor of the block.
AIIR_CAPI_EXPORTED AiirBlock aiirBlockGetSuccessor(AiirBlock block,
                                                   intptr_t pos);

/// Returns the number of predecessor blocks of the block.
AIIR_CAPI_EXPORTED intptr_t aiirBlockGetNumPredecessors(AiirBlock block);

/// Returns `pos`-th predecessor of the block.
///
/// WARNING: This getter is more expensive than the others here because
/// the impl actually iterates the use-def chain (of block operands) anew for
/// each indexed access.
AIIR_CAPI_EXPORTED AiirBlock aiirBlockGetPredecessor(AiirBlock block,
                                                     intptr_t pos);

//===----------------------------------------------------------------------===//
// Value API.
//===----------------------------------------------------------------------===//

/// Returns whether the value is null.
static inline bool aiirValueIsNull(AiirValue value) { return !value.ptr; }

/// Returns 1 if two values are equal, 0 otherwise.
AIIR_CAPI_EXPORTED bool aiirValueEqual(AiirValue value1, AiirValue value2);

/// Returns 1 if the value is a block argument, 0 otherwise.
AIIR_CAPI_EXPORTED bool aiirValueIsABlockArgument(AiirValue value);

/// Returns 1 if the value is an operation result, 0 otherwise.
AIIR_CAPI_EXPORTED bool aiirValueIsAOpResult(AiirValue value);

/// Returns the block in which this value is defined as an argument. Asserts if
/// the value is not a block argument.
AIIR_CAPI_EXPORTED AiirBlock aiirBlockArgumentGetOwner(AiirValue value);

/// Returns the position of the value in the argument list of its block.
AIIR_CAPI_EXPORTED intptr_t aiirBlockArgumentGetArgNumber(AiirValue value);

/// Sets the type of the block argument to the given type.
AIIR_CAPI_EXPORTED void aiirBlockArgumentSetType(AiirValue value,
                                                 AiirType type);

/// Sets the location of the block argument to the given location.
AIIR_CAPI_EXPORTED void aiirBlockArgumentSetLocation(AiirValue value,
                                                     AiirLocation loc);

/// Returns an operation that produced this value as its result. Asserts if the
/// value is not an op result.
AIIR_CAPI_EXPORTED AiirOperation aiirOpResultGetOwner(AiirValue value);

/// Returns the position of the value in the list of results of the operation
/// that produced it.
AIIR_CAPI_EXPORTED intptr_t aiirOpResultGetResultNumber(AiirValue value);

/// Returns the type of the value.
AIIR_CAPI_EXPORTED AiirType aiirValueGetType(AiirValue value);

/// Set the type of the value.
AIIR_CAPI_EXPORTED void aiirValueSetType(AiirValue value, AiirType type);

/// Prints the value to the standard error stream.
AIIR_CAPI_EXPORTED void aiirValueDump(AiirValue value);

/// Prints a value by sending chunks of the string representation and
/// forwarding `userData to `callback`. Note that the callback may be called
/// several times with consecutive chunks of the string.
AIIR_CAPI_EXPORTED void
aiirValuePrint(AiirValue value, AiirStringCallback callback, void *userData);

/// Prints a value as an operand (i.e., the ValueID).
AIIR_CAPI_EXPORTED void aiirValuePrintAsOperand(AiirValue value,
                                                AiirAsmState state,
                                                AiirStringCallback callback,
                                                void *userData);

/// Returns an op operand representing the first use of the value, or a null op
/// operand if there are no uses.
AIIR_CAPI_EXPORTED AiirOpOperand aiirValueGetFirstUse(AiirValue value);

/// Replace all uses of 'of' value with the 'with' value, updating anything in
/// the IR that uses 'of' to use the other value instead.  When this returns
/// there are zero uses of 'of'.
AIIR_CAPI_EXPORTED void aiirValueReplaceAllUsesOfWith(AiirValue of,
                                                      AiirValue with);

/// Replace all uses of 'of' value with 'with' value, updating anything in the
/// IR that uses 'of' to use 'with' instead, except if the user is listed in
/// 'exceptions'. The 'exceptions' parameter is an array of AiirOperation
/// pointers with a length of 'numExceptions'.
AIIR_CAPI_EXPORTED void
aiirValueReplaceAllUsesExcept(AiirValue of, AiirValue with,
                              intptr_t numExceptions,
                              AiirOperation *exceptions);

/// Gets the location of the value.
AIIR_CAPI_EXPORTED AiirLocation aiirValueGetLocation(AiirValue v);

/// Gets the context that a value was created with.
AIIR_CAPI_EXPORTED AiirContext aiirValueGetContext(AiirValue v);

//===----------------------------------------------------------------------===//
// OpOperand API.
//===----------------------------------------------------------------------===//

/// Returns whether the op operand is null.
AIIR_CAPI_EXPORTED bool aiirOpOperandIsNull(AiirOpOperand opOperand);

/// Returns the value of an op operand.
AIIR_CAPI_EXPORTED AiirValue aiirOpOperandGetValue(AiirOpOperand opOperand);

/// Returns the owner operation of an op operand.
AIIR_CAPI_EXPORTED AiirOperation aiirOpOperandGetOwner(AiirOpOperand opOperand);

/// Returns the operand number of an op operand.
AIIR_CAPI_EXPORTED unsigned
aiirOpOperandGetOperandNumber(AiirOpOperand opOperand);

/// Returns an op operand representing the next use of the value, or a null op
/// operand if there is no next use.
AIIR_CAPI_EXPORTED AiirOpOperand
aiirOpOperandGetNextUse(AiirOpOperand opOperand);

//===----------------------------------------------------------------------===//
// Type API.
//===----------------------------------------------------------------------===//

/// Parses a type. The type is owned by the context.
AIIR_CAPI_EXPORTED AiirType aiirTypeParseGet(AiirContext context,
                                             AiirStringRef type);

/// Gets the context that a type was created with.
AIIR_CAPI_EXPORTED AiirContext aiirTypeGetContext(AiirType type);

/// Gets the type ID of the type.
AIIR_CAPI_EXPORTED AiirTypeID aiirTypeGetTypeID(AiirType type);

/// Gets the dialect a type belongs to.
AIIR_CAPI_EXPORTED AiirDialect aiirTypeGetDialect(AiirType type);

/// Checks whether a type is null.
static inline bool aiirTypeIsNull(AiirType type) { return !type.ptr; }

/// Checks if two types are equal.
AIIR_CAPI_EXPORTED bool aiirTypeEqual(AiirType t1, AiirType t2);

/// Prints a location by sending chunks of the string representation and
/// forwarding `userData to `callback`. Note that the callback may be called
/// several times with consecutive chunks of the string.
AIIR_CAPI_EXPORTED void
aiirTypePrint(AiirType type, AiirStringCallback callback, void *userData);

/// Prints the type to the standard error stream.
AIIR_CAPI_EXPORTED void aiirTypeDump(AiirType type);

//===----------------------------------------------------------------------===//
// Attribute API.
//===----------------------------------------------------------------------===//

/// Parses an attribute. The attribute is owned by the context.
AIIR_CAPI_EXPORTED AiirAttribute aiirAttributeParseGet(AiirContext context,
                                                       AiirStringRef attr);

/// Gets the context that an attribute was created with.
AIIR_CAPI_EXPORTED AiirContext aiirAttributeGetContext(AiirAttribute attribute);

/// Gets the type of this attribute.
AIIR_CAPI_EXPORTED AiirType aiirAttributeGetType(AiirAttribute attribute);

/// Gets the type id of the attribute.
AIIR_CAPI_EXPORTED AiirTypeID aiirAttributeGetTypeID(AiirAttribute attribute);

/// Gets the dialect of the attribute.
AIIR_CAPI_EXPORTED AiirDialect aiirAttributeGetDialect(AiirAttribute attribute);

/// Checks whether an attribute is null.
static inline bool aiirAttributeIsNull(AiirAttribute attr) { return !attr.ptr; }

/// Checks if two attributes are equal.
AIIR_CAPI_EXPORTED bool aiirAttributeEqual(AiirAttribute a1, AiirAttribute a2);

/// Prints an attribute by sending chunks of the string representation and
/// forwarding `userData to `callback`. Note that the callback may be called
/// several times with consecutive chunks of the string.
AIIR_CAPI_EXPORTED void aiirAttributePrint(AiirAttribute attr,
                                           AiirStringCallback callback,
                                           void *userData);

/// Prints the attribute to the standard error stream.
AIIR_CAPI_EXPORTED void aiirAttributeDump(AiirAttribute attr);

/// Associates an attribute with the name. Takes ownership of neither.
AIIR_CAPI_EXPORTED AiirNamedAttribute aiirNamedAttributeGet(AiirIdentifier name,
                                                            AiirAttribute attr);

//===----------------------------------------------------------------------===//
// Identifier API.
//===----------------------------------------------------------------------===//

/// Gets an identifier with the given string value.
AIIR_CAPI_EXPORTED AiirIdentifier aiirIdentifierGet(AiirContext context,
                                                    AiirStringRef str);

/// Returns the context associated with this identifier
AIIR_CAPI_EXPORTED AiirContext aiirIdentifierGetContext(AiirIdentifier);

/// Checks whether two identifiers are the same.
AIIR_CAPI_EXPORTED bool aiirIdentifierEqual(AiirIdentifier ident,
                                            AiirIdentifier other);

/// Gets the string value of the identifier.
AIIR_CAPI_EXPORTED AiirStringRef aiirIdentifierStr(AiirIdentifier ident);

//===----------------------------------------------------------------------===//
// Symbol and SymbolTable API.
//===----------------------------------------------------------------------===//

/// Returns the name of the attribute used to store symbol names compatible with
/// symbol tables.
AIIR_CAPI_EXPORTED AiirStringRef aiirSymbolTableGetSymbolAttributeName(void);

/// Returns the name of the attribute used to store symbol visibility.
AIIR_CAPI_EXPORTED AiirStringRef
aiirSymbolTableGetVisibilityAttributeName(void);

/// Creates a symbol table for the given operation. If the operation does not
/// have the SymbolTable trait, returns a null symbol table.
AIIR_CAPI_EXPORTED AiirSymbolTable
aiirSymbolTableCreate(AiirOperation operation);

/// Returns true if the symbol table is null.
static inline bool aiirSymbolTableIsNull(AiirSymbolTable symbolTable) {
  return !symbolTable.ptr;
}

/// Destroys the symbol table created with aiirSymbolTableCreate. This does not
/// affect the operations in the table.
AIIR_CAPI_EXPORTED void aiirSymbolTableDestroy(AiirSymbolTable symbolTable);

/// Looks up a symbol with the given name in the given symbol table and returns
/// the operation that corresponds to the symbol. If the symbol cannot be found,
/// returns a null operation.
AIIR_CAPI_EXPORTED AiirOperation
aiirSymbolTableLookup(AiirSymbolTable symbolTable, AiirStringRef name);

/// Inserts the given operation into the given symbol table. The operation must
/// have the symbol trait. If the symbol table already has a symbol with the
/// same name, renames the symbol being inserted to ensure name uniqueness. Note
/// that this does not move the operation itself into the block of the symbol
/// table operation, this should be done separately. Returns the name of the
/// symbol after insertion.
AIIR_CAPI_EXPORTED AiirAttribute
aiirSymbolTableInsert(AiirSymbolTable symbolTable, AiirOperation operation);

/// Removes the given operation from the symbol table and erases it.
AIIR_CAPI_EXPORTED void aiirSymbolTableErase(AiirSymbolTable symbolTable,
                                             AiirOperation operation);

/// Attempt to replace all uses that are nested within the given operation
/// of the given symbol 'oldSymbol' with the provided 'newSymbol'. This does
/// not traverse into nested symbol tables. Will fail atomically if there are
/// any unknown operations that may be potential symbol tables.
AIIR_CAPI_EXPORTED AiirLogicalResult aiirSymbolTableReplaceAllSymbolUses(
    AiirStringRef oldSymbol, AiirStringRef newSymbol, AiirOperation from);

/// Walks all symbol table operations nested within, and including, `op`. For
/// each symbol table operation, the provided callback is invoked with the op
/// and a boolean signifying if the symbols within that symbol table can be
/// treated as if all uses within the IR are visible to the caller.
/// `allSymUsesVisible` identifies whether all of the symbol uses of symbols
/// within `op` are visible.
AIIR_CAPI_EXPORTED void aiirSymbolTableWalkSymbolTables(
    AiirOperation from, bool allSymUsesVisible,
    void (*callback)(AiirOperation, bool, void *userData), void *userData);

#ifdef __cplusplus
}
#endif

#endif // AIIR_C_IR_H
