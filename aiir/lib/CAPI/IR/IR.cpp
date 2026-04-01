//===- IR.cpp - C Interface for Core AIIR APIs ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir-c/IR.h"
#include "aiir-c/Support.h"

#include "aiir/AsmParser/AsmParser.h"
#include "aiir/Bytecode/BytecodeWriter.h"
#include "aiir/CAPI/IR.h"
#include "aiir/CAPI/Support.h"
#include "aiir/CAPI/Utils.h"
#include "aiir/IR/Attributes.h"
#include "aiir/IR/BuiltinAttributes.h"
#include "aiir/IR/BuiltinOps.h"
#include "aiir/IR/Diagnostics.h"
#include "aiir/IR/Dialect.h"
#include "aiir/IR/Location.h"
#include "aiir/IR/Operation.h"
#include "aiir/IR/OperationSupport.h"
#include "aiir/IR/OwningOpRef.h"
#include "aiir/IR/Types.h"
#include "aiir/IR/Value.h"
#include "aiir/IR/Verifier.h"
#include "aiir/IR/Visitors.h"
#include "aiir/Interfaces/InferTypeOpInterface.h"
#include "aiir/Parser/Parser.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <cstddef>
#include <memory>
#include <optional>

using namespace aiir;

//===----------------------------------------------------------------------===//
// Context API.
//===----------------------------------------------------------------------===//

AiirContext aiirContextCreate() {
  auto *context = new AIIRContext;
  return wrap(context);
}

static inline AIIRContext::Threading toThreadingEnum(bool threadingEnabled) {
  return threadingEnabled ? AIIRContext::Threading::ENABLED
                          : AIIRContext::Threading::DISABLED;
}

AiirContext aiirContextCreateWithThreading(bool threadingEnabled) {
  auto *context = new AIIRContext(toThreadingEnum(threadingEnabled));
  return wrap(context);
}

AiirContext aiirContextCreateWithRegistry(AiirDialectRegistry registry,
                                          bool threadingEnabled) {
  auto *context =
      new AIIRContext(*unwrap(registry), toThreadingEnum(threadingEnabled));
  return wrap(context);
}

bool aiirContextEqual(AiirContext ctx1, AiirContext ctx2) {
  return unwrap(ctx1) == unwrap(ctx2);
}

void aiirContextDestroy(AiirContext context) { delete unwrap(context); }

void aiirContextSetAllowUnregisteredDialects(AiirContext context, bool allow) {
  unwrap(context)->allowUnregisteredDialects(allow);
}

bool aiirContextGetAllowUnregisteredDialects(AiirContext context) {
  return unwrap(context)->allowsUnregisteredDialects();
}
intptr_t aiirContextGetNumRegisteredDialects(AiirContext context) {
  return static_cast<intptr_t>(unwrap(context)->getAvailableDialects().size());
}

void aiirContextAppendDialectRegistry(AiirContext ctx,
                                      AiirDialectRegistry registry) {
  unwrap(ctx)->appendDialectRegistry(*unwrap(registry));
}

// TODO: expose a cheaper way than constructing + sorting a vector only to take
// its size.
intptr_t aiirContextGetNumLoadedDialects(AiirContext context) {
  return static_cast<intptr_t>(unwrap(context)->getLoadedDialects().size());
}

AiirDialect aiirContextGetOrLoadDialect(AiirContext context,
                                        AiirStringRef name) {
  return wrap(unwrap(context)->getOrLoadDialect(unwrap(name)));
}

bool aiirContextIsRegisteredOperation(AiirContext context, AiirStringRef name) {
  return unwrap(context)->isOperationRegistered(unwrap(name));
}

void aiirContextEnableMultithreading(AiirContext context, bool enable) {
  return unwrap(context)->enableMultithreading(enable);
}

void aiirContextLoadAllAvailableDialects(AiirContext context) {
  unwrap(context)->loadAllAvailableDialects();
}

void aiirContextSetThreadPool(AiirContext context,
                              AiirLlvmThreadPool threadPool) {
  unwrap(context)->setThreadPool(*unwrap(threadPool));
}

unsigned aiirContextGetNumThreads(AiirContext context) {
  return unwrap(context)->getNumThreads();
}

AiirLlvmThreadPool aiirContextGetThreadPool(AiirContext context) {
  return wrap(&unwrap(context)->getThreadPool());
}

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//

AiirContext aiirDialectGetContext(AiirDialect dialect) {
  return wrap(unwrap(dialect)->getContext());
}

bool aiirDialectEqual(AiirDialect dialect1, AiirDialect dialect2) {
  return unwrap(dialect1) == unwrap(dialect2);
}

AiirStringRef aiirDialectGetNamespace(AiirDialect dialect) {
  return wrap(unwrap(dialect)->getNamespace());
}

//===----------------------------------------------------------------------===//
// DialectRegistry API.
//===----------------------------------------------------------------------===//

AiirDialectRegistry aiirDialectRegistryCreate() {
  return wrap(new DialectRegistry());
}

void aiirDialectRegistryDestroy(AiirDialectRegistry registry) {
  delete unwrap(registry);
}

//===----------------------------------------------------------------------===//
// AsmState API.
//===----------------------------------------------------------------------===//

AiirAsmState aiirAsmStateCreateForOperation(AiirOperation op,
                                            AiirOpPrintingFlags flags) {
  return wrap(new AsmState(unwrap(op), *unwrap(flags)));
}

static Operation *findParent(Operation *op, bool shouldUseLocalScope) {
  do {
    // If we are printing local scope, stop at the first operation that is
    // isolated from above.
    if (shouldUseLocalScope && op->hasTrait<OpTrait::IsIsolatedFromAbove>())
      break;

    // Otherwise, traverse up to the next parent.
    Operation *parentOp = op->getParentOp();
    if (!parentOp)
      break;
    op = parentOp;
  } while (true);
  return op;
}

AiirAsmState aiirAsmStateCreateForValue(AiirValue value,
                                        AiirOpPrintingFlags flags) {
  Operation *op;
  aiir::Value val = unwrap(value);
  if (auto result = llvm::dyn_cast<OpResult>(val)) {
    op = result.getOwner();
  } else {
    op = llvm::cast<BlockArgument>(val).getOwner()->getParentOp();
    if (!op) {
      emitError(val.getLoc()) << "<<UNKNOWN SSA VALUE>>";
      return {nullptr};
    }
  }
  op = findParent(op, unwrap(flags)->shouldUseLocalScope());
  return wrap(new AsmState(op, *unwrap(flags)));
}

/// Destroys printing flags created with aiirAsmStateCreate.
void aiirAsmStateDestroy(AiirAsmState state) { delete unwrap(state); }

//===----------------------------------------------------------------------===//
// Printing flags API.
//===----------------------------------------------------------------------===//

AiirOpPrintingFlags aiirOpPrintingFlagsCreate() {
  return wrap(new OpPrintingFlags());
}

void aiirOpPrintingFlagsDestroy(AiirOpPrintingFlags flags) {
  delete unwrap(flags);
}

void aiirOpPrintingFlagsElideLargeElementsAttrs(AiirOpPrintingFlags flags,
                                                intptr_t largeElementLimit) {
  unwrap(flags)->elideLargeElementsAttrs(largeElementLimit);
}

void aiirOpPrintingFlagsElideLargeResourceString(AiirOpPrintingFlags flags,
                                                 intptr_t largeResourceLimit) {
  unwrap(flags)->elideLargeResourceString(largeResourceLimit);
}

void aiirOpPrintingFlagsEnableDebugInfo(AiirOpPrintingFlags flags, bool enable,
                                        bool prettyForm) {
  unwrap(flags)->enableDebugInfo(enable, /*prettyForm=*/prettyForm);
}

void aiirOpPrintingFlagsPrintGenericOpForm(AiirOpPrintingFlags flags) {
  unwrap(flags)->printGenericOpForm();
}

void aiirOpPrintingFlagsPrintNameLocAsPrefix(AiirOpPrintingFlags flags) {
  unwrap(flags)->printNameLocAsPrefix();
}

void aiirOpPrintingFlagsUseLocalScope(AiirOpPrintingFlags flags) {
  unwrap(flags)->useLocalScope();
}

void aiirOpPrintingFlagsAssumeVerified(AiirOpPrintingFlags flags) {
  unwrap(flags)->assumeVerified();
}

void aiirOpPrintingFlagsSkipRegions(AiirOpPrintingFlags flags) {
  unwrap(flags)->skipRegions();
}
//===----------------------------------------------------------------------===//
// Bytecode printing flags API.
//===----------------------------------------------------------------------===//

AiirBytecodeWriterConfig aiirBytecodeWriterConfigCreate() {
  return wrap(new BytecodeWriterConfig());
}

void aiirBytecodeWriterConfigDestroy(AiirBytecodeWriterConfig config) {
  delete unwrap(config);
}

void aiirBytecodeWriterConfigDesiredEmitVersion(AiirBytecodeWriterConfig flags,
                                                int64_t version) {
  unwrap(flags)->setDesiredBytecodeVersion(version);
}

//===----------------------------------------------------------------------===//
// Location API.
//===----------------------------------------------------------------------===//

AiirAttribute aiirLocationGetAttribute(AiirLocation location) {
  return wrap(LocationAttr(unwrap(location)));
}

AiirLocation aiirLocationFromAttribute(AiirAttribute attribute) {
  return wrap(Location(llvm::dyn_cast<LocationAttr>(unwrap(attribute))));
}

AiirLocation aiirLocationFileLineColGet(AiirContext context,
                                        AiirStringRef filename, unsigned line,
                                        unsigned col) {
  return wrap(Location(
      FileLineColLoc::get(unwrap(context), unwrap(filename), line, col)));
}

AiirLocation
aiirLocationFileLineColRangeGet(AiirContext context, AiirStringRef filename,
                                unsigned startLine, unsigned startCol,
                                unsigned endLine, unsigned endCol) {
  return wrap(
      Location(FileLineColRange::get(unwrap(context), unwrap(filename),
                                     startLine, startCol, endLine, endCol)));
}

AiirIdentifier aiirLocationFileLineColRangeGetFilename(AiirLocation location) {
  return wrap(llvm::dyn_cast<FileLineColRange>(unwrap(location)).getFilename());
}

int aiirLocationFileLineColRangeGetStartLine(AiirLocation location) {
  if (auto loc = llvm::dyn_cast<FileLineColRange>(unwrap(location)))
    return loc.getStartLine();
  return -1;
}

int aiirLocationFileLineColRangeGetStartColumn(AiirLocation location) {
  if (auto loc = llvm::dyn_cast<FileLineColRange>(unwrap(location)))
    return loc.getStartColumn();
  return -1;
}

int aiirLocationFileLineColRangeGetEndLine(AiirLocation location) {
  if (auto loc = llvm::dyn_cast<FileLineColRange>(unwrap(location)))
    return loc.getEndLine();
  return -1;
}

int aiirLocationFileLineColRangeGetEndColumn(AiirLocation location) {
  if (auto loc = llvm::dyn_cast<FileLineColRange>(unwrap(location)))
    return loc.getEndColumn();
  return -1;
}

AiirTypeID aiirLocationFileLineColRangeGetTypeID() {
  return wrap(FileLineColRange::getTypeID());
}

bool aiirLocationIsAFileLineColRange(AiirLocation location) {
  return isa<FileLineColRange>(unwrap(location));
}

AiirLocation aiirLocationCallSiteGet(AiirLocation callee, AiirLocation caller) {
  return wrap(Location(CallSiteLoc::get(unwrap(callee), unwrap(caller))));
}

AiirLocation aiirLocationCallSiteGetCallee(AiirLocation location) {
  return wrap(
      Location(llvm::dyn_cast<CallSiteLoc>(unwrap(location)).getCallee()));
}

AiirLocation aiirLocationCallSiteGetCaller(AiirLocation location) {
  return wrap(
      Location(llvm::dyn_cast<CallSiteLoc>(unwrap(location)).getCaller()));
}

AiirTypeID aiirLocationCallSiteGetTypeID() {
  return wrap(CallSiteLoc::getTypeID());
}

bool aiirLocationIsACallSite(AiirLocation location) {
  return isa<CallSiteLoc>(unwrap(location));
}

AiirLocation aiirLocationFusedGet(AiirContext ctx, intptr_t nLocations,
                                  AiirLocation const *locations,
                                  AiirAttribute metadata) {
  SmallVector<Location, 4> locs;
  ArrayRef<Location> unwrappedLocs = unwrapList(nLocations, locations, locs);
  return wrap(FusedLoc::get(unwrappedLocs, unwrap(metadata), unwrap(ctx)));
}

unsigned aiirLocationFusedGetNumLocations(AiirLocation location) {
  if (auto locationsArrRef = llvm::dyn_cast<FusedLoc>(unwrap(location)))
    return locationsArrRef.getLocations().size();
  return 0;
}

void aiirLocationFusedGetLocations(AiirLocation location,
                                   AiirLocation *locationsCPtr) {
  if (auto locationsArrRef = llvm::dyn_cast<FusedLoc>(unwrap(location))) {
    for (auto [i, location] : llvm::enumerate(locationsArrRef.getLocations()))
      locationsCPtr[i] = wrap(location);
  }
}

AiirAttribute aiirLocationFusedGetMetadata(AiirLocation location) {
  return wrap(llvm::dyn_cast<FusedLoc>(unwrap(location)).getMetadata());
}

AiirTypeID aiirLocationFusedGetTypeID() { return wrap(FusedLoc::getTypeID()); }

bool aiirLocationIsAFused(AiirLocation location) {
  return isa<FusedLoc>(unwrap(location));
}

AiirLocation aiirLocationNameGet(AiirContext context, AiirStringRef name,
                                 AiirLocation childLoc) {
  if (aiirLocationIsNull(childLoc))
    return wrap(
        Location(NameLoc::get(StringAttr::get(unwrap(context), unwrap(name)))));
  return wrap(Location(NameLoc::get(
      StringAttr::get(unwrap(context), unwrap(name)), unwrap(childLoc))));
}

AiirIdentifier aiirLocationNameGetName(AiirLocation location) {
  return wrap((llvm::dyn_cast<NameLoc>(unwrap(location)).getName()));
}

AiirLocation aiirLocationNameGetChildLoc(AiirLocation location) {
  return wrap(
      Location(llvm::dyn_cast<NameLoc>(unwrap(location)).getChildLoc()));
}

AiirTypeID aiirLocationNameGetTypeID() { return wrap(NameLoc::getTypeID()); }

bool aiirLocationIsAName(AiirLocation location) {
  return isa<NameLoc>(unwrap(location));
}

AiirLocation aiirLocationUnknownGet(AiirContext context) {
  return wrap(Location(UnknownLoc::get(unwrap(context))));
}

bool aiirLocationEqual(AiirLocation l1, AiirLocation l2) {
  return unwrap(l1) == unwrap(l2);
}

AiirContext aiirLocationGetContext(AiirLocation location) {
  return wrap(unwrap(location).getContext());
}

void aiirLocationPrint(AiirLocation location, AiirStringCallback callback,
                       void *userData) {
  detail::CallbackOstream stream(callback, userData);
  unwrap(location).print(stream);
}

//===----------------------------------------------------------------------===//
// Module API.
//===----------------------------------------------------------------------===//

AiirModule aiirModuleCreateEmpty(AiirLocation location) {
  return wrap(ModuleOp::create(unwrap(location)));
}

AiirModule aiirModuleCreateParse(AiirContext context, AiirStringRef module) {
  OwningOpRef<ModuleOp> owning =
      parseSourceString<ModuleOp>(unwrap(module), unwrap(context));
  if (!owning)
    return AiirModule{nullptr};
  return AiirModule{owning.release().getOperation()};
}

AiirModule aiirModuleCreateParseFromFile(AiirContext context,
                                         AiirStringRef fileName) {
  OwningOpRef<ModuleOp> owning =
      parseSourceFile<ModuleOp>(unwrap(fileName), unwrap(context));
  if (!owning)
    return AiirModule{nullptr};
  return AiirModule{owning.release().getOperation()};
}

AiirContext aiirModuleGetContext(AiirModule module) {
  return wrap(unwrap(module).getContext());
}

AiirBlock aiirModuleGetBody(AiirModule module) {
  return wrap(unwrap(module).getBody());
}

void aiirModuleDestroy(AiirModule module) {
  // Transfer ownership to an OwningOpRef<ModuleOp> so that its destructor is
  // called.
  OwningOpRef<ModuleOp>(unwrap(module));
}

AiirOperation aiirModuleGetOperation(AiirModule module) {
  return wrap(unwrap(module).getOperation());
}

AiirModule aiirModuleFromOperation(AiirOperation op) {
  return wrap(dyn_cast<ModuleOp>(unwrap(op)));
}

bool aiirModuleEqual(AiirModule lhs, AiirModule rhs) {
  return unwrap(lhs) == unwrap(rhs);
}

size_t aiirModuleHashValue(AiirModule mod) {
  return OperationEquivalence::computeHash(unwrap(mod).getOperation());
}

//===----------------------------------------------------------------------===//
// Operation state API.
//===----------------------------------------------------------------------===//

AiirOperationState aiirOperationStateGet(AiirStringRef name, AiirLocation loc) {
  AiirOperationState state;
  state.name = name;
  state.location = loc;
  state.nResults = 0;
  state.results = nullptr;
  state.nOperands = 0;
  state.operands = nullptr;
  state.nRegions = 0;
  state.regions = nullptr;
  state.nSuccessors = 0;
  state.successors = nullptr;
  state.nAttributes = 0;
  state.attributes = nullptr;
  state.enableResultTypeInference = false;
  return state;
}

#define APPEND_ELEMS(type, sizeName, elemName)                                 \
  state->elemName =                                                            \
      (type *)realloc(state->elemName, (state->sizeName + n) * sizeof(type));  \
  memcpy(state->elemName + state->sizeName, elemName, n * sizeof(type));       \
  state->sizeName += n;

void aiirOperationStateAddResults(AiirOperationState *state, intptr_t n,
                                  AiirType const *results) {
  APPEND_ELEMS(AiirType, nResults, results);
}

void aiirOperationStateAddOperands(AiirOperationState *state, intptr_t n,
                                   AiirValue const *operands) {
  APPEND_ELEMS(AiirValue, nOperands, operands);
}
void aiirOperationStateAddOwnedRegions(AiirOperationState *state, intptr_t n,
                                       AiirRegion const *regions) {
  APPEND_ELEMS(AiirRegion, nRegions, regions);
}
void aiirOperationStateAddSuccessors(AiirOperationState *state, intptr_t n,
                                     AiirBlock const *successors) {
  APPEND_ELEMS(AiirBlock, nSuccessors, successors);
}
void aiirOperationStateAddAttributes(AiirOperationState *state, intptr_t n,
                                     AiirNamedAttribute const *attributes) {
  APPEND_ELEMS(AiirNamedAttribute, nAttributes, attributes);
}

void aiirOperationStateEnableResultTypeInference(AiirOperationState *state) {
  state->enableResultTypeInference = true;
}

//===----------------------------------------------------------------------===//
// Operation API.
//===----------------------------------------------------------------------===//

static LogicalResult inferOperationTypes(OperationState &state) {
  AIIRContext *context = state.getContext();
  std::optional<RegisteredOperationName> info = state.name.getRegisteredInfo();
  if (!info) {
    emitError(state.location)
        << "type inference was requested for the operation " << state.name
        << ", but the operation was not registered; ensure that the dialect "
           "containing the operation is linked into AIIR and registered with "
           "the context";
    return failure();
  }

  auto *inferInterface = info->getInterface<InferTypeOpInterface>();
  if (!inferInterface) {
    emitError(state.location)
        << "type inference was requested for the operation " << state.name
        << ", but the operation does not support type inference; result "
           "types must be specified explicitly";
    return failure();
  }

  DictionaryAttr attributes = state.attributes.getDictionary(context);
  OpaqueProperties properties = state.getRawProperties();

  if (!properties && info->getOpPropertyByteSize() > 0 && !attributes.empty()) {
    auto prop = std::make_unique<char[]>(info->getOpPropertyByteSize());
    properties = OpaqueProperties(prop.get());
    if (properties) {
      auto emitError = [&]() {
        return aiir::emitError(state.location)
               << " failed properties conversion while building "
               << state.name.getStringRef() << " with `" << attributes << "`: ";
      };
      if (failed(info->setOpPropertiesFromAttribute(state.name, properties,
                                                    attributes, emitError)))
        return failure();
    }
    if (succeeded(inferInterface->inferReturnTypes(
            context, state.location, state.operands, attributes, properties,
            state.regions, state.types))) {
      return success();
    }
    // Diagnostic emitted by interface.
    return failure();
  }

  if (succeeded(inferInterface->inferReturnTypes(
          context, state.location, state.operands, attributes, properties,
          state.regions, state.types)))
    return success();

  // Diagnostic emitted by interface.
  return failure();
}

AiirOperation aiirOperationCreate(AiirOperationState *state) {
  assert(state);
  OperationState cppState(unwrap(state->location), unwrap(state->name));
  SmallVector<Type, 4> resultStorage;
  SmallVector<Value, 8> operandStorage;
  SmallVector<Block *, 2> successorStorage;
  cppState.addTypes(unwrapList(state->nResults, state->results, resultStorage));
  cppState.addOperands(
      unwrapList(state->nOperands, state->operands, operandStorage));
  cppState.addSuccessors(
      unwrapList(state->nSuccessors, state->successors, successorStorage));

  cppState.attributes.reserve(state->nAttributes);
  for (intptr_t i = 0; i < state->nAttributes; ++i)
    cppState.addAttribute(unwrap(state->attributes[i].name),
                          unwrap(state->attributes[i].attribute));

  for (intptr_t i = 0; i < state->nRegions; ++i)
    cppState.addRegion(std::unique_ptr<Region>(unwrap(state->regions[i])));

  free(state->results);
  free(state->operands);
  free(state->successors);
  free(state->regions);
  free(state->attributes);

  // Infer result types.
  if (state->enableResultTypeInference) {
    assert(cppState.types.empty() &&
           "result type inference enabled and result types provided");
    if (failed(inferOperationTypes(cppState)))
      return {nullptr};
  }

  return wrap(Operation::create(cppState));
}

AiirOperation aiirOperationCreateParse(AiirContext context,
                                       AiirStringRef sourceStr,
                                       AiirStringRef sourceName) {

  return wrap(
      parseSourceString(unwrap(sourceStr), unwrap(context), unwrap(sourceName))
          .release());
}

AiirOperation aiirOperationClone(AiirOperation op) {
  return wrap(unwrap(op)->clone());
}

void aiirOperationDestroy(AiirOperation op) { unwrap(op)->erase(); }

void aiirOperationRemoveFromParent(AiirOperation op) { unwrap(op)->remove(); }

bool aiirOperationEqual(AiirOperation op, AiirOperation other) {
  return unwrap(op) == unwrap(other);
}

size_t aiirOperationHashValue(AiirOperation op) {
  return OperationEquivalence::computeHash(unwrap(op));
}

AiirContext aiirOperationGetContext(AiirOperation op) {
  return wrap(unwrap(op)->getContext());
}

bool aiirOperationNameHasTrait(AiirStringRef opName, AiirTypeID traitTypeID,
                               AiirContext context) {
  return OperationName(unwrap(opName), unwrap(context))
      .hasTrait(unwrap(traitTypeID));
}

AiirLocation aiirOperationGetLocation(AiirOperation op) {
  return wrap(unwrap(op)->getLoc());
}

void aiirOperationSetLocation(AiirOperation op, AiirLocation loc) {
  unwrap(op)->setLoc(unwrap(loc));
}

AiirTypeID aiirOperationGetTypeID(AiirOperation op) {
  if (auto info = unwrap(op)->getRegisteredInfo())
    return wrap(info->getTypeID());
  return {nullptr};
}

AiirIdentifier aiirOperationGetName(AiirOperation op) {
  return wrap(unwrap(op)->getName().getIdentifier());
}

AiirBlock aiirOperationGetBlock(AiirOperation op) {
  return wrap(unwrap(op)->getBlock());
}

AiirOperation aiirOperationGetParentOperation(AiirOperation op) {
  return wrap(unwrap(op)->getParentOp());
}

intptr_t aiirOperationGetNumRegions(AiirOperation op) {
  return static_cast<intptr_t>(unwrap(op)->getNumRegions());
}

AiirRegion aiirOperationGetRegion(AiirOperation op, intptr_t pos) {
  return wrap(&unwrap(op)->getRegion(static_cast<unsigned>(pos)));
}

AiirRegion aiirOperationGetFirstRegion(AiirOperation op) {
  Operation *cppOp = unwrap(op);
  if (cppOp->getNumRegions() == 0)
    return wrap(static_cast<Region *>(nullptr));
  return wrap(&cppOp->getRegion(0));
}

AiirRegion aiirRegionGetNextInOperation(AiirRegion region) {
  Region *cppRegion = unwrap(region);
  Operation *parent = cppRegion->getParentOp();
  intptr_t next = cppRegion->getRegionNumber() + 1;
  if (parent->getNumRegions() > next)
    return wrap(&parent->getRegion(next));
  return wrap(static_cast<Region *>(nullptr));
}

AiirOperation aiirOperationGetNextInBlock(AiirOperation op) {
  return wrap(unwrap(op)->getNextNode());
}

intptr_t aiirOperationGetNumOperands(AiirOperation op) {
  return static_cast<intptr_t>(unwrap(op)->getNumOperands());
}

AiirValue aiirOperationGetOperand(AiirOperation op, intptr_t pos) {
  return wrap(unwrap(op)->getOperand(static_cast<unsigned>(pos)));
}

AiirOpOperand aiirOperationGetOpOperand(AiirOperation op, intptr_t pos) {
  return wrap(&unwrap(op)->getOpOperand(static_cast<unsigned>(pos)));
}

void aiirOperationSetOperand(AiirOperation op, intptr_t pos,
                             AiirValue newValue) {
  unwrap(op)->setOperand(static_cast<unsigned>(pos), unwrap(newValue));
}

void aiirOperationSetOperands(AiirOperation op, intptr_t nOperands,
                              AiirValue const *operands) {
  SmallVector<Value> ops;
  unwrap(op)->setOperands(unwrapList(nOperands, operands, ops));
}

intptr_t aiirOperationGetNumResults(AiirOperation op) {
  return static_cast<intptr_t>(unwrap(op)->getNumResults());
}

AiirValue aiirOperationGetResult(AiirOperation op, intptr_t pos) {
  return wrap(unwrap(op)->getResult(static_cast<unsigned>(pos)));
}

intptr_t aiirOperationGetNumSuccessors(AiirOperation op) {
  return static_cast<intptr_t>(unwrap(op)->getNumSuccessors());
}

AiirBlock aiirOperationGetSuccessor(AiirOperation op, intptr_t pos) {
  return wrap(unwrap(op)->getSuccessor(static_cast<unsigned>(pos)));
}

AIIR_CAPI_EXPORTED bool
aiirOperationHasInherentAttributeByName(AiirOperation op, AiirStringRef name) {
  std::optional<Attribute> attr = unwrap(op)->getInherentAttr(unwrap(name));
  return attr.has_value();
}

AiirAttribute aiirOperationGetInherentAttributeByName(AiirOperation op,
                                                      AiirStringRef name) {
  std::optional<Attribute> attr = unwrap(op)->getInherentAttr(unwrap(name));
  if (attr.has_value())
    return wrap(*attr);
  return {};
}

void aiirOperationSetInherentAttributeByName(AiirOperation op,
                                             AiirStringRef name,
                                             AiirAttribute attr) {
  unwrap(op)->setInherentAttr(
      StringAttr::get(unwrap(op)->getContext(), unwrap(name)), unwrap(attr));
}

intptr_t aiirOperationGetNumDiscardableAttributes(AiirOperation op) {
  return static_cast<intptr_t>(
      llvm::range_size(unwrap(op)->getDiscardableAttrs()));
}

AiirNamedAttribute aiirOperationGetDiscardableAttribute(AiirOperation op,
                                                        intptr_t pos) {
  NamedAttribute attr =
      *std::next(unwrap(op)->getDiscardableAttrs().begin(), pos);
  return AiirNamedAttribute{wrap(attr.getName()), wrap(attr.getValue())};
}

AiirAttribute aiirOperationGetDiscardableAttributeByName(AiirOperation op,
                                                         AiirStringRef name) {
  return wrap(unwrap(op)->getDiscardableAttr(unwrap(name)));
}

void aiirOperationSetDiscardableAttributeByName(AiirOperation op,
                                                AiirStringRef name,
                                                AiirAttribute attr) {
  unwrap(op)->setDiscardableAttr(unwrap(name), unwrap(attr));
}

bool aiirOperationRemoveDiscardableAttributeByName(AiirOperation op,
                                                   AiirStringRef name) {
  return !!unwrap(op)->removeDiscardableAttr(unwrap(name));
}

void aiirOperationSetSuccessor(AiirOperation op, intptr_t pos,
                               AiirBlock block) {
  unwrap(op)->setSuccessor(unwrap(block), static_cast<unsigned>(pos));
}

intptr_t aiirOperationGetNumAttributes(AiirOperation op) {
  return static_cast<intptr_t>(unwrap(op)->getAttrs().size());
}

AiirNamedAttribute aiirOperationGetAttribute(AiirOperation op, intptr_t pos) {
  NamedAttribute attr = unwrap(op)->getAttrs()[pos];
  return AiirNamedAttribute{wrap(attr.getName()), wrap(attr.getValue())};
}

AiirAttribute aiirOperationGetAttributeByName(AiirOperation op,
                                              AiirStringRef name) {
  return wrap(unwrap(op)->getAttr(unwrap(name)));
}

void aiirOperationSetAttributeByName(AiirOperation op, AiirStringRef name,
                                     AiirAttribute attr) {
  unwrap(op)->setAttr(unwrap(name), unwrap(attr));
}

bool aiirOperationRemoveAttributeByName(AiirOperation op, AiirStringRef name) {
  return !!unwrap(op)->removeAttr(unwrap(name));
}

void aiirOperationPrint(AiirOperation op, AiirStringCallback callback,
                        void *userData) {
  detail::CallbackOstream stream(callback, userData);
  unwrap(op)->print(stream);
}

void aiirOperationPrintWithFlags(AiirOperation op, AiirOpPrintingFlags flags,
                                 AiirStringCallback callback, void *userData) {
  detail::CallbackOstream stream(callback, userData);
  unwrap(op)->print(stream, *unwrap(flags));
}

void aiirOperationPrintWithState(AiirOperation op, AiirAsmState state,
                                 AiirStringCallback callback, void *userData) {
  detail::CallbackOstream stream(callback, userData);
  if (state.ptr)
    unwrap(op)->print(stream, *unwrap(state));
  else
    unwrap(op)->print(stream);
}

void aiirOperationWriteBytecode(AiirOperation op, AiirStringCallback callback,
                                void *userData) {
  detail::CallbackOstream stream(callback, userData);
  // As no desired version is set, no failure can occur.
  (void)writeBytecodeToFile(unwrap(op), stream);
}

AiirLogicalResult aiirOperationWriteBytecodeWithConfig(
    AiirOperation op, AiirBytecodeWriterConfig config,
    AiirStringCallback callback, void *userData) {
  detail::CallbackOstream stream(callback, userData);
  return wrap(writeBytecodeToFile(unwrap(op), stream, *unwrap(config)));
}

void aiirOperationDump(AiirOperation op) { return unwrap(op)->dump(); }

bool aiirOperationVerify(AiirOperation op) {
  return succeeded(verify(unwrap(op)));
}

void aiirOperationMoveAfter(AiirOperation op, AiirOperation other) {
  return unwrap(op)->moveAfter(unwrap(other));
}

void aiirOperationMoveBefore(AiirOperation op, AiirOperation other) {
  return unwrap(op)->moveBefore(unwrap(other));
}

bool aiirOperationIsBeforeInBlock(AiirOperation op, AiirOperation other) {
  return unwrap(op)->isBeforeInBlock(unwrap(other));
}

static aiir::WalkResult unwrap(AiirWalkResult result) {
  switch (result) {
  case AiirWalkResultAdvance:
    return aiir::WalkResult::advance();

  case AiirWalkResultInterrupt:
    return aiir::WalkResult::interrupt();

  case AiirWalkResultSkip:
    return aiir::WalkResult::skip();
  }
  llvm_unreachable("unknown result in WalkResult::unwrap");
}

void aiirOperationWalk(AiirOperation op, AiirOperationWalkCallback callback,
                       void *userData, AiirWalkOrder walkOrder) {
  switch (walkOrder) {

  case AiirWalkPreOrder:
    unwrap(op)->walk<aiir::WalkOrder::PreOrder>(
        [callback, userData](Operation *op) {
          return unwrap(callback(wrap(op), userData));
        });
    break;
  case AiirWalkPostOrder:
    unwrap(op)->walk<aiir::WalkOrder::PostOrder>(
        [callback, userData](Operation *op) {
          return unwrap(callback(wrap(op), userData));
        });
  }
}

void aiirOperationReplaceUsesOfWith(AiirOperation op, AiirValue oldValue,
                                    AiirValue newValue) {
  unwrap(op)->replaceUsesOfWith(unwrap(oldValue), unwrap(newValue));
}

//===----------------------------------------------------------------------===//
// Region API.
//===----------------------------------------------------------------------===//

AiirRegion aiirRegionCreate() { return wrap(new Region); }

bool aiirRegionEqual(AiirRegion region, AiirRegion other) {
  return unwrap(region) == unwrap(other);
}

AiirBlock aiirRegionGetFirstBlock(AiirRegion region) {
  Region *cppRegion = unwrap(region);
  if (cppRegion->empty())
    return wrap(static_cast<Block *>(nullptr));
  return wrap(&cppRegion->front());
}

void aiirRegionAppendOwnedBlock(AiirRegion region, AiirBlock block) {
  unwrap(region)->push_back(unwrap(block));
}

void aiirRegionInsertOwnedBlock(AiirRegion region, intptr_t pos,
                                AiirBlock block) {
  auto &blockList = unwrap(region)->getBlocks();
  blockList.insert(std::next(blockList.begin(), pos), unwrap(block));
}

void aiirRegionInsertOwnedBlockAfter(AiirRegion region, AiirBlock reference,
                                     AiirBlock block) {
  Region *cppRegion = unwrap(region);
  if (aiirBlockIsNull(reference)) {
    cppRegion->getBlocks().insert(cppRegion->begin(), unwrap(block));
    return;
  }

  assert(unwrap(reference)->getParent() == unwrap(region) &&
         "expected reference block to belong to the region");
  cppRegion->getBlocks().insertAfter(Region::iterator(unwrap(reference)),
                                     unwrap(block));
}

void aiirRegionInsertOwnedBlockBefore(AiirRegion region, AiirBlock reference,
                                      AiirBlock block) {
  if (aiirBlockIsNull(reference))
    return aiirRegionAppendOwnedBlock(region, block);

  assert(unwrap(reference)->getParent() == unwrap(region) &&
         "expected reference block to belong to the region");
  unwrap(region)->getBlocks().insert(Region::iterator(unwrap(reference)),
                                     unwrap(block));
}

void aiirRegionDestroy(AiirRegion region) {
  delete static_cast<Region *>(region.ptr);
}

void aiirRegionTakeBody(AiirRegion target, AiirRegion source) {
  unwrap(target)->takeBody(*unwrap(source));
}

//===----------------------------------------------------------------------===//
// Block API.
//===----------------------------------------------------------------------===//

AiirBlock aiirBlockCreate(intptr_t nArgs, AiirType const *args,
                          AiirLocation const *locs) {
  Block *b = new Block;
  for (intptr_t i = 0; i < nArgs; ++i)
    b->addArgument(unwrap(args[i]), unwrap(locs[i]));
  return wrap(b);
}

bool aiirBlockEqual(AiirBlock block, AiirBlock other) {
  return unwrap(block) == unwrap(other);
}

AiirOperation aiirBlockGetParentOperation(AiirBlock block) {
  return wrap(unwrap(block)->getParentOp());
}

AiirRegion aiirBlockGetParentRegion(AiirBlock block) {
  return wrap(unwrap(block)->getParent());
}

AiirBlock aiirBlockGetNextInRegion(AiirBlock block) {
  return wrap(unwrap(block)->getNextNode());
}

AiirOperation aiirBlockGetFirstOperation(AiirBlock block) {
  Block *cppBlock = unwrap(block);
  if (cppBlock->empty())
    return wrap(static_cast<Operation *>(nullptr));
  return wrap(&cppBlock->front());
}

AiirOperation aiirBlockGetTerminator(AiirBlock block) {
  Block *cppBlock = unwrap(block);
  if (cppBlock->empty())
    return wrap(static_cast<Operation *>(nullptr));
  Operation &back = cppBlock->back();
  if (!back.hasTrait<OpTrait::IsTerminator>())
    return wrap(static_cast<Operation *>(nullptr));
  return wrap(&back);
}

void aiirBlockAppendOwnedOperation(AiirBlock block, AiirOperation operation) {
  unwrap(block)->push_back(unwrap(operation));
}

void aiirBlockInsertOwnedOperation(AiirBlock block, intptr_t pos,
                                   AiirOperation operation) {
  auto &opList = unwrap(block)->getOperations();
  opList.insert(std::next(opList.begin(), pos), unwrap(operation));
}

void aiirBlockInsertOwnedOperationAfter(AiirBlock block,
                                        AiirOperation reference,
                                        AiirOperation operation) {
  Block *cppBlock = unwrap(block);
  if (aiirOperationIsNull(reference)) {
    cppBlock->getOperations().insert(cppBlock->begin(), unwrap(operation));
    return;
  }

  assert(unwrap(reference)->getBlock() == unwrap(block) &&
         "expected reference operation to belong to the block");
  cppBlock->getOperations().insertAfter(Block::iterator(unwrap(reference)),
                                        unwrap(operation));
}

void aiirBlockInsertOwnedOperationBefore(AiirBlock block,
                                         AiirOperation reference,
                                         AiirOperation operation) {
  if (aiirOperationIsNull(reference))
    return aiirBlockAppendOwnedOperation(block, operation);

  assert(unwrap(reference)->getBlock() == unwrap(block) &&
         "expected reference operation to belong to the block");
  unwrap(block)->getOperations().insert(Block::iterator(unwrap(reference)),
                                        unwrap(operation));
}

void aiirBlockDestroy(AiirBlock block) { delete unwrap(block); }

void aiirBlockDetach(AiirBlock block) {
  Block *b = unwrap(block);
  b->getParent()->getBlocks().remove(b);
}

intptr_t aiirBlockGetNumArguments(AiirBlock block) {
  return static_cast<intptr_t>(unwrap(block)->getNumArguments());
}

AiirValue aiirBlockAddArgument(AiirBlock block, AiirType type,
                               AiirLocation loc) {
  return wrap(unwrap(block)->addArgument(unwrap(type), unwrap(loc)));
}

void aiirBlockEraseArgument(AiirBlock block, unsigned index) {
  return unwrap(block)->eraseArgument(index);
}

AiirValue aiirBlockInsertArgument(AiirBlock block, intptr_t pos, AiirType type,
                                  AiirLocation loc) {
  return wrap(unwrap(block)->insertArgument(pos, unwrap(type), unwrap(loc)));
}

AiirValue aiirBlockGetArgument(AiirBlock block, intptr_t pos) {
  return wrap(unwrap(block)->getArgument(static_cast<unsigned>(pos)));
}

void aiirBlockPrint(AiirBlock block, AiirStringCallback callback,
                    void *userData) {
  detail::CallbackOstream stream(callback, userData);
  unwrap(block)->print(stream);
}

intptr_t aiirBlockGetNumSuccessors(AiirBlock block) {
  return static_cast<intptr_t>(unwrap(block)->getNumSuccessors());
}

AiirBlock aiirBlockGetSuccessor(AiirBlock block, intptr_t pos) {
  return wrap(unwrap(block)->getSuccessor(static_cast<unsigned>(pos)));
}

intptr_t aiirBlockGetNumPredecessors(AiirBlock block) {
  Block *b = unwrap(block);
  return static_cast<intptr_t>(std::distance(b->pred_begin(), b->pred_end()));
}

AiirBlock aiirBlockGetPredecessor(AiirBlock block, intptr_t pos) {
  Block *b = unwrap(block);
  Block::pred_iterator it = b->pred_begin();
  std::advance(it, pos);
  return wrap(*it);
}

//===----------------------------------------------------------------------===//
// Value API.
//===----------------------------------------------------------------------===//

bool aiirValueEqual(AiirValue value1, AiirValue value2) {
  return unwrap(value1) == unwrap(value2);
}

bool aiirValueIsABlockArgument(AiirValue value) {
  return llvm::isa<BlockArgument>(unwrap(value));
}

bool aiirValueIsAOpResult(AiirValue value) {
  return llvm::isa<OpResult>(unwrap(value));
}

AiirBlock aiirBlockArgumentGetOwner(AiirValue value) {
  return wrap(llvm::dyn_cast<BlockArgument>(unwrap(value)).getOwner());
}

intptr_t aiirBlockArgumentGetArgNumber(AiirValue value) {
  return static_cast<intptr_t>(
      llvm::dyn_cast<BlockArgument>(unwrap(value)).getArgNumber());
}

void aiirBlockArgumentSetType(AiirValue value, AiirType type) {
  if (auto blockArg = llvm::dyn_cast<BlockArgument>(unwrap(value)))
    blockArg.setType(unwrap(type));
}

void aiirBlockArgumentSetLocation(AiirValue value, AiirLocation loc) {
  if (auto blockArg = llvm::dyn_cast<BlockArgument>(unwrap(value)))
    blockArg.setLoc(unwrap(loc));
}

AiirOperation aiirOpResultGetOwner(AiirValue value) {
  return wrap(llvm::dyn_cast<OpResult>(unwrap(value)).getOwner());
}

intptr_t aiirOpResultGetResultNumber(AiirValue value) {
  return static_cast<intptr_t>(
      llvm::dyn_cast<OpResult>(unwrap(value)).getResultNumber());
}

AiirType aiirValueGetType(AiirValue value) {
  return wrap(unwrap(value).getType());
}

void aiirValueSetType(AiirValue value, AiirType type) {
  unwrap(value).setType(unwrap(type));
}

void aiirValueDump(AiirValue value) { unwrap(value).dump(); }

void aiirValuePrint(AiirValue value, AiirStringCallback callback,
                    void *userData) {
  detail::CallbackOstream stream(callback, userData);
  unwrap(value).print(stream);
}

void aiirValuePrintAsOperand(AiirValue value, AiirAsmState state,
                             AiirStringCallback callback, void *userData) {
  detail::CallbackOstream stream(callback, userData);
  Value cppValue = unwrap(value);
  cppValue.printAsOperand(stream, *unwrap(state));
}

AiirOpOperand aiirValueGetFirstUse(AiirValue value) {
  Value cppValue = unwrap(value);
  if (cppValue.use_empty())
    return {};

  OpOperand *opOperand = cppValue.use_begin().getOperand();

  return wrap(opOperand);
}

void aiirValueReplaceAllUsesOfWith(AiirValue oldValue, AiirValue newValue) {
  unwrap(oldValue).replaceAllUsesWith(unwrap(newValue));
}

void aiirValueReplaceAllUsesExcept(AiirValue oldValue, AiirValue newValue,
                                   intptr_t numExceptions,
                                   AiirOperation *exceptions) {
  Value oldValueCpp = unwrap(oldValue);
  Value newValueCpp = unwrap(newValue);

  llvm::SmallPtrSet<aiir::Operation *, 4> exceptionSet;
  for (intptr_t i = 0; i < numExceptions; ++i) {
    exceptionSet.insert(unwrap(exceptions[i]));
  }

  oldValueCpp.replaceAllUsesExcept(newValueCpp, exceptionSet);
}

AiirLocation aiirValueGetLocation(AiirValue v) {
  return wrap(unwrap(v).getLoc());
}

AiirContext aiirValueGetContext(AiirValue v) {
  return wrap(unwrap(v).getContext());
}

//===----------------------------------------------------------------------===//
// OpOperand API.
//===----------------------------------------------------------------------===//

bool aiirOpOperandIsNull(AiirOpOperand opOperand) { return !opOperand.ptr; }

AiirOperation aiirOpOperandGetOwner(AiirOpOperand opOperand) {
  return wrap(unwrap(opOperand)->getOwner());
}

AiirValue aiirOpOperandGetValue(AiirOpOperand opOperand) {
  return wrap(unwrap(opOperand)->get());
}

unsigned aiirOpOperandGetOperandNumber(AiirOpOperand opOperand) {
  return unwrap(opOperand)->getOperandNumber();
}

AiirOpOperand aiirOpOperandGetNextUse(AiirOpOperand opOperand) {
  if (aiirOpOperandIsNull(opOperand))
    return {};

  OpOperand *nextOpOperand = static_cast<OpOperand *>(
      unwrap(opOperand)->getNextOperandUsingThisValue());

  if (!nextOpOperand)
    return {};

  return wrap(nextOpOperand);
}

//===----------------------------------------------------------------------===//
// Type API.
//===----------------------------------------------------------------------===//

AiirType aiirTypeParseGet(AiirContext context, AiirStringRef type) {
  return wrap(aiir::parseType(unwrap(type), unwrap(context)));
}

AiirContext aiirTypeGetContext(AiirType type) {
  return wrap(unwrap(type).getContext());
}

AiirTypeID aiirTypeGetTypeID(AiirType type) {
  return wrap(unwrap(type).getTypeID());
}

AiirDialect aiirTypeGetDialect(AiirType type) {
  return wrap(&unwrap(type).getDialect());
}

bool aiirTypeEqual(AiirType t1, AiirType t2) {
  return unwrap(t1) == unwrap(t2);
}

void aiirTypePrint(AiirType type, AiirStringCallback callback, void *userData) {
  detail::CallbackOstream stream(callback, userData);
  unwrap(type).print(stream);
}

void aiirTypeDump(AiirType type) { unwrap(type).dump(); }

//===----------------------------------------------------------------------===//
// Attribute API.
//===----------------------------------------------------------------------===//

AiirAttribute aiirAttributeParseGet(AiirContext context, AiirStringRef attr) {
  return wrap(aiir::parseAttribute(unwrap(attr), unwrap(context)));
}

AiirContext aiirAttributeGetContext(AiirAttribute attribute) {
  return wrap(unwrap(attribute).getContext());
}

AiirType aiirAttributeGetType(AiirAttribute attribute) {
  Attribute attr = unwrap(attribute);
  if (auto typedAttr = llvm::dyn_cast<TypedAttr>(attr))
    return wrap(typedAttr.getType());
  return wrap(NoneType::get(attr.getContext()));
}

AiirTypeID aiirAttributeGetTypeID(AiirAttribute attr) {
  return wrap(unwrap(attr).getTypeID());
}

AiirDialect aiirAttributeGetDialect(AiirAttribute attr) {
  return wrap(&unwrap(attr).getDialect());
}

bool aiirAttributeEqual(AiirAttribute a1, AiirAttribute a2) {
  return unwrap(a1) == unwrap(a2);
}

void aiirAttributePrint(AiirAttribute attr, AiirStringCallback callback,
                        void *userData) {
  detail::CallbackOstream stream(callback, userData);
  unwrap(attr).print(stream);
}

void aiirAttributeDump(AiirAttribute attr) { unwrap(attr).dump(); }

AiirNamedAttribute aiirNamedAttributeGet(AiirIdentifier name,
                                         AiirAttribute attr) {
  return AiirNamedAttribute{name, attr};
}

//===----------------------------------------------------------------------===//
// Identifier API.
//===----------------------------------------------------------------------===//

AiirIdentifier aiirIdentifierGet(AiirContext context, AiirStringRef str) {
  return wrap(StringAttr::get(unwrap(context), unwrap(str)));
}

AiirContext aiirIdentifierGetContext(AiirIdentifier ident) {
  return wrap(unwrap(ident).getContext());
}

bool aiirIdentifierEqual(AiirIdentifier ident, AiirIdentifier other) {
  return unwrap(ident) == unwrap(other);
}

AiirStringRef aiirIdentifierStr(AiirIdentifier ident) {
  return wrap(unwrap(ident).strref());
}

//===----------------------------------------------------------------------===//
// Symbol and SymbolTable API.
//===----------------------------------------------------------------------===//

AiirStringRef aiirSymbolTableGetSymbolAttributeName() {
  return wrap(SymbolTable::getSymbolAttrName());
}

AiirStringRef aiirSymbolTableGetVisibilityAttributeName() {
  return wrap(SymbolTable::getVisibilityAttrName());
}

AiirSymbolTable aiirSymbolTableCreate(AiirOperation operation) {
  if (!unwrap(operation)->hasTrait<OpTrait::SymbolTable>())
    return wrap(static_cast<SymbolTable *>(nullptr));
  return wrap(new SymbolTable(unwrap(operation)));
}

void aiirSymbolTableDestroy(AiirSymbolTable symbolTable) {
  delete unwrap(symbolTable);
}

AiirOperation aiirSymbolTableLookup(AiirSymbolTable symbolTable,
                                    AiirStringRef name) {
  return wrap(unwrap(symbolTable)->lookup(StringRef(name.data, name.length)));
}

AiirAttribute aiirSymbolTableInsert(AiirSymbolTable symbolTable,
                                    AiirOperation operation) {
  return wrap((Attribute)unwrap(symbolTable)->insert(unwrap(operation)));
}

void aiirSymbolTableErase(AiirSymbolTable symbolTable,
                          AiirOperation operation) {
  unwrap(symbolTable)->erase(unwrap(operation));
}

AiirLogicalResult aiirSymbolTableReplaceAllSymbolUses(AiirStringRef oldSymbol,
                                                      AiirStringRef newSymbol,
                                                      AiirOperation from) {
  auto *cppFrom = unwrap(from);
  auto *context = cppFrom->getContext();
  auto oldSymbolAttr = StringAttr::get(context, unwrap(oldSymbol));
  auto newSymbolAttr = StringAttr::get(context, unwrap(newSymbol));
  return wrap(SymbolTable::replaceAllSymbolUses(oldSymbolAttr, newSymbolAttr,
                                                unwrap(from)));
}

void aiirSymbolTableWalkSymbolTables(AiirOperation from, bool allSymUsesVisible,
                                     void (*callback)(AiirOperation, bool,
                                                      void *userData),
                                     void *userData) {
  SymbolTable::walkSymbolTables(unwrap(from), allSymUsesVisible,
                                [&](Operation *foundOpCpp, bool isVisible) {
                                  callback(wrap(foundOpCpp), isVisible,
                                           userData);
                                });
}
