//===-- aiir-c/Interop.h - Constants for Python/C-API interop -----*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares constants and helpers necessary for C-level
// interop with the AIIR Python extension module. Since the Python bindings
// are a thin wrapper around the AIIR C-API, a further C-API is not provided
// specifically for the Python extension. Instead, simple facilities are
// provided for translating between Python types and corresponding AIIR C-API
// types.
//
// This header is standalone, requiring nothing beyond normal linking against
// the Python implementation.
//===----------------------------------------------------------------------===//

#ifndef AIIR_C_BINDINGS_PYTHON_INTEROP_H
#define AIIR_C_BINDINGS_PYTHON_INTEROP_H

// We *should*, in theory, include Python.h here in order to import the correct
// definitions for what we need below, however, importing Python.h directly on
// Windows results in the enforcement of either pythonX.lib or pythonX_d.lib
// depending on the build flavor. Instead, we rely on the fact that this file
// (Interop.h) is always included AFTER pybind11 and will therefore have access
// to the definitions from Python.h in addition to having a workaround applied
// through the pybind11 headers that allows us to control which python library
// is used.
#if !defined(_MSC_VER)
#include <Python.h>
#endif

#include "aiir-c/AffineExpr.h"
#include "aiir-c/AffineMap.h"
#include "aiir-c/ExecutionEngine.h"
#include "aiir-c/IR.h"
#include "aiir-c/IntegerSet.h"
#include "aiir-c/Pass.h"
#include "aiir-c/Rewrite.h"

// The 'aiir' Python package is relocatable and supports co-existing in multiple
// projects. Each project must define its outer package prefix with this define
// in order to provide proper isolation and local name resolution.
// The default is for the upstream "import aiir" package layout.
// Note that this prefix is internally stringified, allowing it to be passed
// unquoted on the compiler command line without shell quote escaping issues.
#ifndef AIIR_PYTHON_PACKAGE_PREFIX
#define AIIR_PYTHON_PACKAGE_PREFIX aiir.
#endif

// Makes a fully-qualified name relative to the AIIR python package.
#define AIIR_PYTHON_STRINGIZE(s) #s
#define AIIR_PYTHON_STRINGIZE_ARG(arg) AIIR_PYTHON_STRINGIZE(arg)
#define MAKE_AIIR_PYTHON_QUALNAME(local)                                       \
  AIIR_PYTHON_STRINGIZE_ARG(AIIR_PYTHON_PACKAGE_PREFIX) local

#define AIIR_PYTHON_CAPSULE_AFFINE_EXPR                                        \
  MAKE_AIIR_PYTHON_QUALNAME("ir.AffineExpr._CAPIPtr")
#define AIIR_PYTHON_CAPSULE_AFFINE_MAP                                         \
  MAKE_AIIR_PYTHON_QUALNAME("ir.AffineMap._CAPIPtr")
#define AIIR_PYTHON_CAPSULE_ATTRIBUTE                                          \
  MAKE_AIIR_PYTHON_QUALNAME("ir.Attribute._CAPIPtr")
#define AIIR_PYTHON_CAPSULE_BLOCK MAKE_AIIR_PYTHON_QUALNAME("ir.Block._CAPIPtr")
#define AIIR_PYTHON_CAPSULE_CONTEXT                                            \
  MAKE_AIIR_PYTHON_QUALNAME("ir.Context._CAPIPtr")
#define AIIR_PYTHON_CAPSULE_DIALECT_REGISTRY                                   \
  MAKE_AIIR_PYTHON_QUALNAME("ir.DialectRegistry._CAPIPtr")
#define AIIR_PYTHON_CAPSULE_EXECUTION_ENGINE                                   \
  MAKE_AIIR_PYTHON_QUALNAME("execution_engine.ExecutionEngine._CAPIPtr")
#define AIIR_PYTHON_CAPSULE_INTEGER_SET                                        \
  MAKE_AIIR_PYTHON_QUALNAME("ir.IntegerSet._CAPIPtr")
#define AIIR_PYTHON_CAPSULE_LOCATION                                           \
  MAKE_AIIR_PYTHON_QUALNAME("ir.Location._CAPIPtr")
#define AIIR_PYTHON_CAPSULE_MODULE                                             \
  MAKE_AIIR_PYTHON_QUALNAME("ir.Module._CAPIPtr")
#define AIIR_PYTHON_CAPSULE_OPERATION                                          \
  MAKE_AIIR_PYTHON_QUALNAME("ir.Operation._CAPIPtr")
#define AIIR_PYTHON_CAPSULE_TYPE MAKE_AIIR_PYTHON_QUALNAME("ir.Type._CAPIPtr")
#define AIIR_PYTHON_CAPSULE_PASS_MANAGER                                       \
  MAKE_AIIR_PYTHON_QUALNAME("passmanager.PassManager._CAPIPtr")
#define AIIR_PYTHON_CAPSULE_VALUE MAKE_AIIR_PYTHON_QUALNAME("ir.Value._CAPIPtr")
#define AIIR_PYTHON_CAPSULE_TYPEID                                             \
  MAKE_AIIR_PYTHON_QUALNAME("ir.TypeID._CAPIPtr")

/** Attribute on AIIR Python objects that expose their C-API pointer.
 * This will be a type-specific capsule created as per one of the helpers
 * below.
 *
 * Ownership is not transferred by acquiring a capsule in this way: the
 * validity of the pointer wrapped by the capsule will be bounded by the
 * lifetime of the Python object that produced it. Only the name and pointer
 * of the capsule are set. The caller is free to set a destructor and context
 * as needed to manage anything further. */
#define AIIR_PYTHON_CAPI_PTR_ATTR "_CAPIPtr"

/** Attribute on AIIR Python objects that exposes a factory function for
 * constructing the corresponding Python object from a type-specific
 * capsule wrapping the C-API pointer. The signature of the function is:
 *   def _CAPICreate(capsule) -> object
 * Calling such a function implies a transfer of ownership of the object the
 * capsule wraps: after such a call, the capsule should be considered invalid,
 * and its wrapped pointer must not be destroyed.
 *
 * Only a very small number of Python objects can be created in such a fashion
 * (i.e. top-level types such as Context where the lifetime can be cleanly
 * delineated). */
#define AIIR_PYTHON_CAPI_FACTORY_ATTR "_CAPICreate"

/** Attribute on AIIR Python objects that expose a function for downcasting the
 * corresponding Python object to a subclass if the object is in fact a subclass
 * (Concrete or aiir_type_subclass) of ir.Type. The signature of the function
 * is: def maybe_downcast(self) -> object where the resulting object will
 * (possibly) be an instance of the subclass.
 */
#define AIIR_PYTHON_MAYBE_DOWNCAST_ATTR "maybe_downcast"

/** Attribute on main C extension module (_aiir) that corresponds to the
 * type caster registration binding. The signature of the function is:
 *   def register_type_caster(AiirTypeID aiirTypeID, *, bool replace)
 * which then takes a typeCaster (register_type_caster is meant to be used as a
 * decorator from python), and where replace indicates the typeCaster should
 * replace any existing registered type casters (such as those for upstream
 * ConcreteTypes). The interface of the typeCaster is: def type_caster(ir.Type)
 * -> SubClassTypeT where SubClassTypeT indicates the result should be a
 * subclass (inherit from) ir.Type.
 */
#define AIIR_PYTHON_CAPI_TYPE_CASTER_REGISTER_ATTR "register_type_caster"

/** Attribute on main C extension module (_aiir) that corresponds to the
 * value caster registration binding. The signature of the function is:
 *   def register_value_caster(AiirTypeID aiirTypeID, *, bool replace)
 * which then takes a valueCaster (register_value_caster is meant to be used as
 * a decorator, from python), and where replace indicates the valueCaster should
 * replace any existing registered value casters. The interface of the
 * valueCaster is: def value_caster(ir.Value) -> SubClassValueT where
 * SubClassValueT indicates the result should be a subclass (inherit from)
 * ir.Value.
 */
#define AIIR_PYTHON_CAPI_VALUE_CASTER_REGISTER_ATTR "register_value_caster"

/// Gets a void* from a wrapped struct. Needed because const cast is different
/// between C/C++.
#ifdef __cplusplus
#define AIIR_PYTHON_GET_WRAPPED_POINTER(object)                                \
  (const_cast<void *>((object).ptr))
#else
#define AIIR_PYTHON_GET_WRAPPED_POINTER(object) (void *)(object.ptr)
#endif

#ifdef __cplusplus
extern "C" {
#endif

/** Creates a capsule object encapsulating the raw C-API AiirAffineExpr. The
 * returned capsule does not extend or affect ownership of any Python objects
 * that reference the expression in any way.
 */
static inline PyObject *aiirPythonAffineExprToCapsule(AiirAffineExpr expr) {
  return PyCapsule_New(AIIR_PYTHON_GET_WRAPPED_POINTER(expr),
                       AIIR_PYTHON_CAPSULE_AFFINE_EXPR, NULL);
}

/** Extracts an AiirAffineExpr from a capsule as produced from
 * aiirPythonAffineExprToCapsule. If the capsule is not of the right type, then
 * a null expression is returned (as checked via aiirAffineExprIsNull). In such
 * a case, the Python APIs will have already set an error. */
static inline AiirAffineExpr aiirPythonCapsuleToAffineExpr(PyObject *capsule) {
  void *ptr = PyCapsule_GetPointer(capsule, AIIR_PYTHON_CAPSULE_AFFINE_EXPR);
  AiirAffineExpr expr = {ptr};
  return expr;
}

/** Creates a capsule object encapsulating the raw C-API AiirAttribute.
 * The returned capsule does not extend or affect ownership of any Python
 * objects that reference the attribute in any way.
 */
static inline PyObject *aiirPythonAttributeToCapsule(AiirAttribute attribute) {
  return PyCapsule_New(AIIR_PYTHON_GET_WRAPPED_POINTER(attribute),
                       AIIR_PYTHON_CAPSULE_ATTRIBUTE, NULL);
}

/** Extracts an AiirAttribute from a capsule as produced from
 * aiirPythonAttributeToCapsule. If the capsule is not of the right type, then
 * a null attribute is returned (as checked via aiirAttributeIsNull). In such a
 * case, the Python APIs will have already set an error. */
static inline AiirAttribute aiirPythonCapsuleToAttribute(PyObject *capsule) {
  void *ptr = PyCapsule_GetPointer(capsule, AIIR_PYTHON_CAPSULE_ATTRIBUTE);
  AiirAttribute attr = {ptr};
  return attr;
}

/** Creates a capsule object encapsulating the raw C-API AiirBlock.
 * The returned capsule does not extend or affect ownership of any Python
 * objects that reference the module in any way. */
static inline PyObject *aiirPythonBlockToCapsule(AiirBlock block) {
  return PyCapsule_New(AIIR_PYTHON_GET_WRAPPED_POINTER(block),
                       AIIR_PYTHON_CAPSULE_BLOCK, NULL);
}

/** Extracts an AiirBlock from a capsule as produced from
 * aiirPythonBlockToCapsule. If the capsule is not of the right type, then
 * a null pass manager is returned (as checked via aiirBlockIsNull). */
static inline AiirBlock aiirPythonCapsuleToBlock(PyObject *capsule) {
  void *ptr = PyCapsule_GetPointer(capsule, AIIR_PYTHON_CAPSULE_BLOCK);
  AiirBlock block = {ptr};
  return block;
}

/** Creates a capsule object encapsulating the raw C-API AiirContext.
 * The returned capsule does not extend or affect ownership of any Python
 * objects that reference the context in any way.
 */
static inline PyObject *aiirPythonContextToCapsule(AiirContext context) {
  return PyCapsule_New(context.ptr, AIIR_PYTHON_CAPSULE_CONTEXT, NULL);
}

/** Extracts a AiirContext from a capsule as produced from
 * aiirPythonContextToCapsule. If the capsule is not of the right type, then
 * a null context is returned (as checked via aiirContextIsNull). In such a
 * case, the Python APIs will have already set an error. */
static inline AiirContext aiirPythonCapsuleToContext(PyObject *capsule) {
  void *ptr = PyCapsule_GetPointer(capsule, AIIR_PYTHON_CAPSULE_CONTEXT);
  AiirContext context = {ptr};
  return context;
}

/** Creates a capsule object encapsulating the raw C-API AiirDialectRegistry.
 * The returned capsule does not extend or affect ownership of any Python
 * objects that reference the context in any way.
 */
static inline PyObject *
aiirPythonDialectRegistryToCapsule(AiirDialectRegistry registry) {
  return PyCapsule_New(registry.ptr, AIIR_PYTHON_CAPSULE_DIALECT_REGISTRY,
                       NULL);
}

/** Extracts an AiirDialectRegistry from a capsule as produced from
 * aiirPythonDialectRegistryToCapsule. If the capsule is not of the right type,
 * then a null context is returned (as checked via aiirContextIsNull). In such a
 * case, the Python APIs will have already set an error. */
static inline AiirDialectRegistry
aiirPythonCapsuleToDialectRegistry(PyObject *capsule) {
  void *ptr =
      PyCapsule_GetPointer(capsule, AIIR_PYTHON_CAPSULE_DIALECT_REGISTRY);
  AiirDialectRegistry registry = {ptr};
  return registry;
}

/** Creates a capsule object encapsulating the raw C-API AiirLocation.
 * The returned capsule does not extend or affect ownership of any Python
 * objects that reference the location in any way. */
static inline PyObject *aiirPythonLocationToCapsule(AiirLocation loc) {
  return PyCapsule_New(AIIR_PYTHON_GET_WRAPPED_POINTER(loc),
                       AIIR_PYTHON_CAPSULE_LOCATION, NULL);
}

/** Extracts an AiirLocation from a capsule as produced from
 * aiirPythonLocationToCapsule. If the capsule is not of the right type, then
 * a null module is returned (as checked via aiirLocationIsNull). In such a
 * case, the Python APIs will have already set an error. */
static inline AiirLocation aiirPythonCapsuleToLocation(PyObject *capsule) {
  void *ptr = PyCapsule_GetPointer(capsule, AIIR_PYTHON_CAPSULE_LOCATION);
  AiirLocation loc = {ptr};
  return loc;
}

/** Creates a capsule object encapsulating the raw C-API AiirModule.
 * The returned capsule does not extend or affect ownership of any Python
 * objects that reference the module in any way. */
static inline PyObject *aiirPythonModuleToCapsule(AiirModule module) {
  return PyCapsule_New(AIIR_PYTHON_GET_WRAPPED_POINTER(module),
                       AIIR_PYTHON_CAPSULE_MODULE, NULL);
}

/** Extracts an AiirModule from a capsule as produced from
 * aiirPythonModuleToCapsule. If the capsule is not of the right type, then
 * a null module is returned (as checked via aiirModuleIsNull). In such a
 * case, the Python APIs will have already set an error. */
static inline AiirModule aiirPythonCapsuleToModule(PyObject *capsule) {
  void *ptr = PyCapsule_GetPointer(capsule, AIIR_PYTHON_CAPSULE_MODULE);
  AiirModule module = {ptr};
  return module;
}

/** Creates a capsule object encapsulating the raw C-API
 * AiirFrozenRewritePatternSet.
 * The returned capsule does not extend or affect ownership of any Python
 * objects that reference the module in any way. */
static inline PyObject *
aiirPythonFrozenRewritePatternSetToCapsule(AiirFrozenRewritePatternSet pm) {
  return PyCapsule_New(AIIR_PYTHON_GET_WRAPPED_POINTER(pm),
                       AIIR_PYTHON_CAPSULE_PASS_MANAGER, NULL);
}

/** Extracts an AiirFrozenRewritePatternSet from a capsule as produced from
 * aiirPythonFrozenRewritePatternSetToCapsule. If the capsule is not of the
 * right type, then a null module is returned. */
static inline AiirFrozenRewritePatternSet
aiirPythonCapsuleToFrozenRewritePatternSet(PyObject *capsule) {
  void *ptr = PyCapsule_GetPointer(capsule, AIIR_PYTHON_CAPSULE_PASS_MANAGER);
  AiirFrozenRewritePatternSet pm = {ptr};
  return pm;
}

/** Creates a capsule object encapsulating the raw C-API AiirPassManager.
 * The returned capsule does not extend or affect ownership of any Python
 * objects that reference the module in any way. */
static inline PyObject *aiirPythonPassManagerToCapsule(AiirPassManager pm) {
  return PyCapsule_New(AIIR_PYTHON_GET_WRAPPED_POINTER(pm),
                       AIIR_PYTHON_CAPSULE_PASS_MANAGER, NULL);
}

/** Extracts an AiirPassManager from a capsule as produced from
 * aiirPythonPassManagerToCapsule. If the capsule is not of the right type, then
 * a null pass manager is returned (as checked via aiirPassManagerIsNull). */
static inline AiirPassManager
aiirPythonCapsuleToPassManager(PyObject *capsule) {
  void *ptr = PyCapsule_GetPointer(capsule, AIIR_PYTHON_CAPSULE_PASS_MANAGER);
  AiirPassManager pm = {ptr};
  return pm;
}

/** Creates a capsule object encapsulating the raw C-API AiirOperation.
 * The returned capsule does not extend or affect ownership of any Python
 * objects that reference the operation in any way.
 */
static inline PyObject *aiirPythonOperationToCapsule(AiirOperation operation) {
  return PyCapsule_New(operation.ptr, AIIR_PYTHON_CAPSULE_OPERATION, NULL);
}

/** Extracts an AiirOperations from a capsule as produced from
 * aiirPythonOperationToCapsule. If the capsule is not of the right type, then
 * a null type is returned (as checked via aiirOperationIsNull). In such a
 * case, the Python APIs will have already set an error. */
static inline AiirOperation aiirPythonCapsuleToOperation(PyObject *capsule) {
  void *ptr = PyCapsule_GetPointer(capsule, AIIR_PYTHON_CAPSULE_OPERATION);
  AiirOperation op = {ptr};
  return op;
}

/** Creates a capsule object encapsulating the raw C-API AiirTypeID.
 * The returned capsule does not extend or affect ownership of any Python
 * objects that reference the type in any way.
 */
static inline PyObject *aiirPythonTypeIDToCapsule(AiirTypeID typeID) {
  return PyCapsule_New(AIIR_PYTHON_GET_WRAPPED_POINTER(typeID),
                       AIIR_PYTHON_CAPSULE_TYPEID, NULL);
}

/** Extracts an AiirTypeID from a capsule as produced from
 * aiirPythonTypeIDToCapsule. If the capsule is not of the right type, then
 * a null type is returned (as checked via aiirTypeIDIsNull). In such a
 * case, the Python APIs will have already set an error. */
static inline AiirTypeID aiirPythonCapsuleToTypeID(PyObject *capsule) {
  void *ptr = PyCapsule_GetPointer(capsule, AIIR_PYTHON_CAPSULE_TYPEID);
  AiirTypeID typeID = {ptr};
  return typeID;
}

/** Creates a capsule object encapsulating the raw C-API AiirType.
 * The returned capsule does not extend or affect ownership of any Python
 * objects that reference the type in any way.
 */
static inline PyObject *aiirPythonTypeToCapsule(AiirType type) {
  return PyCapsule_New(AIIR_PYTHON_GET_WRAPPED_POINTER(type),
                       AIIR_PYTHON_CAPSULE_TYPE, NULL);
}

/** Extracts an AiirType from a capsule as produced from
 * aiirPythonTypeToCapsule. If the capsule is not of the right type, then
 * a null type is returned (as checked via aiirTypeIsNull). In such a
 * case, the Python APIs will have already set an error. */
static inline AiirType aiirPythonCapsuleToType(PyObject *capsule) {
  void *ptr = PyCapsule_GetPointer(capsule, AIIR_PYTHON_CAPSULE_TYPE);
  AiirType type = {ptr};
  return type;
}

/** Creates a capsule object encapsulating the raw C-API AiirAffineMap.
 * The returned capsule does not extend or affect ownership of any Python
 * objects that reference the type in any way.
 */
static inline PyObject *aiirPythonAffineMapToCapsule(AiirAffineMap affineMap) {
  return PyCapsule_New(AIIR_PYTHON_GET_WRAPPED_POINTER(affineMap),
                       AIIR_PYTHON_CAPSULE_AFFINE_MAP, NULL);
}

/** Extracts an AiirAffineMap from a capsule as produced from
 * aiirPythonAffineMapToCapsule. If the capsule is not of the right type, then
 * a null type is returned (as checked via aiirAffineMapIsNull). In such a
 * case, the Python APIs will have already set an error. */
static inline AiirAffineMap aiirPythonCapsuleToAffineMap(PyObject *capsule) {
  void *ptr = PyCapsule_GetPointer(capsule, AIIR_PYTHON_CAPSULE_AFFINE_MAP);
  AiirAffineMap affineMap = {ptr};
  return affineMap;
}

/** Creates a capsule object encapsulating the raw C-API AiirIntegerSet.
 * The returned capsule does not extend or affect ownership of any Python
 * objects that reference the set in any way. */
static inline PyObject *
aiirPythonIntegerSetToCapsule(AiirIntegerSet integerSet) {
  return PyCapsule_New(AIIR_PYTHON_GET_WRAPPED_POINTER(integerSet),
                       AIIR_PYTHON_CAPSULE_INTEGER_SET, NULL);
}

/** Extracts an AiirIntegerSet from a capsule as produced from
 * aiirPythonIntegerSetToCapsule. If the capsule is not of the right type, then
 * a null set is returned (as checked via aiirIntegerSetIsNull). In such a
 * case, the Python APIs will have already set an error. */
static inline AiirIntegerSet aiirPythonCapsuleToIntegerSet(PyObject *capsule) {
  void *ptr = PyCapsule_GetPointer(capsule, AIIR_PYTHON_CAPSULE_INTEGER_SET);
  AiirIntegerSet integerSet = {ptr};
  return integerSet;
}

/** Creates a capsule object encapsulating the raw C-API AiirExecutionEngine.
 * The returned capsule does not extend or affect ownership of any Python
 * objects that reference the set in any way. */
static inline PyObject *
aiirPythonExecutionEngineToCapsule(AiirExecutionEngine jit) {
  return PyCapsule_New(AIIR_PYTHON_GET_WRAPPED_POINTER(jit),
                       AIIR_PYTHON_CAPSULE_EXECUTION_ENGINE, NULL);
}

/** Extracts an AiirExecutionEngine from a capsule as produced from
 * aiirPythonIntegerSetToCapsule. If the capsule is not of the right type, then
 * a null set is returned (as checked via aiirExecutionEngineIsNull). In such a
 * case, the Python APIs will have already set an error. */
static inline AiirExecutionEngine
aiirPythonCapsuleToExecutionEngine(PyObject *capsule) {
  void *ptr =
      PyCapsule_GetPointer(capsule, AIIR_PYTHON_CAPSULE_EXECUTION_ENGINE);
  AiirExecutionEngine jit = {ptr};
  return jit;
}

/** Creates a capsule object encapsulating the raw C-API AiirValue.
 * The returned capsule does not extend or affect ownership of any Python
 * objects that reference the operation in any way.
 */
static inline PyObject *aiirPythonValueToCapsule(AiirValue value) {
  return PyCapsule_New(AIIR_PYTHON_GET_WRAPPED_POINTER(value),
                       AIIR_PYTHON_CAPSULE_VALUE, NULL);
}

/** Extracts an AiirValue from a capsule as produced from
 * aiirPythonValueToCapsule. If the capsule is not of the right type, then a
 * null type is returned (as checked via aiirValueIsNull). In such a case, the
 * Python APIs will have already set an error. */
static inline AiirValue aiirPythonCapsuleToValue(PyObject *capsule) {
  void *ptr = PyCapsule_GetPointer(capsule, AIIR_PYTHON_CAPSULE_VALUE);
  AiirValue value = {ptr};
  return value;
}

#ifdef __cplusplus
}
#endif

#endif // AIIR_C_BINDINGS_PYTHON_INTEROP_H
