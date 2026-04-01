//===- IRCore.h - IR helpers of python bindings ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_BINDINGS_PYTHON_IRCORE_H
#define AIIR_BINDINGS_PYTHON_IRCORE_H

#include <cstddef>
#include <exception>
#include <optional>
#include <sstream>
#include <utility>
#include <vector>

#include "Globals.h"
#include "NanobindUtils.h"
#include "aiir-c/AffineExpr.h"
#include "aiir-c/AffineMap.h"
#include "aiir-c/BuiltinAttributes.h"
#include "aiir-c/Debug.h"
#include "aiir-c/Diagnostics.h"
#include "aiir-c/ExtensibleDialect.h"
#include "aiir-c/IR.h"
#include "aiir-c/IntegerSet.h"
#include "aiir-c/Support.h"
#include "aiir-c/Transforms.h"
#include "aiir/Bindings/Python/Nanobind.h"
#include "aiir/Bindings/Python/NanobindAdaptors.h"

namespace aiir {
namespace python {
namespace AIIR_BINDINGS_PYTHON_DOMAIN {

class PyBlock;
class PyDiagnostic;
class PyDiagnosticHandler;
class PyInsertionPoint;
class PyLocation;
class DefaultingPyLocation;
class PyAiirContext;
class DefaultingPyAiirContext;
class PyModule;
class PyOperation;
class PyOperationBase;
class PyType;
class PySymbolTable;
class PyValue;

/// Wrapper for the global LLVM debugging flag.
struct AIIR_PYTHON_API_EXPORTED PyGlobalDebugFlag {
  static void set(nanobind::object &o, bool enable);
  static bool get(const nanobind::object &);
  static void bind(nanobind::module_ &m);

private:
  static nanobind::ft_mutex mutex;
};

/// Template for a reference to a concrete type which captures a python
/// reference to its underlying python object.
template <typename T>
class AIIR_PYTHON_API_EXPORTED PyObjectRef {
public:
  PyObjectRef(T *referrent, nanobind::object object)
      : referrent(referrent), object(std::move(object)) {
    assert(this->referrent &&
           "cannot construct PyObjectRef with null referrent");
    assert(this->object && "cannot construct PyObjectRef with null object");
  }
  PyObjectRef(PyObjectRef &&other) noexcept
      : referrent(other.referrent), object(std::move(other.object)) {
    other.referrent = nullptr;
    assert(!other.object);
  }
  PyObjectRef(const PyObjectRef &other)
      : referrent(other.referrent), object(other.object /* copies */) {}
  PyObjectRef &operator=(const PyObjectRef &other) {
    referrent = other.referrent;
    object = other.object;
    return *this;
  }
  PyObjectRef &operator=(PyObjectRef &&other) noexcept {
    referrent = other.referrent;
    object = std::move(other.object);
    other.referrent = nullptr;
    assert(!other.object);
    return *this;
  }
  ~PyObjectRef() = default;

  int getRefCount() {
    if (!object)
      return 0;
    return Py_REFCNT(object.ptr());
  }

  /// Releases the object held by this instance, returning it.
  /// This is the proper thing to return from a function that wants to return
  /// the reference. Note that this does not work from initializers.
  nanobind::object releaseObject() {
    assert(referrent && object);
    referrent = nullptr;
    auto stolen = std::move(object);
    return stolen;
  }

  T *get() { return referrent; }
  T *operator->() {
    assert(referrent && object);
    return referrent;
  }
  nanobind::object getObject() {
    assert(referrent && object);
    return object;
  }
  operator bool() const { return referrent && object; }

  using NBTypedT = nanobind::typed<nanobind::object, T>;

private:
  T *referrent;
  nanobind::object object;
};

/// Tracks an entry in the thread context stack. New entries are pushed onto
/// here for each with block that activates a new InsertionPoint, Context or
/// Location.
///
/// Pushing either a Location or InsertionPoint also pushes its associated
/// Context. Pushing a Context will not modify the Location or InsertionPoint
/// unless if they are from a different context, in which case, they are
/// cleared.
class AIIR_PYTHON_API_EXPORTED PyThreadContextEntry {
public:
  enum class FrameKind {
    Context,
    InsertionPoint,
    Location,
  };

  PyThreadContextEntry(FrameKind frameKind, nanobind::object context,
                       nanobind::object insertionPoint,
                       nanobind::object location)
      : context(std::move(context)), insertionPoint(std::move(insertionPoint)),
        location(std::move(location)), frameKind(frameKind) {}

  /// Gets the top of stack context and return nullptr if not defined.
  static PyAiirContext *getDefaultContext();

  /// Gets the top of stack insertion point and return nullptr if not defined.
  static PyInsertionPoint *getDefaultInsertionPoint();

  /// Gets the top of stack location and returns nullptr if not defined.
  static PyLocation *getDefaultLocation();

  PyAiirContext *getContext();
  PyInsertionPoint *getInsertionPoint();
  PyLocation *getLocation();
  FrameKind getFrameKind() { return frameKind; }

  /// Stack management.
  static PyThreadContextEntry *getTopOfStack();
  static nanobind::object pushContext(nanobind::object context);
  static void popContext(PyAiirContext &context);
  static nanobind::object pushInsertionPoint(nanobind::object insertionPoint);
  static void popInsertionPoint(PyInsertionPoint &insertionPoint);
  static nanobind::object pushLocation(nanobind::object location);
  static void popLocation(PyLocation &location);

  /// Gets the thread local stack.
  static std::vector<PyThreadContextEntry> &getStack();

private:
  static void push(FrameKind frameKind, nanobind::object context,
                   nanobind::object insertionPoint, nanobind::object location);

  /// An object reference to the PyContext.
  nanobind::object context;
  /// An object reference to the current insertion point.
  nanobind::object insertionPoint;
  /// An object reference to the current location.
  nanobind::object location;
  // The kind of push that was performed.
  FrameKind frameKind;
};

/// Wrapper around AiirLlvmThreadPool
/// Python object owns the C++ thread pool
class AIIR_PYTHON_API_EXPORTED PyThreadPool {
public:
  PyThreadPool();
  ~PyThreadPool();
  PyThreadPool(const PyThreadPool &) = delete;
  PyThreadPool(PyThreadPool &&) = delete;

  int getMaxConcurrency() const;
  AiirLlvmThreadPool get() { return threadPool; }

  std::string _aiir_thread_pool_ptr() const;

private:
  AiirLlvmThreadPool threadPool;
};

/// Wrapper around AiirContext.
using PyAiirContextRef = PyObjectRef<PyAiirContext>;
class AIIR_PYTHON_API_EXPORTED PyAiirContext {
public:
  PyAiirContext() = delete;
  PyAiirContext(AiirContext context);
  PyAiirContext(const PyAiirContext &) = delete;
  PyAiirContext(PyAiirContext &&) = delete;

  /// Returns a context reference for the singleton PyAiirContext wrapper for
  /// the given context.
  static PyAiirContextRef forContext(AiirContext context);
  ~PyAiirContext();

  /// Accesses the underlying AiirContext.
  AiirContext get() { return context; }

  /// Gets a strong reference to this context, which will ensure it is kept
  /// alive for the life of the reference.
  PyAiirContextRef getRef();

  /// Gets a capsule wrapping the void* within the AiirContext.
  nanobind::object getCapsule();

  /// Creates a PyAiirContext from the AiirContext wrapped by a capsule.
  /// Note that PyAiirContext instances are uniqued, so the returned object
  /// may be a pre-existing object. Ownership of the underlying AiirContext
  /// is taken by calling this function.
  static nanobind::object createFromCapsule(nanobind::object capsule);

  /// Gets the count of live context objects. Used for testing.
  static size_t getLiveCount();

  /// Gets the count of live modules associated with this context.
  /// Used for testing.
  size_t getLiveModuleCount();

  /// Enter and exit the context manager.
  static nanobind::object contextEnter(nanobind::object context);
  void contextExit(const nanobind::object &excType,
                   const nanobind::object &excVal,
                   const nanobind::object &excTb);

  /// Attaches a Python callback as a diagnostic handler, returning a
  /// registration object (internally a PyDiagnosticHandler).
  nanobind::object attachDiagnosticHandler(nanobind::object callback);

  /// Controls whether error diagnostics should be propagated to diagnostic
  /// handlers, instead of being captured by `ErrorCapture`.
  void setEmitErrorDiagnostics(bool value) { emitErrorDiagnostics = value; }
  bool getEmitErrorDiagnostics() { return emitErrorDiagnostics; }
  struct ErrorCapture;

private:
  // Interns the mapping of live AiirContext::ptr to PyAiirContext instances,
  // preserving the relationship that an AiirContext maps to a single
  // PyAiirContext wrapper. This could be replaced in the future with an
  // extension mechanism on the AiirContext for stashing user pointers.
  // Note that this holds a handle, which does not imply ownership.
  // Mappings will be removed when the context is destructed.
  using LiveContextMap = std::unordered_map<void *, PyAiirContext *>;
  static nanobind::ft_mutex live_contexts_mutex;
  static LiveContextMap &getLiveContexts();

  // Interns all live modules associated with this context. Modules tracked
  // in this map are valid. When a module is invalidated, it is removed
  // from this map, and while it still exists as an instance, any
  // attempt to access it will raise an error.
  using LiveModuleMap =
      std::unordered_map<const void *, std::pair<nanobind::handle, PyModule *>>;
  LiveModuleMap liveModules;

  bool emitErrorDiagnostics = false;

  AiirContext context;
  friend class PyModule;
  friend class PyOperation;
};

/// Used in function arguments when None should resolve to the current context
/// manager set instance.
class AIIR_PYTHON_API_EXPORTED DefaultingPyAiirContext
    : public Defaulting<DefaultingPyAiirContext, PyAiirContext> {
public:
  using Defaulting::Defaulting;
  static constexpr const char kTypeDescription[] = "_aiir.ir.Context";
  static PyAiirContext &resolve();
};

/// Base class for all objects that directly or indirectly depend on an
/// AiirContext. The lifetime of the context will extend at least to the
/// lifetime of these instances.
/// Immutable objects that depend on a context extend this directly.
class AIIR_PYTHON_API_EXPORTED BaseContextObject {
public:
  BaseContextObject(PyAiirContextRef ref) : contextRef(std::move(ref)) {
    assert(this->contextRef &&
           "context object constructed with null context ref");
  }

  /// Accesses the context reference.
  PyAiirContextRef &getContext() { return contextRef; }

private:
  PyAiirContextRef contextRef;
};

/// Wrapper around an AiirLocation.
class AIIR_PYTHON_API_EXPORTED PyLocation : public BaseContextObject {
public:
  PyLocation(PyAiirContextRef contextRef, AiirLocation loc)
      : BaseContextObject(std::move(contextRef)), loc(loc) {}

  operator AiirLocation() const { return loc; }
  AiirLocation get() const { return loc; }

  /// Enter and exit the context manager.
  static nanobind::object contextEnter(nanobind::object location);
  void contextExit(const nanobind::object &excType,
                   const nanobind::object &excVal,
                   const nanobind::object &excTb);

  /// Gets a capsule wrapping the void* within the AiirLocation.
  nanobind::object getCapsule();

  /// Creates a PyLocation from the AiirLocation wrapped by a capsule.
  /// Note that PyLocation instances are uniqued, so the returned object
  /// may be a pre-existing object. Ownership of the underlying AiirLocation
  /// is taken by calling this function.
  static PyLocation createFromCapsule(nanobind::object capsule);

private:
  AiirLocation loc;
};

enum class PyDiagnosticSeverity : std::underlying_type_t<
    AiirDiagnosticSeverity> {
  Error = AiirDiagnosticError,
  Warning = AiirDiagnosticWarning,
  Note = AiirDiagnosticNote,
  Remark = AiirDiagnosticRemark
};

enum class PyWalkResult : std::underlying_type_t<AiirWalkResult> {
  Advance = AiirWalkResultAdvance,
  Interrupt = AiirWalkResultInterrupt,
  Skip = AiirWalkResultSkip
};

/// Traversal order for operation walk.
enum class PyWalkOrder : std::underlying_type_t<AiirWalkOrder> {
  PreOrder = AiirWalkPreOrder,
  PostOrder = AiirWalkPostOrder
};

/// Python class mirroring the C AiirDiagnostic struct. Note that these structs
/// are only valid for the duration of a diagnostic callback and attempting
/// to access them outside of that will raise an exception. This applies to
/// nested diagnostics (in the notes) as well.
class AIIR_PYTHON_API_EXPORTED PyDiagnostic {
public:
  PyDiagnostic(AiirDiagnostic diagnostic) : diagnostic(diagnostic) {}
  void invalidate();
  bool isValid() { return valid; }
  PyDiagnosticSeverity getSeverity();
  PyLocation getLocation();
  nanobind::str getMessage();
  nanobind::typed<nanobind::tuple, PyDiagnostic> getNotes();

  /// Materialized diagnostic information. This is safe to access outside the
  /// diagnostic callback.
  struct DiagnosticInfo {
    PyDiagnosticSeverity severity;
    PyLocation location;
    std::string message;
    std::vector<DiagnosticInfo> notes;
  };
  DiagnosticInfo getInfo();

private:
  AiirDiagnostic diagnostic;

  void checkValid();
  /// If notes have been materialized from the diagnostic, then this will
  /// be populated with the corresponding objects (all castable to
  /// PyDiagnostic).
  std::optional<nanobind::tuple> materializedNotes;
  bool valid = true;
};

/// Represents a diagnostic handler attached to the context. The handler's
/// callback will be invoked with PyDiagnostic instances until the detach()
/// method is called or the context is destroyed. A diagnostic handler can be
/// the subject of a `with` block, which will detach it when the block exits.
///
/// Since diagnostic handlers can call back into Python code which can do
/// unsafe things (i.e. recursively emitting diagnostics, raising exceptions,
/// etc), this is generally not deemed to be a great user-level API. Users
/// should generally use some form of DiagnosticCollector. If the handler raises
/// any exceptions, they will just be emitted to stderr and dropped.
///
/// The unique usage of this class means that its lifetime management is
/// different from most other parts of the API. Instances are always created
/// in an attached state and can transition to a detached state by either:
///   a) The context being destroyed and unregistering all handlers.
///   b) An explicit call to detach().
/// The object may remain live from a Python perspective for an arbitrary time
/// after detachment, but there is nothing the user can do with it (since there
/// is no way to attach an existing handler object).
class AIIR_PYTHON_API_EXPORTED PyDiagnosticHandler {
public:
  PyDiagnosticHandler(AiirContext context, nanobind::object callback);
  ~PyDiagnosticHandler();

  bool isAttached() { return registeredID.has_value(); }
  bool getHadError() { return hadError; }

  /// Detaches the handler. Does nothing if not attached.
  void detach();

  nanobind::object contextEnter() { return nanobind::cast(this); }
  void contextExit(const nanobind::object &excType,
                   const nanobind::object &excVal,
                   const nanobind::object &excTb) {
    detach();
  }

private:
  AiirContext context;
  nanobind::object callback;
  std::optional<AiirDiagnosticHandlerID> registeredID;
  bool hadError = false;
  friend class PyAiirContext;
};

/// RAII object that captures any error diagnostics emitted to the provided
/// context.
struct AIIR_PYTHON_API_EXPORTED PyAiirContext::ErrorCapture {
  ErrorCapture(PyAiirContextRef ctx)
      : ctx(ctx), handlerID(aiirContextAttachDiagnosticHandler(
                      ctx->get(), handler, /*userData=*/this,
                      /*deleteUserData=*/nullptr)) {}
  ~ErrorCapture() {
    aiirContextDetachDiagnosticHandler(ctx->get(), handlerID);
    assert(errors.empty() && "unhandled captured errors");
  }

  std::vector<PyDiagnostic::DiagnosticInfo> take() {
    return std::move(errors);
  };

private:
  PyAiirContextRef ctx;
  AiirDiagnosticHandlerID handlerID;
  std::vector<PyDiagnostic::DiagnosticInfo> errors;

  static AiirLogicalResult handler(AiirDiagnostic diag, void *userData);
};

/// Wrapper around an AiirDialect. This is exported as `DialectDescriptor` in
/// order to differentiate it from the `Dialect` base class which is extended by
/// plugins which extend dialect functionality through extension python code.
/// This should be seen as the "low-level" object and `Dialect` as the
/// high-level, user facing object.
class AIIR_PYTHON_API_EXPORTED PyDialectDescriptor : public BaseContextObject {
public:
  PyDialectDescriptor(PyAiirContextRef contextRef, AiirDialect dialect)
      : BaseContextObject(std::move(contextRef)), dialect(dialect) {}

  AiirDialect get() { return dialect; }

private:
  AiirDialect dialect;
};

/// User-level object for accessing dialects with dotted syntax such as:
///   ctx.dialect.std
class AIIR_PYTHON_API_EXPORTED PyDialects : public BaseContextObject {
public:
  PyDialects(PyAiirContextRef contextRef)
      : BaseContextObject(std::move(contextRef)) {}

  AiirDialect getDialectForKey(const std::string &key, bool attrError);
};

/// User-level dialect object. For dialects that have a registered extension,
/// this will be the base class of the extension dialect type. For un-extended,
/// objects of this type will be returned directly.
class AIIR_PYTHON_API_EXPORTED PyDialect {
public:
  PyDialect(nanobind::object descriptor) : descriptor(std::move(descriptor)) {}

  nanobind::object getDescriptor() { return descriptor; }

private:
  nanobind::object descriptor;
};

/// Wrapper around an AiirDialectRegistry.
/// Upon construction, the Python wrapper takes ownership of the
/// underlying AiirDialectRegistry.
class AIIR_PYTHON_API_EXPORTED PyDialectRegistry {
public:
  PyDialectRegistry() : registry(aiirDialectRegistryCreate()) {}
  PyDialectRegistry(AiirDialectRegistry registry) : registry(registry) {}
  ~PyDialectRegistry() {
    if (!aiirDialectRegistryIsNull(registry))
      aiirDialectRegistryDestroy(registry);
  }
  PyDialectRegistry(PyDialectRegistry &) = delete;
  PyDialectRegistry(PyDialectRegistry &&other) noexcept
      : registry(other.registry) {
    other.registry = {nullptr};
  }

  operator AiirDialectRegistry() const { return registry; }
  AiirDialectRegistry get() const { return registry; }

  nanobind::object getCapsule();
  static PyDialectRegistry createFromCapsule(nanobind::object capsule);

private:
  AiirDialectRegistry registry;
};

/// Used in function arguments when None should resolve to the current context
/// manager set instance.
class AIIR_PYTHON_API_EXPORTED DefaultingPyLocation
    : public Defaulting<DefaultingPyLocation, PyLocation> {
public:
  using Defaulting::Defaulting;
  static constexpr const char kTypeDescription[] = "_aiir.ir.Location";
  static PyLocation &resolve();

  operator AiirLocation() const { return *get(); }
};

/// Wrapper around AiirModule.
/// This is the top-level, user-owned object that contains regions/ops/blocks.
class PyModule;
using PyModuleRef = PyObjectRef<PyModule>;
class AIIR_PYTHON_API_EXPORTED PyModule : public BaseContextObject {
public:
  /// Returns a PyModule reference for the given AiirModule. This always returns
  /// a new object.
  static PyModuleRef forModule(AiirModule module);
  PyModule(PyModule &) = delete;
  PyModule(PyAiirContext &&) = delete;
  ~PyModule();

  /// Gets the backing AiirModule.
  AiirModule get() { return module; }

  /// Gets a strong reference to this module.
  PyModuleRef getRef() {
    return PyModuleRef(this, nanobind::borrow<nanobind::object>(handle));
  }

  /// Gets a capsule wrapping the void* within the AiirModule.
  /// Note that the module does not (yet) provide a corresponding factory for
  /// constructing from a capsule as that would require uniquing PyModule
  /// instances, which is not currently done.
  nanobind::object getCapsule();

  /// Creates a PyModule from the AiirModule wrapped by a capsule.
  /// Note this returns a new object BUT clearAiirModule() must be called to
  /// prevent double-frees (of the underlying aiir::Module).
  static nanobind::object createFromCapsule(nanobind::object capsule);

  void clearAiirModule() { module = {nullptr}; }

private:
  PyModule(PyAiirContextRef contextRef, AiirModule module);
  AiirModule module;
  nanobind::handle handle;
};

class PyAsmState;

/// Base class for PyOperation and PyOpView which exposes the primary, user
/// visible methods for manipulating it.
class AIIR_PYTHON_API_EXPORTED PyOperationBase {
public:
  virtual ~PyOperationBase() = default;
  /// Implements the bound 'print' method and helps with others.
  void print(std::optional<int64_t> largeElementsLimit,
             std::optional<int64_t> largeResourceLimit, bool enableDebugInfo,
             bool prettyDebugInfo, bool printGenericOpForm, bool useLocalScope,
             bool useNameLocAsPrefix, bool assumeVerified,
             nanobind::object fileObject, bool binary, bool skipRegions);
  void print(PyAsmState &state, nanobind::object fileObject, bool binary);

  nanobind::object
  getAsm(bool binary, std::optional<int64_t> largeElementsLimit,
         std::optional<int64_t> largeResourceLimit, bool enableDebugInfo,
         bool prettyDebugInfo, bool printGenericOpForm, bool useLocalScope,
         bool useNameLocAsPrefix, bool assumeVerified, bool skipRegions);

  // Implement the bound 'writeBytecode' method.
  void writeBytecode(const nanobind::object &fileObject,
                     std::optional<int64_t> bytecodeVersion);

  // Implement the walk method.
  void walk(std::function<PyWalkResult(AiirOperation)> callback,
            PyWalkOrder walkOrder);

  /// Moves the operation before or after the other operation.
  void moveAfter(PyOperationBase &other);
  void moveBefore(PyOperationBase &other);

  /// Given an operation 'other' that is within the same parent block, return
  /// whether the current operation is before 'other' in the operation list
  /// of the parent block.
  /// Note: This function has an average complexity of O(1), but worst case may
  /// take O(N) where N is the number of operations within the parent block.
  bool isBeforeInBlock(PyOperationBase &other);

  /// Verify the operation. Throws `AIIRError` if verification fails, and
  /// returns `true` otherwise.
  bool verify();

  /// Each must provide access to the raw Operation.
  virtual PyOperation &getOperation() = 0;
};

/// Wrapper around PyOperation.
/// Operations exist in either an attached (dependent) or detached (top-level)
/// state. In the detached state (as on creation), an operation is owned by
/// the creator and its lifetime extends either until its reference count
/// drops to zero or it is attached to a parent, at which point its lifetime
/// is bounded by its top-level parent reference.
class PyOperation;
class PyOpView;
using PyOperationRef = PyObjectRef<PyOperation>;
class AIIR_PYTHON_API_EXPORTED PyOperation : public PyOperationBase,
                                             public BaseContextObject {
public:
  ~PyOperation() override;
  PyOperation &getOperation() override { return *this; }

  /// Returns a PyOperation for the given AiirOperation, optionally associating
  /// it with a parentKeepAlive.
  static PyOperationRef
  forOperation(PyAiirContextRef contextRef, AiirOperation operation,
               nanobind::object parentKeepAlive = nanobind::object());

  /// Creates a detached operation. The operation must not be associated with
  /// any existing live operation.
  static PyOperationRef
  createDetached(PyAiirContextRef contextRef, AiirOperation operation,
                 nanobind::object parentKeepAlive = nanobind::object());

  /// Parses a source string (either text assembly or bytecode), creating a
  /// detached operation.
  static PyOperationRef parse(PyAiirContextRef contextRef,
                              const std::string &sourceStr,
                              const std::string &sourceName);

  /// Detaches the operation from its parent block and updates its state
  /// accordingly.
  void detachFromParent();

  /// Gets the backing operation.
  operator AiirOperation() const { return get(); }
  AiirOperation get() const;

  PyOperationRef getRef();

  bool isAttached() { return attached; }
  void setAttached(const nanobind::object &parent = nanobind::object());
  void setDetached();
  void checkValid() const;

  /// Gets the owning block or raises an exception if the operation has no
  /// owning block.
  PyBlock getBlock();

  /// Gets the parent operation or raises an exception if the operation has
  /// no parent.
  std::optional<PyOperationRef> getParentOperation();

  /// Gets a capsule wrapping the void* within the AiirOperation.
  nanobind::object getCapsule();

  /// Creates a PyOperation from the AiirOperation wrapped by a capsule.
  /// Ownership of the underlying AiirOperation is taken by calling this
  /// function.
  static nanobind::object createFromCapsule(const nanobind::object &capsule);

  /// Creates an operation. See corresponding python docstring.
  static nanobind::object
  create(std::string_view name, std::optional<std::vector<PyType *>> results,
         const AiirValue *operands, size_t numOperands,
         std::optional<nanobind::dict> attributes,
         std::optional<std::vector<PyBlock *>> successors, int regions,
         PyLocation &location, const nanobind::object &ip, bool inferType);

  /// Creates an OpView suitable for this operation.
  nanobind::object createOpView();

  /// Erases the underlying AiirOperation, removes its pointer from the
  /// parent context's live operations map, and sets the valid bit false.
  void erase();

  /// Invalidate the operation.
  void setInvalid() { valid = false; }

  /// Clones this operation.
  nanobind::object clone(const nanobind::object &ip);

  PyOperation(PyAiirContextRef contextRef, AiirOperation operation);

private:
  static PyOperationRef createInstance(PyAiirContextRef contextRef,
                                       AiirOperation operation,
                                       nanobind::object parentKeepAlive);

  AiirOperation operation;
  nanobind::handle handle;
  // Keeps the parent alive, regardless of whether it is an Operation or
  // Module.
  // TODO: As implemented, this facility is only sufficient for modeling the
  // trivial module parent back-reference. Generalize this to also account for
  // transitions from detached to attached and address TODOs in the
  // ir_operation.py regarding testing corresponding lifetime guarantees.
  nanobind::object parentKeepAlive;
  bool attached = true;
  bool valid = true;

  friend class PyOperationBase;
  friend class PySymbolTable;
};

/// A PyOpView is equivalent to the C++ "Op" wrappers: these are the basis for
/// providing more instance-specific accessors and serve as the base class for
/// custom ODS-style operation classes. Since this class is subclass on the
/// python side, it must present an __init__ method that operates in pure
/// python types.
class AIIR_PYTHON_API_EXPORTED PyOpView : public PyOperationBase {
public:
  PyOpView(const nanobind::object &operationObject);
  PyOperation &getOperation() override { return operation; }

  nanobind::object getOperationObject() { return operationObject; }

  static nanobind::typed<nanobind::object, PyOperation>
  buildGeneric(std::string_view name, std::tuple<int, bool> opRegionSpec,
               nanobind::object operandSegmentSpecObj,
               nanobind::object resultSegmentSpecObj,
               std::optional<nanobind::sequence> resultTypeList,
               nanobind::sequence operandList,
               std::optional<nanobind::dict> attributes,
               std::optional<std::vector<PyBlock *>> successors,
               std::optional<int> regions, PyLocation &location,
               const nanobind::object &maybeIp);

  /// Construct an instance of a class deriving from OpView, bypassing its
  /// `__init__` method. The derived class will typically define a constructor
  /// that provides a convenient builder, but we need to side-step this when
  /// constructing an `OpView` for an already-built operation.
  ///
  /// The caller is responsible for verifying that `operation` is a valid
  /// operation to construct `cls` with.
  static nanobind::object constructDerived(const nanobind::object &cls,
                                           const nanobind::object &operation);

private:
  PyOperation &operation;           // For efficient, cast-free access from C++
  nanobind::object operationObject; // Holds the reference.
};

/// Wrapper around an AiirRegion.
/// Regions are managed completely by their containing operation. Unlike the
/// C++ API, the python API does not support detached regions.
class AIIR_PYTHON_API_EXPORTED PyRegion {
public:
  PyRegion(PyOperationRef parentOperation, AiirRegion region)
      : parentOperation(std::move(parentOperation)), region(region) {
    assert(!aiirRegionIsNull(region) && "python region cannot be null");
  }
  operator AiirRegion() const { return region; }

  AiirRegion get() { return region; }
  PyOperationRef &getParentOperation() { return parentOperation; }

  void checkValid() { return parentOperation->checkValid(); }

private:
  PyOperationRef parentOperation;
  AiirRegion region;
};

/// Wrapper around an AiirAsmState.
class AIIR_PYTHON_API_EXPORTED PyAsmState {
public:
  PyAsmState(AiirValue value, bool useLocalScope);
  PyAsmState(PyOperationBase &operation, bool useLocalScope);
  ~PyAsmState() { aiirOpPrintingFlagsDestroy(flags); }
  // Delete copy constructors.
  PyAsmState(PyAsmState &other) = delete;
  PyAsmState(const PyAsmState &other) = delete;

  AiirAsmState get() { return state; }

private:
  AiirAsmState state;
  AiirOpPrintingFlags flags;
};

/// Wrapper around an AiirBlock.
/// Blocks are managed completely by their containing operation. Unlike the
/// C++ API, the python API does not support detached blocks.
class AIIR_PYTHON_API_EXPORTED PyBlock {
public:
  PyBlock(PyOperationRef parentOperation, AiirBlock block)
      : parentOperation(std::move(parentOperation)), block(block) {
    assert(!aiirBlockIsNull(block) && "python block cannot be null");
  }

  AiirBlock get() { return block; }
  PyOperationRef &getParentOperation() { return parentOperation; }

  void checkValid() { return parentOperation->checkValid(); }

  /// Gets a capsule wrapping the void* within the AiirBlock.
  nanobind::object getCapsule();

private:
  PyOperationRef parentOperation;
  AiirBlock block;
};

/// An insertion point maintains a pointer to a Block and a reference operation.
/// Calls to insert() will insert a new operation before the
/// reference operation. If the reference operation is null, then appends to
/// the end of the block.
class AIIR_PYTHON_API_EXPORTED PyInsertionPoint {
public:
  /// Creates an insertion point positioned after the last operation in the
  /// block, but still inside the block.
  PyInsertionPoint(const PyBlock &block);
  /// Creates an insertion point positioned before a reference operation.
  PyInsertionPoint(PyOperationBase &beforeOperationBase);
  /// Creates an insertion point positioned before a reference operation.
  PyInsertionPoint(PyOperationRef beforeOperationRef);

  /// Shortcut to create an insertion point at the beginning of the block.
  static PyInsertionPoint atBlockBegin(PyBlock &block);
  /// Shortcut to create an insertion point before the block terminator.
  static PyInsertionPoint atBlockTerminator(PyBlock &block);
  /// Shortcut to create an insertion point to the node after the specified
  /// operation.
  static PyInsertionPoint after(PyOperationBase &op);

  /// Inserts an operation.
  void insert(PyOperationBase &operationBase);

  /// Enter and exit the context manager.
  static nanobind::object contextEnter(nanobind::object insertionPoint);
  void contextExit(const nanobind::object &excType,
                   const nanobind::object &excVal,
                   const nanobind::object &excTb);

  PyBlock &getBlock() { return block; }
  std::optional<PyOperationRef> &getRefOperation() { return refOperation; }

private:
  // Trampoline constructor that avoids null initializing members while
  // looking up parents.
  PyInsertionPoint(PyBlock block, std::optional<PyOperationRef> refOperation)
      : refOperation(std::move(refOperation)), block(std::move(block)) {}

  std::optional<PyOperationRef> refOperation;
  PyBlock block;
};

/// Wrapper around the generic AiirType.
/// The lifetime of a type is bound by the PyContext that created it.
class AIIR_PYTHON_API_EXPORTED PyType : public BaseContextObject {
public:
  PyType(PyAiirContextRef contextRef, AiirType type)
      : BaseContextObject(std::move(contextRef)), type(type) {}
  bool operator==(const PyType &other) const;
  operator AiirType() const { return type; }
  AiirType get() const { return type; }

  /// Gets a capsule wrapping the void* within the AiirType.
  nanobind::object getCapsule();

  /// Creates a PyType from the AiirType wrapped by a capsule.
  /// Note that PyType instances are uniqued, so the returned object
  /// may be a pre-existing object. Ownership of the underlying AiirType
  /// is taken by calling this function.
  static PyType createFromCapsule(nanobind::object capsule);

  nanobind::typed<nanobind::object, PyType> maybeDownCast();

private:
  AiirType type;
};

/// A TypeID provides an efficient and unique identifier for a specific C++
/// type. This allows for a C++ type to be compared, hashed, and stored in an
/// opaque context. This class wraps around the generic AiirTypeID.
class AIIR_PYTHON_API_EXPORTED PyTypeID {
public:
  PyTypeID(AiirTypeID typeID) : typeID(typeID) {}
  // Note, this tests whether the underlying TypeIDs are the same,
  // not whether the wrapper AiirTypeIDs are the same, nor whether
  // the PyTypeID objects are the same (i.e., PyTypeID is a value type).
  bool operator==(const PyTypeID &other) const;
  operator AiirTypeID() const { return typeID; }
  AiirTypeID get() { return typeID; }

  /// Gets a capsule wrapping the void* within the AiirTypeID.
  nanobind::object getCapsule();

  /// Creates a PyTypeID from the AiirTypeID wrapped by a capsule.
  static PyTypeID createFromCapsule(nanobind::object capsule);

private:
  AiirTypeID typeID;
};

/// CRTP base classes for Python types that subclass Type and should be
/// castable from it (i.e. via something like IntegerType(t)).
/// By default, type class hierarchies are one level deep (i.e. a
/// concrete type class extends PyType); however, intermediate python-visible
/// base classes can be modeled by specifying a BaseTy.
template <typename DerivedTy, typename BaseTy = PyType>
class AIIR_PYTHON_API_EXPORTED PyConcreteType : public BaseTy {
public:
  // Derived classes must define statics for:
  //   IsAFunctionTy isaFunction
  //   const char *pyClassName
  using ClassTy = nanobind::class_<DerivedTy, BaseTy>;
  using IsAFunctionTy = bool (*)(AiirType);
  using GetTypeIDFunctionTy = AiirTypeID (*)();
  using Base = PyConcreteType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction = nullptr;
  static inline const AiirStringRef name{};

  PyConcreteType() = default;
  PyConcreteType(PyAiirContextRef contextRef, AiirType t)
      : BaseTy(std::move(contextRef), t) {}
  PyConcreteType(PyType &orig)
      : PyConcreteType(orig.getContext(), castFrom(orig)) {}

  static AiirType castFrom(PyType &orig) {
    if (!DerivedTy::isaFunction(orig)) {
      auto origRepr =
          nanobind::cast<std::string>(nanobind::repr(nanobind::cast(orig)));
      throw nanobind::value_error((std::string("Cannot cast type to ") +
                                   DerivedTy::pyClassName + " (from " +
                                   origRepr + ")")
                                      .c_str());
    }
    return orig;
  }

  static void bind(nanobind::module_ &m) {
    auto cls = ClassTy(m, DerivedTy::pyClassName, nanobind::is_generic());
    cls.def(nanobind::init<PyType &>(), nanobind::keep_alive<0, 1>(),
            nanobind::arg("cast_from_type"));
    cls.def_prop_ro_static("static_typeid", [](nanobind::object & /*class*/) {
      if (DerivedTy::getTypeIdFunction)
        return PyTypeID(DerivedTy::getTypeIdFunction());
      throw nanobind::attribute_error(
          (DerivedTy::pyClassName + std::string(" has no typeid.")).c_str());
    });
    cls.def_prop_ro("typeid", [](PyType &self) {
      return nanobind::cast<PyTypeID>(nanobind::cast(self).attr("typeid"));
    });
    cls.def("__repr__", [](DerivedTy &self) {
      PyPrintAccumulator printAccum;
      printAccum.parts.append(DerivedTy::pyClassName);
      printAccum.parts.append("(");
      aiirTypePrint(self, printAccum.getCallback(), printAccum.getUserData());
      printAccum.parts.append(")");
      return printAccum.join();
    });

    if (DerivedTy::getTypeIdFunction) {
      PyGlobals::get().registerTypeCaster(
          DerivedTy::getTypeIdFunction(),
          nanobind::cast<nanobind::callable>(nanobind::cpp_function(
              [](PyType pyType) -> DerivedTy { return pyType; })),
          /*replace*/ true);
    }

    if (DerivedTy::name.length != 0) {
      cls.def_prop_ro_static("type_name", [](nanobind::object & /*self*/) {
        return nanobind::str(DerivedTy::name.data, DerivedTy::name.length);
      });
    }

    DerivedTy::bindDerived(cls);
  }

  /// Implemented by derived classes to add methods to the Python subclass.
  static void bindDerived(ClassTy &m) {}
};

/// Wrapper around the generic AiirAttribute.
/// The lifetime of a type is bound by the PyContext that created it.
class AIIR_PYTHON_API_EXPORTED PyAttribute : public BaseContextObject {
public:
  PyAttribute(PyAiirContextRef contextRef, AiirAttribute attr)
      : BaseContextObject(std::move(contextRef)), attr(attr) {}
  bool operator==(const PyAttribute &other) const;
  operator AiirAttribute() const { return attr; }
  AiirAttribute get() const { return attr; }

  /// Gets a capsule wrapping the void* within the AiirAttribute.
  nanobind::object getCapsule();

  /// Creates a PyAttribute from the AiirAttribute wrapped by a capsule.
  /// Note that PyAttribute instances are uniqued, so the returned object
  /// may be a pre-existing object. Ownership of the underlying AiirAttribute
  /// is taken by calling this function.
  static PyAttribute createFromCapsule(const nanobind::object &capsule);

  nanobind::typed<nanobind::object, PyAttribute> maybeDownCast();

private:
  AiirAttribute attr;
};

/// Represents a Python AiirNamedAttr, carrying an optional owned name.
/// TODO: Refactor this and the C-API to be based on an Identifier owned
/// by the context so as to avoid ownership issues here.
class AIIR_PYTHON_API_EXPORTED PyNamedAttribute {
public:
  /// Constructs a PyNamedAttr that retains an owned name. This should be
  /// used in any code that originates an AiirNamedAttribute from a python
  /// string.
  /// The lifetime of the PyNamedAttr must extend to the lifetime of the
  /// passed attribute.
  PyNamedAttribute(AiirAttribute attr, std::string ownedName);

  AiirNamedAttribute namedAttr;

private:
  // Since the AiirNamedAttr contains an internal pointer to the actual
  // memory of the owned string, it must be heap allocated to remain valid.
  // Otherwise, strings that fit within the small object optimization threshold
  // will have their memory address change as the containing object is moved,
  // resulting in an invalid aliased pointer.
  std::unique_ptr<std::string> ownedName;
};

/// CRTP base classes for Python attributes that subclass Attribute and should
/// be castable from it (i.e. via something like StringAttr(attr)).
/// By default, attribute class hierarchies are one level deep (i.e. a
/// concrete attribute class extends PyAttribute); however, intermediate
/// python-visible base classes can be modeled by specifying a BaseTy.
template <typename DerivedTy, typename BaseTy = PyAttribute>
class AIIR_PYTHON_API_EXPORTED PyConcreteAttribute : public BaseTy {
public:
  // Derived classes must define statics for:
  //   IsAFunctionTy isaFunction
  //   const char *pyClassName
  using ClassTy = nanobind::class_<DerivedTy, BaseTy>;
  using IsAFunctionTy = bool (*)(AiirAttribute);
  using GetTypeIDFunctionTy = AiirTypeID (*)();
  static constexpr GetTypeIDFunctionTy getTypeIdFunction = nullptr;
  static inline const AiirStringRef name{};
  using Base = PyConcreteAttribute;

  PyConcreteAttribute() = default;
  PyConcreteAttribute(PyAiirContextRef contextRef, AiirAttribute attr)
      : BaseTy(std::move(contextRef), attr) {}
  PyConcreteAttribute(PyAttribute &orig)
      : PyConcreteAttribute(orig.getContext(), castFrom(orig)) {}

  static AiirAttribute castFrom(PyAttribute &orig) {
    if (!DerivedTy::isaFunction(orig)) {
      auto origRepr =
          nanobind::cast<std::string>(nanobind::repr(nanobind::cast(orig)));
      throw nanobind::value_error((std::string("Cannot cast attribute to ") +
                                   DerivedTy::pyClassName + " (from " +
                                   origRepr + ")")
                                      .c_str());
    }
    return orig;
  }

  static void bind(nanobind::module_ &m, PyType_Slot *slots = nullptr) {
    ClassTy cls;
    if (slots) {
      cls = ClassTy(m, DerivedTy::pyClassName, nanobind::type_slots(slots),
                    nanobind::is_generic());
    } else {
      cls = ClassTy(m, DerivedTy::pyClassName, nanobind::is_generic());
    }
    cls.def(nanobind::init<PyAttribute &>(), nanobind::keep_alive<0, 1>(),
            nanobind::arg("cast_from_attr"));
    cls.def_prop_ro(
        "type",
        [](PyAttribute &attr) -> nanobind::typed<nanobind::object, PyType> {
          return PyType(attr.getContext(), aiirAttributeGetType(attr))
              .maybeDownCast();
        });
    cls.def_prop_ro_static("static_typeid", [](nanobind::object & /*class*/) {
      if (DerivedTy::getTypeIdFunction)
        return PyTypeID(DerivedTy::getTypeIdFunction());
      throw nanobind::attribute_error(
          (DerivedTy::pyClassName + std::string(" has no typeid.")).c_str());
    });
    cls.def_prop_ro("typeid", [](PyAttribute &self) {
      return nanobind::cast<PyTypeID>(nanobind::cast(self).attr("typeid"));
    });
    cls.def("__repr__", [](DerivedTy &self) {
      PyPrintAccumulator printAccum;
      printAccum.parts.append(DerivedTy::pyClassName);
      printAccum.parts.append("(");
      aiirAttributePrint(self, printAccum.getCallback(),
                         printAccum.getUserData());
      printAccum.parts.append(")");
      return printAccum.join();
    });

    if (DerivedTy::getTypeIdFunction) {
      PyGlobals::get().registerTypeCaster(
          DerivedTy::getTypeIdFunction(),
          nanobind::cast<nanobind::callable>(
              nanobind::cpp_function([](PyAttribute pyAttribute) -> DerivedTy {
                return pyAttribute;
              })),
          /*replace*/ true);
    }

    if (DerivedTy::name.length != 0) {
      cls.def_prop_ro_static("attr_name", [](nanobind::object & /*self*/) {
        return nanobind::str(DerivedTy::name.data, DerivedTy::name.length);
      });
    }

    DerivedTy::bindDerived(cls);
  }

  /// Implemented by derived classes to add methods to the Python subclass.
  static void bindDerived(ClassTy &m) {}
};

class AIIR_PYTHON_API_EXPORTED PyStringAttribute
    : public PyConcreteAttribute<PyStringAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirAttributeIsAString;
  static constexpr const char *pyClassName = "StringAttr";
  using PyConcreteAttribute::PyConcreteAttribute;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirStringAttrGetTypeID;
  static inline const AiirStringRef name = aiirStringAttrGetName();

  static void bindDerived(ClassTy &c);
};

/// Wrapper around the generic AiirValue.
/// Values are managed completely by the operation that resulted in their
/// definition. For op result value, this is the operation that defines the
/// value. For block argument values, this is the operation that contains the
/// block to which the value is an argument (blocks cannot be detached in Python
/// bindings so such operation always exists).
class PyBlockArgument;
class PyOpResult;
class AIIR_PYTHON_API_EXPORTED PyValue {
public:
  // The virtual here is "load bearing" in that it enables RTTI
  // for PyConcreteValue CRTP classes that support maybeDownCast.
  // See PyValue::maybeDownCast.
  virtual ~PyValue() = default;
  PyValue(PyOperationRef parentOperation, AiirValue value)
      : parentOperation(std::move(parentOperation)), value(value) {}
  operator AiirValue() const { return value; }

  AiirValue get() { return value; }
  PyOperationRef &getParentOperation() { return parentOperation; }

  void checkValid() { return parentOperation->checkValid(); }

  /// Gets a capsule wrapping the void* within the AiirValue.
  nanobind::object getCapsule();

  nanobind::typed<nanobind::object,
                  std::variant<PyBlockArgument, PyOpResult, PyValue>>
  maybeDownCast();

  /// Creates a PyValue from the AiirValue wrapped by a capsule. Ownership of
  /// the underlying AiirValue is still tied to the owning operation.
  static PyValue createFromCapsule(nanobind::object capsule);

private:
  PyOperationRef parentOperation;
  AiirValue value;
};

/// Wrapper around AiirAffineExpr. Affine expressions are owned by the context.
class AIIR_PYTHON_API_EXPORTED PyAffineExpr : public BaseContextObject {
public:
  PyAffineExpr(PyAiirContextRef contextRef, AiirAffineExpr affineExpr)
      : BaseContextObject(std::move(contextRef)), affineExpr(affineExpr) {}
  bool operator==(const PyAffineExpr &other) const;
  operator AiirAffineExpr() const { return affineExpr; }
  AiirAffineExpr get() const { return affineExpr; }

  /// Gets a capsule wrapping the void* within the AiirAffineExpr.
  nanobind::object getCapsule();

  /// Creates a PyAffineExpr from the AiirAffineExpr wrapped by a capsule.
  /// Note that PyAffineExpr instances are uniqued, so the returned object
  /// may be a pre-existing object. Ownership of the underlying AiirAffineExpr
  /// is taken by calling this function.
  static PyAffineExpr createFromCapsule(const nanobind::object &capsule);

  PyAffineExpr add(const PyAffineExpr &other) const;
  PyAffineExpr mul(const PyAffineExpr &other) const;
  PyAffineExpr floorDiv(const PyAffineExpr &other) const;
  PyAffineExpr ceilDiv(const PyAffineExpr &other) const;
  PyAffineExpr mod(const PyAffineExpr &other) const;

  nanobind::typed<nanobind::object, PyAffineExpr> maybeDownCast();

private:
  AiirAffineExpr affineExpr;
};

class AIIR_PYTHON_API_EXPORTED PyAffineMap : public BaseContextObject {
public:
  PyAffineMap(PyAiirContextRef contextRef, AiirAffineMap affineMap)
      : BaseContextObject(std::move(contextRef)), affineMap(affineMap) {}
  bool operator==(const PyAffineMap &other) const;
  operator AiirAffineMap() const { return affineMap; }
  AiirAffineMap get() const { return affineMap; }

  /// Gets a capsule wrapping the void* within the AiirAffineMap.
  nanobind::object getCapsule();

  /// Creates a PyAffineMap from the AiirAffineMap wrapped by a capsule.
  /// Note that PyAffineMap instances are uniqued, so the returned object
  /// may be a pre-existing object. Ownership of the underlying AiirAffineMap
  /// is taken by calling this function.
  static PyAffineMap createFromCapsule(const nanobind::object &capsule);

private:
  AiirAffineMap affineMap;
};

class AIIR_PYTHON_API_EXPORTED PyIntegerSet : public BaseContextObject {
public:
  PyIntegerSet(PyAiirContextRef contextRef, AiirIntegerSet integerSet)
      : BaseContextObject(std::move(contextRef)), integerSet(integerSet) {}
  bool operator==(const PyIntegerSet &other) const;
  operator AiirIntegerSet() const { return integerSet; }
  AiirIntegerSet get() const { return integerSet; }

  /// Gets a capsule wrapping the void* within the AiirIntegerSet.
  nanobind::object getCapsule();

  /// Creates a PyIntegerSet from the AiirAffineMap wrapped by a capsule.
  /// Note that PyIntegerSet instances may be uniqued, so the returned object
  /// may be a pre-existing object. Integer sets are owned by the context.
  static PyIntegerSet createFromCapsule(const nanobind::object &capsule);

private:
  AiirIntegerSet integerSet;
};

/// Bindings for AIIR symbol tables.
class AIIR_PYTHON_API_EXPORTED PySymbolTable {
public:
  /// Constructs a symbol table for the given operation.
  explicit PySymbolTable(PyOperationBase &operation);

  /// Destroys the symbol table.
  ~PySymbolTable() { aiirSymbolTableDestroy(symbolTable); }

  /// Returns the symbol (opview) with the given name, throws if there is no
  /// such symbol in the table.
  nanobind::object dunderGetItem(const std::string &name);

  /// Removes the given operation from the symbol table and erases it.
  void erase(PyOperationBase &symbol);

  /// Removes the operation with the given name from the symbol table and erases
  /// it, throws if there is no such symbol in the table.
  void dunderDel(const std::string &name);

  /// Inserts the given operation into the symbol table. The operation must have
  /// the symbol trait.
  PyStringAttribute insert(PyOperationBase &symbol);

  /// Gets and sets the name of a symbol op.
  static PyStringAttribute getSymbolName(PyOperationBase &symbol);
  static void setSymbolName(PyOperationBase &symbol, const std::string &name);

  /// Gets and sets the visibility of a symbol op.
  static PyStringAttribute getVisibility(PyOperationBase &symbol);
  static void setVisibility(PyOperationBase &symbol,
                            const std::string &visibility);

  /// Replaces all symbol uses within an operation. See the API
  /// aiirSymbolTableReplaceAllSymbolUses for all caveats.
  static void replaceAllSymbolUses(const std::string &oldSymbol,
                                   const std::string &newSymbol,
                                   PyOperationBase &from);

  /// Walks all symbol tables under and including 'from'.
  static void walkSymbolTables(PyOperationBase &from, bool allSymUsesVisible,
                               nanobind::object callback);

  /// Casts the bindings class into the C API structure.
  operator AiirSymbolTable() { return symbolTable; }

private:
  PyOperationRef operation;
  AiirSymbolTable symbolTable;
};

/// Custom exception that allows access to error diagnostic information. This is
/// translated to the `ir.AIIRError` python exception when thrown.
struct AIIR_PYTHON_API_EXPORTED AIIRError : std::exception {
  AIIRError(std::string message,
            std::vector<PyDiagnostic::DiagnosticInfo> &&errorDiagnostics = {})
      : message(std::move(message)),
        errorDiagnostics(std::move(errorDiagnostics)) {}
  const char *what() const noexcept override { return message.c_str(); }

  /// Bind the AIIRError exception class to the given module.
  static void bind(nanobind::module_ &m);

  std::string message;
  std::vector<PyDiagnostic::DiagnosticInfo> errorDiagnostics;
};

//------------------------------------------------------------------------------
// Utilities.
//------------------------------------------------------------------------------

inline AiirStringRef toAiirStringRef(const std::string &s) {
  return aiirStringRefCreate(s.data(), s.size());
}

inline AiirStringRef toAiirStringRef(std::string_view s) {
  return aiirStringRefCreate(s.data(), s.size());
}

inline AiirStringRef toAiirStringRef(const nanobind::bytes &s) {
  return aiirStringRefCreate(static_cast<const char *>(s.data()), s.size());
}

/// Create a block, using the current location context if no locations are
/// specified.
AiirBlock AIIR_PYTHON_API_EXPORTED
createBlock(const nanobind::typed<nanobind::sequence, PyType> &pyArgTypes,
            const std::optional<nanobind::typed<nanobind::sequence, PyLocation>>
                &pyArgLocs);

struct AIIR_PYTHON_API_EXPORTED PyAttrBuilderMap {
  static bool dunderContains(const std::string &attributeKind);
  static nanobind::callable
  dunderGetItemNamed(const std::string &attributeKind);
  static void dunderSetItemNamed(const std::string &attributeKind,
                                 nanobind::callable func, bool replace,
                                 bool allow_existing);

  static void bind(nanobind::module_ &m);
};

//------------------------------------------------------------------------------
// Collections.
//------------------------------------------------------------------------------

/// Regions of an op are fixed length and indexed numerically so are represented
/// with a sequence-like container.
class AIIR_PYTHON_API_EXPORTED PyRegionList
    : public Sliceable<PyRegionList, PyRegion> {
public:
  static constexpr const char *pyClassName = "RegionSequence";

  PyRegionList(PyOperationRef operation, intptr_t startIndex = 0,
               intptr_t length = -1, intptr_t step = 1);

private:
  /// Give the parent CRTP class access to hook implementations below.
  friend class Sliceable<PyRegionList, PyRegion>;

  intptr_t getRawNumElements();

  PyRegion getRawElement(intptr_t pos);

  PyRegionList slice(intptr_t startIndex, intptr_t length, intptr_t step) const;

  PyOperationRef operation;
};

class AIIR_PYTHON_API_EXPORTED PyBlockIterator {
public:
  PyBlockIterator(PyOperationRef operation, AiirBlock next)
      : operation(std::move(operation)), next(next) {}

  PyBlockIterator &dunderIter() { return *this; }

  nanobind::typed<nanobind::object, PyBlock> dunderNext();

  static void bind(nanobind::module_ &m);

private:
  PyOperationRef operation;
  AiirBlock next;
};

/// Blocks are exposed by the C-API as a forward-only linked list. In Python,
/// we present them as a more full-featured list-like container but optimize
/// it for forward iteration. Blocks are always owned by a region.
class AIIR_PYTHON_API_EXPORTED PyBlockList {
public:
  PyBlockList(PyOperationRef operation, AiirRegion region)
      : operation(std::move(operation)), region(region) {}

  PyBlockIterator dunderIter();

  intptr_t dunderLen();

  PyBlock dunderGetItem(intptr_t index);

  PyBlock appendBlock(const nanobind::args &pyArgTypes,
                      const std::optional<nanobind::sequence> &pyArgLocs);

  static void bind(nanobind::module_ &m);

private:
  PyOperationRef operation;
  AiirRegion region;
};

class AIIR_PYTHON_API_EXPORTED PyOperationIterator {
public:
  PyOperationIterator(PyOperationRef parentOperation, AiirOperation next)
      : parentOperation(std::move(parentOperation)), next(next) {}

  PyOperationIterator &dunderIter() { return *this; }

  nanobind::typed<nanobind::object, PyOpView> dunderNext();

  static void bind(nanobind::module_ &m);

private:
  PyOperationRef parentOperation;
  AiirOperation next;
};

/// Operations are exposed by the C-API as a forward-only linked list. In
/// Python, we present them as a more full-featured list-like container but
/// optimize it for forward iteration. Iterable operations are always owned
/// by a block.
class AIIR_PYTHON_API_EXPORTED PyOperationList {
public:
  PyOperationList(PyOperationRef parentOperation, AiirBlock block)
      : parentOperation(std::move(parentOperation)), block(block) {}

  PyOperationIterator dunderIter();

  intptr_t dunderLen();

  nanobind::typed<nanobind::object, PyOpView> dunderGetItem(intptr_t index);

  static void bind(nanobind::module_ &m);

private:
  PyOperationRef parentOperation;
  AiirBlock block;
};

class AIIR_PYTHON_API_EXPORTED PyOpOperand {
public:
  PyOpOperand(AiirOpOperand opOperand) : opOperand(opOperand) {}
  operator AiirOpOperand() const { return opOperand; }

  nanobind::typed<nanobind::object, PyOpView> getOwner() const;

  size_t getOperandNumber() const;

  static void bind(nanobind::module_ &m);

private:
  AiirOpOperand opOperand;
};

class AIIR_PYTHON_API_EXPORTED PyOpOperandIterator {
public:
  PyOpOperandIterator(AiirOpOperand opOperand) : opOperand(opOperand) {}

  PyOpOperandIterator &dunderIter() { return *this; }

  nanobind::typed<nanobind::object, PyOpOperand> dunderNext();

  static void bind(nanobind::module_ &m);

private:
  AiirOpOperand opOperand;
};

/// CRTP base class for Python AIIR values that subclass Value and should be
/// castable from it. The value hierarchy is one level deep and is not supposed
/// to accommodate other levels unless core AIIR changes.
template <typename DerivedTy>
class AIIR_PYTHON_API_EXPORTED PyConcreteValue : public PyValue {
public:
  // Derived classes must define statics for:
  //   IsAFunctionTy isaFunction
  //   const char *pyClassName
  // and redefine bindDerived.
  using ClassTy = nanobind::class_<DerivedTy, PyValue>;
  using IsAFunctionTy = bool (*)(AiirValue);
  using GetTypeIDFunctionTy = AiirTypeID (*)();
  static constexpr GetTypeIDFunctionTy getTypeIdFunction = nullptr;
  using Base = PyConcreteValue;

  PyConcreteValue() = default;
  PyConcreteValue(PyOperationRef operationRef, AiirValue value)
      : PyValue(operationRef, value) {}
  PyConcreteValue(PyValue &orig)
      : PyConcreteValue(orig.getParentOperation(), castFrom(orig)) {}

  /// Attempts to cast the original value to the derived type and throws on
  /// type mismatches.
  static AiirValue castFrom(PyValue &orig) {
    if (!DerivedTy::isaFunction(orig.get())) {
      auto origRepr =
          nanobind::cast<std::string>(nanobind::repr(nanobind::cast(orig)));
      throw nanobind::value_error((std::string("Cannot cast value to ") +
                                   DerivedTy::pyClassName + " (from " +
                                   origRepr + ")")
                                      .c_str());
    }
    return orig.get();
  }

  /// Binds the Python module objects to functions of this class.
  static void bind(nanobind::module_ &m) {
    auto cls = ClassTy(m, DerivedTy::pyClassName, nanobind::is_generic(),
                       nanobind::sig((std::string("class ") +
                                      DerivedTy::pyClassName + "(Value[_T])")
                                         .c_str()));
    cls.def(nanobind::init<PyValue &>(), nanobind::keep_alive<0, 1>(),
            nanobind::arg("value"));
    cls.def(
        AIIR_PYTHON_MAYBE_DOWNCAST_ATTR,
        [](DerivedTy &self) -> nanobind::typed<nanobind::object, DerivedTy> {
          return self.maybeDownCast();
        });
    cls.def("__str__", [](PyValue &self) {
      PyPrintAccumulator printAccum;
      printAccum.parts.append(std::string(DerivedTy::pyClassName) + "(");
      aiirValuePrint(self.get(), printAccum.getCallback(),
                     printAccum.getUserData());
      printAccum.parts.append(")");
      return printAccum.join();
    });

    if (DerivedTy::getTypeIdFunction) {
      PyGlobals::get().registerValueCaster(
          DerivedTy::getTypeIdFunction(),
          nanobind::cast<nanobind::callable>(nanobind::cpp_function(
              [](PyValue pyValue) -> DerivedTy { return pyValue; })),
          /*replace*/ true);
    }

    DerivedTy::bindDerived(cls);
  }

  /// Implemented by derived classes to add methods to the Python subclass.
  static void bindDerived(ClassTy &m) {}
};

/// Python wrapper for AiirOpResult.
class AIIR_PYTHON_API_EXPORTED PyOpResult : public PyConcreteValue<PyOpResult> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirValueIsAOpResult;
  static constexpr const char *pyClassName = "OpResult";
  using PyConcreteValue::PyConcreteValue;

  static void bindDerived(ClassTy &c);
};

/// A list of operation results. Internally, these are stored as consecutive
/// elements, random access is cheap. The (returned) result list is associated
/// with the operation whose results these are, and thus extends the lifetime of
/// this operation.
class AIIR_PYTHON_API_EXPORTED PyOpResultList
    : public Sliceable<PyOpResultList, PyOpResult> {
public:
  static constexpr const char *pyClassName = "OpResultList";
  static constexpr std::array<const char *, 1> typeParams = {"_T"};
  using SliceableT = Sliceable<PyOpResultList, PyOpResult>;

  PyOpResultList(PyOperationRef operation, intptr_t startIndex = 0,
                 intptr_t length = -1, intptr_t step = 1);

  static void bindDerived(ClassTy &c);

  PyOperationRef &getOperation() { return operation; }

private:
  /// Give the parent CRTP class access to hook implementations below.
  friend class Sliceable<PyOpResultList, PyOpResult>;

  intptr_t getRawNumElements();

  PyOpResult getRawElement(intptr_t index);

  PyOpResultList slice(intptr_t startIndex, intptr_t length,
                       intptr_t step) const;

  PyOperationRef operation;
};

/// Python wrapper for AiirBlockArgument.
class AIIR_PYTHON_API_EXPORTED PyBlockArgument
    : public PyConcreteValue<PyBlockArgument> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirValueIsABlockArgument;
  static constexpr const char *pyClassName = "BlockArgument";
  using PyConcreteValue::PyConcreteValue;

  static void bindDerived(ClassTy &c);
};

/// A list of block arguments. Internally, these are stored as consecutive
/// elements, random access is cheap. The argument list is associated with the
/// operation that contains the block (detached blocks are not allowed in
/// Python bindings) and extends its lifetime.
class AIIR_PYTHON_API_EXPORTED PyBlockArgumentList
    : public Sliceable<PyBlockArgumentList, PyBlockArgument> {
public:
  static constexpr const char *pyClassName = "BlockArgumentList";
  using SliceableT = Sliceable<PyBlockArgumentList, PyBlockArgument>;

  PyBlockArgumentList(PyOperationRef operation, AiirBlock block,
                      intptr_t startIndex = 0, intptr_t length = -1,
                      intptr_t step = 1);

  static void bindDerived(ClassTy &c);

private:
  /// Give the parent CRTP class access to hook implementations below.
  friend class Sliceable<PyBlockArgumentList, PyBlockArgument>;

  /// Returns the number of arguments in the list.
  intptr_t getRawNumElements();

  /// Returns `pos`-the element in the list.
  PyBlockArgument getRawElement(intptr_t pos) const;

  /// Returns a sublist of this list.
  PyBlockArgumentList slice(intptr_t startIndex, intptr_t length,
                            intptr_t step) const;

  PyOperationRef operation;
  AiirBlock block;
};

/// A list of operation operands. Internally, these are stored as consecutive
/// elements, random access is cheap. The (returned) operand list is associated
/// with the operation whose operands these are, and thus extends the lifetime
/// of this operation.
class AIIR_PYTHON_API_EXPORTED PyOpOperandList
    : public Sliceable<PyOpOperandList, PyValue> {
public:
  static constexpr const char *pyClassName = "OpOperandList";
  static constexpr std::array<const char *, 1> typeParams = {"_T"};
  using SliceableT = Sliceable<PyOpOperandList, PyValue>;

  PyOpOperandList(PyOperationRef operation, intptr_t startIndex = 0,
                  intptr_t length = -1, intptr_t step = 1);

  void dunderSetItem(intptr_t index, PyValue value);

  static void bindDerived(ClassTy &c);

private:
  /// Give the parent CRTP class access to hook implementations below.
  friend class Sliceable<PyOpOperandList, PyValue>;

  intptr_t getRawNumElements();

  PyValue getRawElement(intptr_t pos);

  PyOpOperandList slice(intptr_t startIndex, intptr_t length,
                        intptr_t step) const;

  PyOperationRef operation;
};

/// A list of operation successors. Internally, these are stored as consecutive
/// elements, random access is cheap. The (returned) successor list is
/// associated with the operation whose successors these are, and thus extends
/// the lifetime of this operation.
class AIIR_PYTHON_API_EXPORTED PyOpSuccessors
    : public Sliceable<PyOpSuccessors, PyBlock> {
public:
  static constexpr const char *pyClassName = "OpSuccessors";

  PyOpSuccessors(PyOperationRef operation, intptr_t startIndex = 0,
                 intptr_t length = -1, intptr_t step = 1);

  void dunderSetItem(intptr_t index, PyBlock block);

  static void bindDerived(ClassTy &c);

private:
  /// Give the parent CRTP class access to hook implementations below.
  friend class Sliceable<PyOpSuccessors, PyBlock>;

  intptr_t getRawNumElements();

  PyBlock getRawElement(intptr_t pos);

  PyOpSuccessors slice(intptr_t startIndex, intptr_t length,
                       intptr_t step) const;

  PyOperationRef operation;
};

/// A list of block successors. Internally, these are stored as consecutive
/// elements, random access is cheap. The (returned) successor list is
/// associated with the operation and block whose successors these are, and thus
/// extends the lifetime of this operation and block.
class AIIR_PYTHON_API_EXPORTED PyBlockSuccessors
    : public Sliceable<PyBlockSuccessors, PyBlock> {
public:
  static constexpr const char *pyClassName = "BlockSuccessors";

  PyBlockSuccessors(PyBlock block, PyOperationRef operation,
                    intptr_t startIndex = 0, intptr_t length = -1,
                    intptr_t step = 1);

private:
  /// Give the parent CRTP class access to hook implementations below.
  friend class Sliceable<PyBlockSuccessors, PyBlock>;

  intptr_t getRawNumElements();

  PyBlock getRawElement(intptr_t pos);

  PyBlockSuccessors slice(intptr_t startIndex, intptr_t length,
                          intptr_t step) const;

  PyOperationRef operation;
  PyBlock block;
};

/// A list of block predecessors. The (returned) predecessor list is
/// associated with the operation and block whose predecessors these are, and
/// thus extends the lifetime of this operation and block.
///
/// WARNING: This Sliceable is more expensive than the others here because
/// aiirBlockGetPredecessor actually iterates the use-def chain (of block
/// operands) anew for each indexed access.
class AIIR_PYTHON_API_EXPORTED PyBlockPredecessors
    : public Sliceable<PyBlockPredecessors, PyBlock> {
public:
  static constexpr const char *pyClassName = "BlockPredecessors";

  PyBlockPredecessors(PyBlock block, PyOperationRef operation,
                      intptr_t startIndex = 0, intptr_t length = -1,
                      intptr_t step = 1);

private:
  /// Give the parent CRTP class access to hook implementations below.
  friend class Sliceable<PyBlockPredecessors, PyBlock>;

  intptr_t getRawNumElements();

  PyBlock getRawElement(intptr_t pos);

  PyBlockPredecessors slice(intptr_t startIndex, intptr_t length,
                            intptr_t step) const;

  PyOperationRef operation;
  PyBlock block;
};

/// A list of operation attributes. Can be indexed by name, producing
/// attributes, or by index, producing named attributes.
class AIIR_PYTHON_API_EXPORTED PyOpAttributeMap {
public:
  PyOpAttributeMap(PyOperationRef operation)
      : operation(std::move(operation)) {}

  nanobind::typed<nanobind::object, PyAttribute>
  dunderGetItemNamed(const std::string &name);

  PyNamedAttribute dunderGetItemIndexed(intptr_t index);

  nanobind::typed<nanobind::object, std::optional<PyAttribute>>
  get(const std::string &key, nanobind::object defaultValue);

  void dunderSetItem(const std::string &name, const PyAttribute &attr);

  void dunderDelItem(const std::string &name);

  intptr_t dunderLen();

  bool dunderContains(const std::string &name);

  static void forEachAttr(AiirOperation op,
                          std::function<void(AiirStringRef, AiirAttribute)> fn);

  static void bind(nanobind::module_ &m);

private:
  PyOperationRef operation;
};

/// Base class of operation adaptors.
class AIIR_PYTHON_API_EXPORTED PyOpAdaptor {
public:
  PyOpAdaptor(nanobind::list operands, PyOpAttributeMap attributes)
      : operands(std::move(operands)), attributes(std::move(attributes)) {}
  PyOpAdaptor(nanobind::list operands, PyOpView &opView)
      : operands(std::move(operands)),
        attributes(opView.getOperation().getRef()) {}

  static void bind(nanobind::module_ &m);

private:
  nanobind::list operands;
  PyOpAttributeMap attributes;
};

class AIIR_PYTHON_API_EXPORTED PyDynamicOpTrait {
public:
  static bool attach(const nanobind::object &opName,
                     const nanobind::object &target, PyAiirContext &context);

  static void bind(nanobind::module_ &m);

  static inline const char *typeIDAttr = "_trait_typeid";
};

namespace PyDynamicOpTraits {

class AIIR_PYTHON_API_EXPORTED IsTerminator : public PyDynamicOpTrait {
public:
  static bool attach(const nanobind::object &opName, PyAiirContext &context);
  static void bind(nanobind::module_ &m);
};

class AIIR_PYTHON_API_EXPORTED NoTerminator : public PyDynamicOpTrait {
public:
  static bool attach(const nanobind::object &opName, PyAiirContext &context);
  static void bind(nanobind::module_ &m);
};

} // namespace PyDynamicOpTraits

AIIR_PYTHON_API_EXPORTED AiirValue getUniqueResult(AiirOperation operation);
AIIR_PYTHON_API_EXPORTED void populateIRCore(nanobind::module_ &m);
AIIR_PYTHON_API_EXPORTED void populateRoot(nanobind::module_ &m);

/// Helper for creating an @classmethod.
template <class Func, typename... Args>
inline nanobind::object classmethod(Func f, Args... args) {
  nanobind::object cf = nanobind::cpp_function(f, args...);
  static SafeInit<nanobind::object> classmethodFn([]() {
    return std::make_unique<nanobind::object>(
        nanobind::module_::import_("builtins").attr("classmethod"));
  });
  return classmethodFn.get()(cf);
}

} // namespace AIIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace aiir

namespace nanobind {
namespace detail {
template <>
struct type_caster<
    aiir::python::AIIR_BINDINGS_PYTHON_DOMAIN::DefaultingPyAiirContext>
    : AiirDefaultingCaster<
          aiir::python::AIIR_BINDINGS_PYTHON_DOMAIN::DefaultingPyAiirContext> {
};
template <>
struct type_caster<
    aiir::python::AIIR_BINDINGS_PYTHON_DOMAIN::DefaultingPyLocation>
    : AiirDefaultingCaster<
          aiir::python::AIIR_BINDINGS_PYTHON_DOMAIN::DefaultingPyLocation> {};

} // namespace detail
} // namespace nanobind

#endif // AIIR_BINDINGS_PYTHON_IRCORE_H
