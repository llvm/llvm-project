//===- IRModules.cpp - IR Submodules of pybind module ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// clang-format off
#include "aiir/Bindings/Python/Globals.h"
#include "aiir/Bindings/Python/IRCore.h"
#include "aiir/Bindings/Python/NanobindUtils.h"
#include "aiir-c/Bindings/Python/Interop.h" // This is expected after nanobind.
// clang-format on
#include "aiir-c/BuiltinAttributes.h"
#include "aiir-c/Debug.h"
#include "aiir-c/Diagnostics.h"
#include "aiir-c/ExtensibleDialect.h"
#include "aiir-c/IR.h"
#include "aiir-c/Support.h"

#include <array>
#include <functional>
#include <optional>
#include <string>

namespace nb = nanobind;
using namespace nb::literals;
using namespace aiir;
using nanobind::detail::join;

static const char kModuleParseDocstring[] =
    R"(Parses a module's assembly format from a string.

Returns a new AiirModule or raises an AIIRError if the parsing fails.

See also: https://aiir.llvm.org/docs/LangRef/
)";

static const char kDumpDocstring[] =
    "Dumps a debug representation of the object to stderr.";

static const char kValueReplaceAllUsesExceptDocstring[] =
    R"(Replace all uses of this value with the `with` value, except for those
in `exceptions`. `exceptions` can be either a single operation or a list of
operations.
)";

//------------------------------------------------------------------------------
// Utilities.
//------------------------------------------------------------------------------

/// Local helper to compute std::hash for a value.
template <typename T>
static size_t hash(const T &value) {
  return std::hash<T>{}(value);
}

static nb::object
createCustomDialectWrapper(const std::string &dialectNamespace,
                           nb::object dialectDescriptor) {
  auto dialectClass =
      python::AIIR_BINDINGS_PYTHON_DOMAIN::PyGlobals::get().lookupDialectClass(
          dialectNamespace);
  if (!dialectClass) {
    // Use the base class.
    return nb::cast(python::AIIR_BINDINGS_PYTHON_DOMAIN::PyDialect(
        std::move(dialectDescriptor)));
  }

  // Create the custom implementation.
  return (*dialectClass)(std::move(dialectDescriptor));
}

namespace aiir {
namespace python {
namespace AIIR_BINDINGS_PYTHON_DOMAIN {

AiirBlock createBlock(
    const nb::typed<nb::sequence, PyType> &pyArgTypes,
    const std::optional<nb::typed<nb::sequence, PyLocation>> &pyArgLocs) {
  std::vector<AiirType> argTypes;
  argTypes.reserve(nb::len(pyArgTypes));
  for (nb::handle pyType : pyArgTypes)
    argTypes.push_back(
        nb::cast<python::AIIR_BINDINGS_PYTHON_DOMAIN::PyType &>(pyType));

  std::vector<AiirLocation> argLocs;
  if (pyArgLocs) {
    argLocs.reserve(nb::len(*pyArgLocs));
    for (nb::handle pyLoc : *pyArgLocs)
      argLocs.push_back(
          nb::cast<python::AIIR_BINDINGS_PYTHON_DOMAIN::PyLocation &>(pyLoc));
  } else if (!argTypes.empty()) {
    argLocs.assign(
        argTypes.size(),
        python::AIIR_BINDINGS_PYTHON_DOMAIN::DefaultingPyLocation::resolve());
  }

  if (argTypes.size() != argLocs.size())
    throw nb::value_error(
        join("Expected ", argTypes.size(), " locations, got: ", argLocs.size())
            .c_str());
  return aiirBlockCreate(argTypes.size(), argTypes.data(), argLocs.data());
}

void PyGlobalDebugFlag::set(nb::object &o, bool enable) {
  nb::ft_lock_guard lock(mutex);
  aiirEnableGlobalDebug(enable);
}

bool PyGlobalDebugFlag::get(const nb::object &) {
  nb::ft_lock_guard lock(mutex);
  return aiirIsGlobalDebugEnabled();
}

void PyGlobalDebugFlag::bind(nb::module_ &m) {
  // Debug flags.
  nb::class_<PyGlobalDebugFlag>(m, "_GlobalDebug")
      .def_prop_rw_static("flag", &PyGlobalDebugFlag::get,
                          &PyGlobalDebugFlag::set, "LLVM-wide debug flag.")
      .def_static(
          "set_types",
          [](const std::string &type) {
            nb::ft_lock_guard lock(mutex);
            aiirSetGlobalDebugType(type.c_str());
          },
          "types"_a, "Sets specific debug types to be produced by LLVM.")
      .def_static(
          "set_types",
          [](const std::vector<std::string> &types) {
            std::vector<const char *> pointers;
            pointers.reserve(types.size());
            for (const std::string &str : types)
              pointers.push_back(str.c_str());
            nb::ft_lock_guard lock(mutex);
            aiirSetGlobalDebugTypes(pointers.data(), pointers.size());
          },
          "types"_a,
          "Sets multiple specific debug types to be produced by LLVM.");
}

nb::ft_mutex PyGlobalDebugFlag::mutex;

bool PyAttrBuilderMap::dunderContains(const std::string &attributeKind) {
  return PyGlobals::get().lookupAttributeBuilder(attributeKind).has_value();
}

nb::callable
PyAttrBuilderMap::dunderGetItemNamed(const std::string &attributeKind) {
  auto builder = PyGlobals::get().lookupAttributeBuilder(attributeKind);
  if (!builder)
    throw nb::key_error(attributeKind.c_str());
  return *builder;
}

void PyAttrBuilderMap::dunderSetItemNamed(const std::string &attributeKind,
                                          nb::callable func, bool replace,
                                          bool allow_existing) {
  PyGlobals::get().registerAttributeBuilder(attributeKind, std::move(func),
                                            replace, allow_existing);
}

void PyAttrBuilderMap::bind(nb::module_ &m) {
  nb::class_<PyAttrBuilderMap>(m, "AttrBuilder")
      .def_static("contains", &PyAttrBuilderMap::dunderContains,
                  "attribute_kind"_a,
                  "Checks whether an attribute builder is registered for the "
                  "given attribute kind.")
      .def_static("get", &PyAttrBuilderMap::dunderGetItemNamed,
                  "attribute_kind"_a,
                  "Gets the registered attribute builder for the given "
                  "attribute kind.")
      .def_static("insert", &PyAttrBuilderMap::dunderSetItemNamed,
                  "attribute_kind"_a, "attr_builder"_a, "replace"_a = false,
                  "allow_existing"_a = false,
                  "Register an attribute builder for building AIIR "
                  "attributes from Python values.");
}

//------------------------------------------------------------------------------
// PyBlock
//------------------------------------------------------------------------------

nb::object PyBlock::getCapsule() {
  return nb::steal<nb::object>(aiirPythonBlockToCapsule(get()));
}

//------------------------------------------------------------------------------
// Collections.
//------------------------------------------------------------------------------

PyRegionList::PyRegionList(PyOperationRef operation, intptr_t startIndex,
                           intptr_t length, intptr_t step)
    : Sliceable(startIndex,
                length == -1 ? aiirOperationGetNumRegions(operation->get())
                             : length,
                step),
      operation(std::move(operation)) {}

intptr_t PyRegionList::getRawNumElements() {
  operation->checkValid();
  return aiirOperationGetNumRegions(operation->get());
}

PyRegion PyRegionList::getRawElement(intptr_t pos) {
  operation->checkValid();
  return PyRegion(operation, aiirOperationGetRegion(operation->get(), pos));
}

PyRegionList PyRegionList::slice(intptr_t startIndex, intptr_t length,
                                 intptr_t step) const {
  return PyRegionList(operation, startIndex, length, step);
}

nb::typed<nb::object, PyBlock> PyBlockIterator::dunderNext() {
  operation->checkValid();
  if (aiirBlockIsNull(next)) {
    PyErr_SetNone(PyExc_StopIteration);
    // python functions should return NULL after setting any exception
    return nb::object();
  }

  PyBlock returnBlock(operation, next);
  next = aiirBlockGetNextInRegion(next);
  return nb::cast(returnBlock);
}

void PyBlockIterator::bind(nb::module_ &m) {
  nb::class_<PyBlockIterator>(m, "BlockIterator")
      .def("__iter__", &PyBlockIterator::dunderIter,
           "Returns an iterator over the blocks in the operation's region.")
      .def("__next__", &PyBlockIterator::dunderNext,
           "Returns the next block in the iteration.");
}

PyBlockIterator PyBlockList::dunderIter() {
  operation->checkValid();
  return PyBlockIterator(operation, aiirRegionGetFirstBlock(region));
}

intptr_t PyBlockList::dunderLen() {
  operation->checkValid();
  intptr_t count = 0;
  AiirBlock block = aiirRegionGetFirstBlock(region);
  while (!aiirBlockIsNull(block)) {
    count += 1;
    block = aiirBlockGetNextInRegion(block);
  }
  return count;
}

PyBlock PyBlockList::dunderGetItem(intptr_t index) {
  operation->checkValid();
  if (index < 0) {
    index += dunderLen();
  }
  if (index < 0) {
    throw nb::index_error("attempt to access out of bounds block");
  }
  AiirBlock block = aiirRegionGetFirstBlock(region);
  while (!aiirBlockIsNull(block)) {
    if (index == 0) {
      return PyBlock(operation, block);
    }
    block = aiirBlockGetNextInRegion(block);
    index -= 1;
  }
  throw nb::index_error("attempt to access out of bounds block");
}

PyBlock PyBlockList::appendBlock(const nb::args &pyArgTypes,
                                 const std::optional<nb::sequence> &pyArgLocs) {
  operation->checkValid();
  AiirBlock block = createBlock(nb::cast<nb::sequence>(pyArgTypes), pyArgLocs);
  aiirRegionAppendOwnedBlock(region, block);
  return PyBlock(operation, block);
}

void PyBlockList::bind(nb::module_ &m) {
  nb::class_<PyBlockList>(m, "BlockList")
      .def("__getitem__", &PyBlockList::dunderGetItem,
           "Returns the block at the specified index.")
      .def("__iter__", &PyBlockList::dunderIter,
           "Returns an iterator over blocks in the operation's region.")
      .def("__len__", &PyBlockList::dunderLen,
           "Returns the number of blocks in the operation's region.")
      .def("append", &PyBlockList::appendBlock,
           R"(
              Appends a new block, with argument types as positional args.

              Returns:
                The created block.
             )",
           "args"_a, nb::kw_only(), "arg_locs"_a = std::nullopt);
}

nb::typed<nb::object, PyOpView> PyOperationIterator::dunderNext() {
  parentOperation->checkValid();
  if (aiirOperationIsNull(next)) {
    PyErr_SetNone(PyExc_StopIteration);
    // python functions should return NULL after setting any exception
    return nb::object();
  }

  PyOperationRef returnOperation =
      PyOperation::forOperation(parentOperation->getContext(), next);
  next = aiirOperationGetNextInBlock(next);
  return returnOperation->createOpView();
}

void PyOperationIterator::bind(nb::module_ &m) {
  nb::class_<PyOperationIterator>(m, "OperationIterator")
      .def("__iter__", &PyOperationIterator::dunderIter,
           "Returns an iterator over the operations in an operation's block.")
      .def("__next__", &PyOperationIterator::dunderNext,
           "Returns the next operation in the iteration.");
}

PyOperationIterator PyOperationList::dunderIter() {
  parentOperation->checkValid();
  return PyOperationIterator(parentOperation,
                             aiirBlockGetFirstOperation(block));
}

intptr_t PyOperationList::dunderLen() {
  parentOperation->checkValid();
  intptr_t count = 0;
  AiirOperation childOp = aiirBlockGetFirstOperation(block);
  while (!aiirOperationIsNull(childOp)) {
    count += 1;
    childOp = aiirOperationGetNextInBlock(childOp);
  }
  return count;
}

nb::typed<nb::object, PyOpView> PyOperationList::dunderGetItem(intptr_t index) {
  parentOperation->checkValid();
  if (index < 0) {
    index += dunderLen();
  }
  if (index < 0) {
    throw nb::index_error("attempt to access out of bounds operation");
  }
  AiirOperation childOp = aiirBlockGetFirstOperation(block);
  while (!aiirOperationIsNull(childOp)) {
    if (index == 0) {
      return PyOperation::forOperation(parentOperation->getContext(), childOp)
          ->createOpView();
    }
    childOp = aiirOperationGetNextInBlock(childOp);
    index -= 1;
  }
  throw nb::index_error("attempt to access out of bounds operation");
}

void PyOperationList::bind(nb::module_ &m) {
  nb::class_<PyOperationList>(m, "OperationList")
      .def("__getitem__", &PyOperationList::dunderGetItem,
           "Returns the operation at the specified index.")
      .def("__iter__", &PyOperationList::dunderIter,
           "Returns an iterator over operations in the list.")
      .def("__len__", &PyOperationList::dunderLen,
           "Returns the number of operations in the list.");
}

nb::typed<nb::object, PyOpView> PyOpOperand::getOwner() const {
  AiirOperation owner = aiirOpOperandGetOwner(opOperand);
  PyAiirContextRef context =
      PyAiirContext::forContext(aiirOperationGetContext(owner));
  return PyOperation::forOperation(context, owner)->createOpView();
}

size_t PyOpOperand::getOperandNumber() const {
  return aiirOpOperandGetOperandNumber(opOperand);
}

void PyOpOperand::bind(nb::module_ &m) {
  nb::class_<PyOpOperand>(m, "OpOperand")
      .def_prop_ro("owner", &PyOpOperand::getOwner,
                   "Returns the operation that owns this operand.")
      .def_prop_ro("operand_number", &PyOpOperand::getOperandNumber,
                   "Returns the operand number in the owning operation.");
}

nb::typed<nb::object, PyOpOperand> PyOpOperandIterator::dunderNext() {
  if (aiirOpOperandIsNull(opOperand)) {
    PyErr_SetNone(PyExc_StopIteration);
    // python functions should return NULL after setting any exception
    return nb::object();
  }

  PyOpOperand returnOpOperand(opOperand);
  opOperand = aiirOpOperandGetNextUse(opOperand);
  return nb::cast(returnOpOperand);
}

void PyOpOperandIterator::bind(nb::module_ &m) {
  nb::class_<PyOpOperandIterator>(m, "OpOperandIterator")
      .def("__iter__", &PyOpOperandIterator::dunderIter,
           "Returns an iterator over operands.")
      .def("__next__", &PyOpOperandIterator::dunderNext,
           "Returns the next operand in the iteration.");
}

//------------------------------------------------------------------------------
// PyThreadPool
//------------------------------------------------------------------------------

PyThreadPool::PyThreadPool() { threadPool = aiirLlvmThreadPoolCreate(); }

PyThreadPool::~PyThreadPool() {
  if (threadPool.ptr)
    aiirLlvmThreadPoolDestroy(threadPool);
}

int PyThreadPool::getMaxConcurrency() const {
  return aiirLlvmThreadPoolGetMaxConcurrency(threadPool);
}

std::string PyThreadPool::_aiir_thread_pool_ptr() const {
  std::stringstream ss;
  ss << threadPool.ptr;
  return ss.str();
}

//------------------------------------------------------------------------------
// PyAiirContext
//------------------------------------------------------------------------------

PyAiirContext::PyAiirContext(AiirContext context) : context(context) {
  nb::gil_scoped_acquire acquire;
  nb::ft_lock_guard lock(live_contexts_mutex);
  auto &liveContexts = getLiveContexts();
  liveContexts[context.ptr] = this;
}

PyAiirContext::~PyAiirContext() {
  // Note that the only public way to construct an instance is via the
  // forContext method, which always puts the associated handle into
  // liveContexts.
  nb::gil_scoped_acquire acquire;
  {
    nb::ft_lock_guard lock(live_contexts_mutex);
    getLiveContexts().erase(context.ptr);
  }
  aiirContextDestroy(context);
}

PyAiirContextRef PyAiirContext::getRef() {
  return PyAiirContextRef(this, nb::cast(this));
}

nb::object PyAiirContext::getCapsule() {
  return nb::steal<nb::object>(aiirPythonContextToCapsule(get()));
}

nb::object PyAiirContext::createFromCapsule(nb::object capsule) {
  AiirContext rawContext = aiirPythonCapsuleToContext(capsule.ptr());
  if (aiirContextIsNull(rawContext))
    throw nb::python_error();
  return forContext(rawContext).releaseObject();
}

PyAiirContextRef PyAiirContext::forContext(AiirContext context) {
  nb::gil_scoped_acquire acquire;
  nb::ft_lock_guard lock(live_contexts_mutex);
  auto &liveContexts = getLiveContexts();
  auto it = liveContexts.find(context.ptr);
  if (it == liveContexts.end()) {
    // Create.
    PyAiirContext *unownedContextWrapper = new PyAiirContext(context);
    nb::object pyRef = nb::cast(unownedContextWrapper);
    assert(pyRef && "cast to nb::object failed");
    liveContexts[context.ptr] = unownedContextWrapper;
    return PyAiirContextRef(unownedContextWrapper, std::move(pyRef));
  }
  // Use existing.
  nb::object pyRef = nb::cast(it->second);
  return PyAiirContextRef(it->second, std::move(pyRef));
}

nb::ft_mutex PyAiirContext::live_contexts_mutex;

PyAiirContext::LiveContextMap &PyAiirContext::getLiveContexts() {
  static LiveContextMap liveContexts;
  return liveContexts;
}

size_t PyAiirContext::getLiveCount() {
  nb::ft_lock_guard lock(live_contexts_mutex);
  return getLiveContexts().size();
}

nb::object PyAiirContext::contextEnter(nb::object context) {
  return PyThreadContextEntry::pushContext(context);
}

void PyAiirContext::contextExit(const nb::object &excType,
                                const nb::object &excVal,
                                const nb::object &excTb) {
  PyThreadContextEntry::popContext(*this);
}

nb::object PyAiirContext::attachDiagnosticHandler(nb::object callback) {
  // Note that ownership is transferred to the delete callback below by way of
  // an explicit inc_ref (borrow).
  PyDiagnosticHandler *pyHandler =
      new PyDiagnosticHandler(get(), std::move(callback));
  nb::object pyHandlerObject =
      nb::cast(pyHandler, nb::rv_policy::take_ownership);
  (void)pyHandlerObject.inc_ref();

  // In these C callbacks, the userData is a PyDiagnosticHandler* that is
  // guaranteed to be known to pybind.
  auto handlerCallback =
      +[](AiirDiagnostic diagnostic, void *userData) -> AiirLogicalResult {
    PyDiagnostic *pyDiagnostic = new PyDiagnostic(diagnostic);
    nb::object pyDiagnosticObject =
        nb::cast(pyDiagnostic, nb::rv_policy::take_ownership);

    auto *pyHandler = static_cast<PyDiagnosticHandler *>(userData);
    bool result = false;
    {
      // Since this can be called from arbitrary C++ contexts, always get the
      // gil.
      nb::gil_scoped_acquire gil;
      try {
        result = nb::cast<bool>(pyHandler->callback(pyDiagnostic));
      } catch (std::exception &e) {
        fprintf(stderr, "AIIR Python Diagnostic handler raised exception: %s\n",
                e.what());
        pyHandler->hadError = true;
      }
    }

    pyDiagnostic->invalidate();
    return result ? aiirLogicalResultSuccess() : aiirLogicalResultFailure();
  };
  auto deleteCallback = +[](void *userData) {
    auto *pyHandler = static_cast<PyDiagnosticHandler *>(userData);
    assert(pyHandler->registeredID && "handler is not registered");
    pyHandler->registeredID.reset();

    // Decrement reference, balancing the inc_ref() above.
    nb::object pyHandlerObject = nb::cast(pyHandler, nb::rv_policy::reference);
    pyHandlerObject.dec_ref();
  };

  pyHandler->registeredID = aiirContextAttachDiagnosticHandler(
      get(), handlerCallback, static_cast<void *>(pyHandler), deleteCallback);
  return pyHandlerObject;
}

AiirLogicalResult PyAiirContext::ErrorCapture::handler(AiirDiagnostic diag,
                                                       void *userData) {
  auto *self = static_cast<ErrorCapture *>(userData);
  // Check if the context requested we emit errors instead of capturing them.
  if (self->ctx->emitErrorDiagnostics)
    return aiirLogicalResultFailure();

  if (aiirDiagnosticGetSeverity(diag) !=
      AiirDiagnosticSeverity::AiirDiagnosticError)
    return aiirLogicalResultFailure();

  self->errors.emplace_back(PyDiagnostic(diag).getInfo());
  return aiirLogicalResultSuccess();
}

PyAiirContext &DefaultingPyAiirContext::resolve() {
  PyAiirContext *context = PyThreadContextEntry::getDefaultContext();
  if (!context) {
    throw std::runtime_error(
        "An AIIR function requires a Context but none was provided in the call "
        "or from the surrounding environment. Either pass to the function with "
        "a 'context=' argument or establish a default using 'with Context():'");
  }
  return *context;
}

//------------------------------------------------------------------------------
// PyThreadContextEntry management
//------------------------------------------------------------------------------

std::vector<PyThreadContextEntry> &PyThreadContextEntry::getStack() {
  static thread_local std::vector<PyThreadContextEntry> stack;
  return stack;
}

PyThreadContextEntry *PyThreadContextEntry::getTopOfStack() {
  auto &stack = getStack();
  if (stack.empty())
    return nullptr;
  return &stack.back();
}

void PyThreadContextEntry::push(FrameKind frameKind, nb::object context,
                                nb::object insertionPoint,
                                nb::object location) {
  auto &stack = getStack();
  stack.emplace_back(frameKind, std::move(context), std::move(insertionPoint),
                     std::move(location));
  // If the new stack has more than one entry and the context of the new top
  // entry matches the previous, copy the insertionPoint and location from the
  // previous entry if missing from the new top entry.
  if (stack.size() > 1) {
    auto &prev = *(stack.rbegin() + 1);
    auto &current = stack.back();
    if (current.context.is(prev.context)) {
      // Default non-context objects from the previous entry.
      if (!current.insertionPoint)
        current.insertionPoint = prev.insertionPoint;
      if (!current.location)
        current.location = prev.location;
    }
  }
}

PyAiirContext *PyThreadContextEntry::getContext() {
  if (!context)
    return nullptr;
  return nb::cast<PyAiirContext *>(context);
}

PyInsertionPoint *PyThreadContextEntry::getInsertionPoint() {
  if (!insertionPoint)
    return nullptr;
  return nb::cast<PyInsertionPoint *>(insertionPoint);
}

PyLocation *PyThreadContextEntry::getLocation() {
  if (!location)
    return nullptr;
  return nb::cast<PyLocation *>(location);
}

PyAiirContext *PyThreadContextEntry::getDefaultContext() {
  auto *tos = getTopOfStack();
  return tos ? tos->getContext() : nullptr;
}

PyInsertionPoint *PyThreadContextEntry::getDefaultInsertionPoint() {
  auto *tos = getTopOfStack();
  return tos ? tos->getInsertionPoint() : nullptr;
}

PyLocation *PyThreadContextEntry::getDefaultLocation() {
  auto *tos = getTopOfStack();
  return tos ? tos->getLocation() : nullptr;
}

nb::object PyThreadContextEntry::pushContext(nb::object context) {
  push(FrameKind::Context, /*context=*/context,
       /*insertionPoint=*/nb::object(),
       /*location=*/nb::object());
  return context;
}

void PyThreadContextEntry::popContext(PyAiirContext &context) {
  auto &stack = getStack();
  if (stack.empty())
    throw std::runtime_error("Unbalanced Context enter/exit");
  auto &tos = stack.back();
  if (tos.frameKind != FrameKind::Context && tos.getContext() != &context)
    throw std::runtime_error("Unbalanced Context enter/exit");
  stack.pop_back();
}

nb::object
PyThreadContextEntry::pushInsertionPoint(nb::object insertionPointObj) {
  PyInsertionPoint &insertionPoint =
      nb::cast<PyInsertionPoint &>(insertionPointObj);
  nb::object contextObj =
      insertionPoint.getBlock().getParentOperation()->getContext().getObject();
  push(FrameKind::InsertionPoint,
       /*context=*/contextObj,
       /*insertionPoint=*/insertionPointObj,
       /*location=*/nb::object());
  return insertionPointObj;
}

void PyThreadContextEntry::popInsertionPoint(PyInsertionPoint &insertionPoint) {
  auto &stack = getStack();
  if (stack.empty())
    throw std::runtime_error("Unbalanced InsertionPoint enter/exit");
  auto &tos = stack.back();
  if (tos.frameKind != FrameKind::InsertionPoint &&
      tos.getInsertionPoint() != &insertionPoint)
    throw std::runtime_error("Unbalanced InsertionPoint enter/exit");
  stack.pop_back();
}

nb::object PyThreadContextEntry::pushLocation(nb::object locationObj) {
  PyLocation &location = nb::cast<PyLocation &>(locationObj);
  nb::object contextObj = location.getContext().getObject();
  push(FrameKind::Location, /*context=*/contextObj,
       /*insertionPoint=*/nb::object(),
       /*location=*/locationObj);
  return locationObj;
}

void PyThreadContextEntry::popLocation(PyLocation &location) {
  auto &stack = getStack();
  if (stack.empty())
    throw std::runtime_error("Unbalanced Location enter/exit");
  auto &tos = stack.back();
  if (tos.frameKind != FrameKind::Location && tos.getLocation() != &location)
    throw std::runtime_error("Unbalanced Location enter/exit");
  stack.pop_back();
}

//------------------------------------------------------------------------------
// PyDiagnostic*
//------------------------------------------------------------------------------

void PyDiagnostic::invalidate() {
  valid = false;
  if (materializedNotes) {
    for (nb::handle noteObject : *materializedNotes) {
      PyDiagnostic *note = nb::cast<PyDiagnostic *>(noteObject);
      note->invalidate();
    }
  }
}

PyDiagnosticHandler::PyDiagnosticHandler(AiirContext context,
                                         nb::object callback)
    : context(context), callback(std::move(callback)) {}

PyDiagnosticHandler::~PyDiagnosticHandler() = default;

void PyDiagnosticHandler::detach() {
  if (!registeredID)
    return;
  AiirDiagnosticHandlerID localID = *registeredID;
  aiirContextDetachDiagnosticHandler(context, localID);
  assert(!registeredID && "should have unregistered");
  // Not strictly necessary but keeps stale pointers from being around to cause
  // issues.
  context = {nullptr};
}

void PyDiagnostic::checkValid() {
  if (!valid) {
    throw std::invalid_argument(
        "Diagnostic is invalid (used outside of callback)");
  }
}

PyDiagnosticSeverity PyDiagnostic::getSeverity() {
  checkValid();
  return static_cast<PyDiagnosticSeverity>(
      aiirDiagnosticGetSeverity(diagnostic));
}

PyLocation PyDiagnostic::getLocation() {
  checkValid();
  AiirLocation loc = aiirDiagnosticGetLocation(diagnostic);
  AiirContext context = aiirLocationGetContext(loc);
  return PyLocation(PyAiirContext::forContext(context), loc);
}

nb::str PyDiagnostic::getMessage() {
  checkValid();
  nb::object fileObject = nb::module_::import_("io").attr("StringIO")();
  PyFileAccumulator accum(fileObject, /*binary=*/false);
  aiirDiagnosticPrint(diagnostic, accum.getCallback(), accum.getUserData());
  return nb::cast<nb::str>(fileObject.attr("getvalue")());
}

nb::typed<nb::tuple, PyDiagnostic> PyDiagnostic::getNotes() {
  checkValid();
  if (materializedNotes)
    return *materializedNotes;
  intptr_t numNotes = aiirDiagnosticGetNumNotes(diagnostic);
  nb::tuple notes = nb::steal<nb::tuple>(PyTuple_New(numNotes));
  for (intptr_t i = 0; i < numNotes; ++i) {
    AiirDiagnostic noteDiag = aiirDiagnosticGetNote(diagnostic, i);
    nb::object diagnostic = nb::cast(PyDiagnostic(noteDiag));
    PyTuple_SetItem(notes.ptr(), i, diagnostic.release().ptr());
  }
  materializedNotes = std::move(notes);

  return *materializedNotes;
}

PyDiagnostic::DiagnosticInfo PyDiagnostic::getInfo() {
  std::vector<DiagnosticInfo> notes;
  for (nb::handle n : getNotes())
    notes.emplace_back(nb::cast<PyDiagnostic>(n).getInfo());
  return {getSeverity(), getLocation(), nb::cast<std::string>(getMessage()),
          std::move(notes)};
}

//------------------------------------------------------------------------------
// PyDialect, PyDialectDescriptor, PyDialects, PyDialectRegistry
//------------------------------------------------------------------------------

AiirDialect PyDialects::getDialectForKey(const std::string &key,
                                         bool attrError) {
  AiirDialect dialect = aiirContextGetOrLoadDialect(getContext()->get(),
                                                    {key.data(), key.size()});
  if (aiirDialectIsNull(dialect)) {
    std::string msg = join("Dialect '", key, "' not found");
    if (attrError)
      throw nb::attribute_error(msg.c_str());
    throw nb::index_error(msg.c_str());
  }
  return dialect;
}

nb::object PyDialectRegistry::getCapsule() {
  return nb::steal<nb::object>(aiirPythonDialectRegistryToCapsule(*this));
}

PyDialectRegistry PyDialectRegistry::createFromCapsule(nb::object capsule) {
  AiirDialectRegistry rawRegistry =
      aiirPythonCapsuleToDialectRegistry(capsule.ptr());
  if (aiirDialectRegistryIsNull(rawRegistry))
    throw nb::python_error();
  return PyDialectRegistry(rawRegistry);
}

//------------------------------------------------------------------------------
// PyLocation
//------------------------------------------------------------------------------

nb::object PyLocation::getCapsule() {
  return nb::steal<nb::object>(aiirPythonLocationToCapsule(*this));
}

PyLocation PyLocation::createFromCapsule(nb::object capsule) {
  AiirLocation rawLoc = aiirPythonCapsuleToLocation(capsule.ptr());
  if (aiirLocationIsNull(rawLoc))
    throw nb::python_error();
  return PyLocation(PyAiirContext::forContext(aiirLocationGetContext(rawLoc)),
                    rawLoc);
}

nb::object PyLocation::contextEnter(nb::object locationObj) {
  return PyThreadContextEntry::pushLocation(locationObj);
}

void PyLocation::contextExit(const nb::object &excType,
                             const nb::object &excVal,
                             const nb::object &excTb) {
  PyThreadContextEntry::popLocation(*this);
}

PyLocation &DefaultingPyLocation::resolve() {
  auto *location = PyThreadContextEntry::getDefaultLocation();
  if (!location) {
    throw std::runtime_error(
        "An AIIR function requires a Location but none was provided in the "
        "call or from the surrounding environment. Either pass to the function "
        "with a 'loc=' argument or establish a default using 'with loc:'");
  }
  return *location;
}

//------------------------------------------------------------------------------
// PyModule
//------------------------------------------------------------------------------

PyModule::PyModule(PyAiirContextRef contextRef, AiirModule module)
    : BaseContextObject(std::move(contextRef)), module(module) {}

PyModule::~PyModule() {
  nb::gil_scoped_acquire acquire;
  auto &liveModules = getContext()->liveModules;
  assert(liveModules.count(module.ptr) == 1 &&
         "destroying module not in live map");
  liveModules.erase(module.ptr);
  aiirModuleDestroy(module);
}

PyModuleRef PyModule::forModule(AiirModule module) {
  AiirContext context = aiirModuleGetContext(module);
  PyAiirContextRef contextRef = PyAiirContext::forContext(context);

  nb::gil_scoped_acquire acquire;
  auto &liveModules = contextRef->liveModules;
  auto it = liveModules.find(module.ptr);
  if (it == liveModules.end()) {
    // Create.
    PyModule *unownedModule = new PyModule(std::move(contextRef), module);
    // Note that the default return value policy on cast is automatic_reference,
    // which does not take ownership (delete will not be called).
    // Just be explicit.
    nb::object pyRef = nb::cast(unownedModule, nb::rv_policy::take_ownership);
    unownedModule->handle = pyRef;
    liveModules[module.ptr] =
        std::make_pair(unownedModule->handle, unownedModule);
    return PyModuleRef(unownedModule, std::move(pyRef));
  }
  // Use existing.
  PyModule *existing = it->second.second;
  nb::object pyRef = nb::borrow<nb::object>(it->second.first);
  return PyModuleRef(existing, std::move(pyRef));
}

nb::object PyModule::createFromCapsule(nb::object capsule) {
  AiirModule rawModule = aiirPythonCapsuleToModule(capsule.ptr());
  if (aiirModuleIsNull(rawModule))
    throw nb::python_error();
  return forModule(rawModule).releaseObject();
}

nb::object PyModule::getCapsule() {
  return nb::steal<nb::object>(aiirPythonModuleToCapsule(get()));
}

//------------------------------------------------------------------------------
// PyOperation
//------------------------------------------------------------------------------

PyOperation::PyOperation(PyAiirContextRef contextRef, AiirOperation operation)
    : BaseContextObject(std::move(contextRef)), operation(operation) {}

PyOperation::~PyOperation() {
  // If the operation has already been invalidated there is nothing to do.
  if (!valid)
    return;
  // Otherwise, invalidate the operation when it is attached.
  if (isAttached())
    setInvalid();
  else {
    // And destroy it when it is detached, i.e. owned by Python.
    erase();
  }
}

namespace {

// Constructs a new object of type T in-place on the Python heap, returning a
// PyObjectRef to it, loosely analogous to std::make_shared<T>().
template <typename T, class... Args>
PyObjectRef<T> makeObjectRef(Args &&...args) {
  nb::handle type = nb::type<T>();
  nb::object instance = nb::inst_alloc(type);
  T *ptr = nb::inst_ptr<T>(instance);
  new (ptr) T(std::forward<Args>(args)...);
  nb::inst_mark_ready(instance);
  return PyObjectRef<T>(ptr, std::move(instance));
}

} // namespace

PyOperationRef PyOperation::createInstance(PyAiirContextRef contextRef,
                                           AiirOperation operation,
                                           nb::object parentKeepAlive) {
  // Create.
  PyOperationRef unownedOperation =
      makeObjectRef<PyOperation>(std::move(contextRef), operation);
  unownedOperation->handle = unownedOperation.getObject();
  if (parentKeepAlive) {
    unownedOperation->parentKeepAlive = std::move(parentKeepAlive);
  }
  return unownedOperation;
}

PyOperationRef PyOperation::forOperation(PyAiirContextRef contextRef,
                                         AiirOperation operation,
                                         nb::object parentKeepAlive) {
  return createInstance(std::move(contextRef), operation,
                        std::move(parentKeepAlive));
}

PyOperationRef PyOperation::createDetached(PyAiirContextRef contextRef,
                                           AiirOperation operation,
                                           nb::object parentKeepAlive) {
  PyOperationRef created = createInstance(std::move(contextRef), operation,
                                          std::move(parentKeepAlive));
  created->attached = false;
  return created;
}

PyOperationRef PyOperation::parse(PyAiirContextRef contextRef,
                                  const std::string &sourceStr,
                                  const std::string &sourceName) {
  PyAiirContext::ErrorCapture errors(contextRef);
  AiirOperation op =
      aiirOperationCreateParse(contextRef->get(), toAiirStringRef(sourceStr),
                               toAiirStringRef(sourceName));
  if (aiirOperationIsNull(op))
    throw AIIRError("Unable to parse operation assembly", errors.take());
  return PyOperation::createDetached(std::move(contextRef), op);
}

void PyOperation::detachFromParent() {
  aiirOperationRemoveFromParent(getOperation());
  setDetached();
  parentKeepAlive = nb::object();
}

AiirOperation PyOperation::get() const {
  checkValid();
  return operation;
}

PyOperationRef PyOperation::getRef() {
  return PyOperationRef(this, nb::borrow<nb::object>(handle));
}

void PyOperation::setAttached(const nb::object &parent) {
  assert(!attached && "operation already attached");
  attached = true;
}

void PyOperation::setDetached() {
  assert(attached && "operation already detached");
  attached = false;
}

void PyOperation::checkValid() const {
  if (!valid) {
    throw std::runtime_error("the operation has been invalidated");
  }
}

void PyOperationBase::print(std::optional<int64_t> largeElementsLimit,
                            std::optional<int64_t> largeResourceLimit,
                            bool enableDebugInfo, bool prettyDebugInfo,
                            bool printGenericOpForm, bool useLocalScope,
                            bool useNameLocAsPrefix, bool assumeVerified,
                            nb::object fileObject, bool binary,
                            bool skipRegions) {
  PyOperation &operation = getOperation();
  operation.checkValid();
  if (fileObject.is_none())
    fileObject = nb::module_::import_("sys").attr("stdout");

  AiirOpPrintingFlags flags = aiirOpPrintingFlagsCreate();
  if (largeElementsLimit)
    aiirOpPrintingFlagsElideLargeElementsAttrs(flags, *largeElementsLimit);
  if (largeResourceLimit)
    aiirOpPrintingFlagsElideLargeResourceString(flags, *largeResourceLimit);
  if (enableDebugInfo)
    aiirOpPrintingFlagsEnableDebugInfo(flags, /*enable=*/true,
                                       /*prettyForm=*/prettyDebugInfo);
  if (printGenericOpForm)
    aiirOpPrintingFlagsPrintGenericOpForm(flags);
  if (useLocalScope)
    aiirOpPrintingFlagsUseLocalScope(flags);
  if (assumeVerified)
    aiirOpPrintingFlagsAssumeVerified(flags);
  if (skipRegions)
    aiirOpPrintingFlagsSkipRegions(flags);
  if (useNameLocAsPrefix)
    aiirOpPrintingFlagsPrintNameLocAsPrefix(flags);

  PyFileAccumulator accum(fileObject, binary);
  aiirOperationPrintWithFlags(operation, flags, accum.getCallback(),
                              accum.getUserData());
  aiirOpPrintingFlagsDestroy(flags);
}

void PyOperationBase::print(PyAsmState &state, nb::object fileObject,
                            bool binary) {
  PyOperation &operation = getOperation();
  operation.checkValid();
  if (fileObject.is_none())
    fileObject = nb::module_::import_("sys").attr("stdout");
  PyFileAccumulator accum(fileObject, binary);
  aiirOperationPrintWithState(operation, state.get(), accum.getCallback(),
                              accum.getUserData());
}

void PyOperationBase::writeBytecode(const nb::object &fileOrStringObject,
                                    std::optional<int64_t> bytecodeVersion) {
  PyOperation &operation = getOperation();
  operation.checkValid();
  PyFileAccumulator accum(fileOrStringObject, /*binary=*/true);

  if (!bytecodeVersion.has_value())
    return aiirOperationWriteBytecode(operation, accum.getCallback(),
                                      accum.getUserData());

  AiirBytecodeWriterConfig config = aiirBytecodeWriterConfigCreate();
  aiirBytecodeWriterConfigDesiredEmitVersion(config, *bytecodeVersion);
  AiirLogicalResult res = aiirOperationWriteBytecodeWithConfig(
      operation, config, accum.getCallback(), accum.getUserData());
  aiirBytecodeWriterConfigDestroy(config);
  if (aiirLogicalResultIsFailure(res))
    throw nb::value_error(
        join("Unable to honor desired bytecode version ", *bytecodeVersion)
            .c_str());
}

void PyOperationBase::walk(std::function<PyWalkResult(AiirOperation)> callback,
                           PyWalkOrder walkOrder) {
  PyOperation &operation = getOperation();
  operation.checkValid();
  struct UserData {
    std::function<PyWalkResult(AiirOperation)> callback;
    bool gotException;
    std::string exceptionWhat;
    nb::object exceptionType;
  };
  UserData userData{callback, false, {}, {}};
  AiirOperationWalkCallback walkCallback = [](AiirOperation op,
                                              void *userData) {
    UserData *calleeUserData = static_cast<UserData *>(userData);
    try {
      return static_cast<AiirWalkResult>((calleeUserData->callback)(op));
    } catch (nb::python_error &e) {
      calleeUserData->gotException = true;
      calleeUserData->exceptionWhat = std::string(e.what());
      calleeUserData->exceptionType = nb::borrow(e.type());
      return AiirWalkResult::AiirWalkResultInterrupt;
    }
  };
  aiirOperationWalk(operation, walkCallback, &userData,
                    static_cast<AiirWalkOrder>(walkOrder));
  if (userData.gotException) {
    std::string message("Exception raised in callback: ");
    message.append(userData.exceptionWhat);
    throw std::runtime_error(message);
  }
}

nb::object PyOperationBase::getAsm(bool binary,
                                   std::optional<int64_t> largeElementsLimit,
                                   std::optional<int64_t> largeResourceLimit,
                                   bool enableDebugInfo, bool prettyDebugInfo,
                                   bool printGenericOpForm, bool useLocalScope,
                                   bool useNameLocAsPrefix, bool assumeVerified,
                                   bool skipRegions) {
  nb::object fileObject;
  if (binary) {
    fileObject = nb::module_::import_("io").attr("BytesIO")();
  } else {
    fileObject = nb::module_::import_("io").attr("StringIO")();
  }
  print(/*largeElementsLimit=*/largeElementsLimit,
        /*largeResourceLimit=*/largeResourceLimit,
        /*enableDebugInfo=*/enableDebugInfo,
        /*prettyDebugInfo=*/prettyDebugInfo,
        /*printGenericOpForm=*/printGenericOpForm,
        /*useLocalScope=*/useLocalScope,
        /*useNameLocAsPrefix=*/useNameLocAsPrefix,
        /*assumeVerified=*/assumeVerified,
        /*fileObject=*/fileObject,
        /*binary=*/binary,
        /*skipRegions=*/skipRegions);

  return fileObject.attr("getvalue")();
}

void PyOperationBase::moveAfter(PyOperationBase &other) {
  PyOperation &operation = getOperation();
  PyOperation &otherOp = other.getOperation();
  operation.checkValid();
  otherOp.checkValid();
  aiirOperationMoveAfter(operation, otherOp);
  operation.parentKeepAlive = otherOp.parentKeepAlive;
}

void PyOperationBase::moveBefore(PyOperationBase &other) {
  PyOperation &operation = getOperation();
  PyOperation &otherOp = other.getOperation();
  operation.checkValid();
  otherOp.checkValid();
  aiirOperationMoveBefore(operation, otherOp);
  operation.parentKeepAlive = otherOp.parentKeepAlive;
}

bool PyOperationBase::isBeforeInBlock(PyOperationBase &other) {
  PyOperation &operation = getOperation();
  PyOperation &otherOp = other.getOperation();
  operation.checkValid();
  otherOp.checkValid();
  return aiirOperationIsBeforeInBlock(operation, otherOp);
}

bool PyOperationBase::verify() {
  PyOperation &op = getOperation();
  PyAiirContext::ErrorCapture errors(op.getContext());
  if (!aiirOperationVerify(op.get()))
    throw AIIRError("Verification failed", errors.take());
  return true;
}

std::optional<PyOperationRef> PyOperation::getParentOperation() {
  checkValid();
  if (!isAttached())
    throw nb::value_error("Detached operations have no parent");
  AiirOperation operation = aiirOperationGetParentOperation(get());
  if (aiirOperationIsNull(operation))
    return {};
  return PyOperation::forOperation(getContext(), operation);
}

PyBlock PyOperation::getBlock() {
  checkValid();
  std::optional<PyOperationRef> parentOperation = getParentOperation();
  AiirBlock block = aiirOperationGetBlock(get());
  assert(!aiirBlockIsNull(block) && "Attached operation has null parent");
  assert(parentOperation && "Operation has no parent");
  return PyBlock{std::move(*parentOperation), block};
}

nb::object PyOperation::getCapsule() {
  checkValid();
  return nb::steal<nb::object>(aiirPythonOperationToCapsule(get()));
}

nb::object PyOperation::createFromCapsule(const nb::object &capsule) {
  AiirOperation rawOperation = aiirPythonCapsuleToOperation(capsule.ptr());
  if (aiirOperationIsNull(rawOperation))
    throw nb::python_error();
  AiirContext rawCtxt = aiirOperationGetContext(rawOperation);
  return forOperation(PyAiirContext::forContext(rawCtxt), rawOperation)
      .releaseObject();
}

static void maybeInsertOperation(PyOperationRef &op,
                                 const nb::object &maybeIp) {
  // InsertPoint active?
  if (!maybeIp.is(nb::cast(false))) {
    PyInsertionPoint *ip;
    if (maybeIp.is_none()) {
      ip = PyThreadContextEntry::getDefaultInsertionPoint();
    } else {
      ip = nb::cast<PyInsertionPoint *>(maybeIp);
    }
    if (ip)
      ip->insert(*op.get());
  }
}

nb::object PyOperation::create(std::string_view name,
                               std::optional<std::vector<PyType *>> results,
                               const AiirValue *operands, size_t numOperands,
                               std::optional<nb::dict> attributes,
                               std::optional<std::vector<PyBlock *>> successors,
                               int regions, PyLocation &location,
                               const nb::object &maybeIp, bool inferType) {
  std::vector<AiirType> aiirResults;
  std::vector<AiirBlock> aiirSuccessors;
  std::vector<std::pair<std::string, AiirAttribute>> aiirAttributes;

  // General parameter validation.
  if (regions < 0)
    throw nb::value_error("number of regions must be >= 0");

  // Unpack/validate results.
  if (results) {
    aiirResults.reserve(results->size());
    for (PyType *result : *results) {
      // TODO: Verify result type originate from the same context.
      if (!result)
        throw nb::value_error("result type cannot be None");
      aiirResults.push_back(*result);
    }
  }
  // Unpack/validate attributes.
  if (attributes) {
    aiirAttributes.reserve(attributes->size());
    for (std::pair<nb::handle, nb::handle> it : *attributes) {
      std::string key;
      try {
        key = nb::cast<std::string>(it.first);
      } catch (nb::cast_error &err) {
        std::string msg = join("Invalid attribute key (not a string) when "
                               "attempting to create the operation \"",
                               name, "\" (", err.what(), ")");
        throw nb::type_error(msg.c_str());
      }
      try {
        auto &attribute = nb::cast<PyAttribute &>(it.second);
        // TODO: Verify attribute originates from the same context.
        aiirAttributes.emplace_back(std::move(key), attribute);
      } catch (nb::cast_error &err) {
        std::string msg = join("Invalid attribute value for the key \"", key,
                               "\" when attempting to create the operation \"",
                               name, "\" (", err.what(), ")");
        throw nb::type_error(msg.c_str());
      } catch (std::runtime_error &) {
        // This exception seems thrown when the value is "None".
        std::string msg = join(
            "Found an invalid (`None`?) attribute value for the key \"", key,
            "\" when attempting to create the operation \"", name, "\"");
        throw std::runtime_error(msg);
      }
    }
  }
  // Unpack/validate successors.
  if (successors) {
    aiirSuccessors.reserve(successors->size());
    for (PyBlock *successor : *successors) {
      // TODO: Verify successor originate from the same context.
      if (!successor)
        throw nb::value_error("successor block cannot be None");
      aiirSuccessors.push_back(successor->get());
    }
  }

  // Apply unpacked/validated to the operation state. Beyond this
  // point, exceptions cannot be thrown or else the state will leak.
  AiirOperationState state =
      aiirOperationStateGet(toAiirStringRef(name), location);
  if (numOperands > 0)
    aiirOperationStateAddOperands(&state, numOperands, operands);
  state.enableResultTypeInference = inferType;
  if (!aiirResults.empty())
    aiirOperationStateAddResults(&state, aiirResults.size(),
                                 aiirResults.data());
  if (!aiirAttributes.empty()) {
    // Note that the attribute names directly reference bytes in
    // aiirAttributes, so that vector must not be changed from here
    // on.
    std::vector<AiirNamedAttribute> aiirNamedAttributes;
    aiirNamedAttributes.reserve(aiirAttributes.size());
    for (const std::pair<std::string, AiirAttribute> &it : aiirAttributes)
      aiirNamedAttributes.push_back(aiirNamedAttributeGet(
          aiirIdentifierGet(aiirAttributeGetContext(it.second),
                            toAiirStringRef(it.first)),
          it.second));
    aiirOperationStateAddAttributes(&state, aiirNamedAttributes.size(),
                                    aiirNamedAttributes.data());
  }
  if (!aiirSuccessors.empty())
    aiirOperationStateAddSuccessors(&state, aiirSuccessors.size(),
                                    aiirSuccessors.data());
  if (regions) {
    std::vector<AiirRegion> aiirRegions;
    aiirRegions.resize(regions);
    for (int i = 0; i < regions; ++i)
      aiirRegions[i] = aiirRegionCreate();
    aiirOperationStateAddOwnedRegions(&state, aiirRegions.size(),
                                      aiirRegions.data());
  }

  // Construct the operation.
  PyAiirContext::ErrorCapture errors(location.getContext());
  AiirOperation operation = aiirOperationCreate(&state);
  if (!operation.ptr)
    throw AIIRError("Operation creation failed", errors.take());
  PyOperationRef created =
      PyOperation::createDetached(location.getContext(), operation);
  maybeInsertOperation(created, maybeIp);

  return created.getObject();
}

nb::object PyOperation::clone(const nb::object &maybeIp) {
  AiirOperation clonedOperation = aiirOperationClone(operation);
  PyOperationRef cloned =
      PyOperation::createDetached(getContext(), clonedOperation);
  maybeInsertOperation(cloned, maybeIp);

  return cloned->createOpView();
}

nb::object PyOperation::createOpView() {
  checkValid();
  AiirIdentifier ident = aiirOperationGetName(get());
  AiirStringRef identStr = aiirIdentifierStr(ident);
  auto operationCls = PyGlobals::get().lookupOperationClass(
      std::string_view(identStr.data, identStr.length));
  if (operationCls)
    return PyOpView::constructDerived(*operationCls, getRef().getObject());
  return nb::cast(PyOpView(getRef().getObject()));
}

void PyOperation::erase() {
  checkValid();
  setInvalid();
  aiirOperationDestroy(operation);
}

void PyOpResult::bindDerived(ClassTy &c) {
  c.def_prop_ro(
      "owner",
      [](PyOpResult &self) -> nb::typed<nb::object, PyOpView> {
        assert(aiirOperationEqual(self.getParentOperation()->get(),
                                  aiirOpResultGetOwner(self.get())) &&
               "expected the owner of the value in Python to match that in "
               "the IR");
        return self.getParentOperation()->createOpView();
      },
      "Returns the operation that produces this result.");
  c.def_prop_ro(
      "result_number",
      [](PyOpResult &self) { return aiirOpResultGetResultNumber(self.get()); },
      "Returns the position of this result in the operation's result list.");
}

/// Returns the list of types of the values held by container.
template <typename Container>
static std::vector<nb::typed<nb::object, PyType>>
getValueTypes(Container &container, PyAiirContextRef &context) {
  std::vector<nb::typed<nb::object, PyType>> result;
  result.reserve(container.size());
  for (int i = 0, e = container.size(); i < e; ++i) {
    result.push_back(PyType(context->getRef(),
                            aiirValueGetType(container.getElement(i).get()))
                         .maybeDownCast());
  }
  return result;
}

PyOpResultList::PyOpResultList(PyOperationRef operation, intptr_t startIndex,
                               intptr_t length, intptr_t step)
    : Sliceable(startIndex,
                length == -1 ? aiirOperationGetNumResults(operation->get())
                             : length,
                step),
      operation(std::move(operation)) {}

void PyOpResultList::bindDerived(ClassTy &c) {
  c.def_prop_ro(
      "types",
      [](PyOpResultList &self) {
        return getValueTypes(self, self.operation->getContext());
      },
      "Returns a list of types for all results in this result list.");
  c.def_prop_ro(
      "owner",
      [](PyOpResultList &self) -> nb::typed<nb::object, PyOpView> {
        return self.operation->createOpView();
      },
      "Returns the operation that owns this result list.");
}

intptr_t PyOpResultList::getRawNumElements() {
  operation->checkValid();
  return aiirOperationGetNumResults(operation->get());
}

PyOpResult PyOpResultList::getRawElement(intptr_t index) {
  PyValue value(operation, aiirOperationGetResult(operation->get(), index));
  return PyOpResult(value);
}

PyOpResultList PyOpResultList::slice(intptr_t startIndex, intptr_t length,
                                     intptr_t step) const {
  return PyOpResultList(operation, startIndex, length, step);
}

//------------------------------------------------------------------------------
// PyOpView
//------------------------------------------------------------------------------

static void populateResultTypes(std::string_view name,
                                nb::sequence resultTypeList,
                                const nb::object &resultSegmentSpecObj,
                                std::vector<int32_t> &resultSegmentLengths,
                                std::vector<PyType *> &resultTypes) {
  resultTypes.reserve(nb::len(resultTypeList));
  if (resultSegmentSpecObj.is_none()) {
    // Non-variadic result unpacking.
    size_t index = 0;
    for (nb::handle resultType : resultTypeList) {
      try {
        resultTypes.push_back(nb::cast<PyType *>(resultType));
        if (!resultTypes.back())
          throw nb::cast_error();
      } catch (nb::cast_error &err) {
        throw nb::value_error(join("Result ", index, " of operation \"", name,
                                   "\" must be a Type (", err.what(), ")")
                                  .c_str());
      }
      ++index;
    }
  } else {
    // Sized result unpacking.
    auto resultSegmentSpec = nb::cast<std::vector<int>>(resultSegmentSpecObj);
    if (resultSegmentSpec.size() != nb::len(resultTypeList)) {
      throw nb::value_error(
          join("Operation \"", name, "\" requires ", resultSegmentSpec.size(),
               " result segments but was provided ", nb::len(resultTypeList))
              .c_str());
    }
    resultSegmentLengths.reserve(nb::len(resultTypeList));
    for (size_t i = 0, e = resultSegmentSpec.size(); i < e; ++i) {
      int segmentSpec = resultSegmentSpec[i];
      if (segmentSpec == 1 || segmentSpec == 0) {
        // Unpack unary element.
        try {
          auto *resultType = nb::cast<PyType *>(resultTypeList[i]);
          if (resultType) {
            resultTypes.push_back(resultType);
            resultSegmentLengths.push_back(1);
          } else if (segmentSpec == 0) {
            // Allowed to be optional.
            resultSegmentLengths.push_back(0);
          } else {
            throw nb::value_error(
                join("Result ", i, " of operation \"", name,
                     "\" must be a Type (was None and result is not optional)")
                    .c_str());
          }
        } catch (nb::cast_error &err) {
          throw nb::value_error(join("Result ", i, " of operation \"", name,
                                     "\" must be a Type (", err.what(), ")")
                                    .c_str());
        }
      } else if (segmentSpec == -1) {
        // Unpack sequence by appending.
        try {
          if (resultTypeList[i].is_none()) {
            // Treat it as an empty list.
            resultSegmentLengths.push_back(0);
          } else {
            // Unpack the list.
            auto segment = nb::cast<nb::sequence>(resultTypeList[i]);
            for (nb::handle segmentItem : segment) {
              resultTypes.push_back(nb::cast<PyType *>(segmentItem));
              if (!resultTypes.back()) {
                throw nb::type_error("contained a None item");
              }
            }
            resultSegmentLengths.push_back(nb::len(segment));
          }
        } catch (std::exception &err) {
          // NOTE: Sloppy to be using a catch-all here, but there are at least
          // three different unrelated exceptions that can be thrown in the
          // above "casts". Just keep the scope above small and catch them all.
          throw nb::value_error(join("Result ", i, " of operation \"", name,
                                     "\" must be a Sequence of Types (",
                                     err.what(), ")")
                                    .c_str());
        }
      } else {
        throw nb::value_error("Unexpected segment spec");
      }
    }
  }
}

AiirValue getUniqueResult(AiirOperation operation) {
  auto numResults = aiirOperationGetNumResults(operation);
  if (numResults != 1) {
    auto name = aiirIdentifierStr(aiirOperationGetName(operation));
    throw nb::value_error(
        join("Cannot call .result on operation ",
             std::string_view(name.data, name.length), " which has ",
             numResults,
             " results (it is only valid for operations with a "
             "single result)")
            .c_str());
  }
  return aiirOperationGetResult(operation, 0);
}

static AiirValue getOpResultOrValue(nb::handle operand) {
  if (operand.is_none()) {
    throw nb::value_error("contained a None item");
  }
  PyOperationBase *op;
  if (nb::try_cast<PyOperationBase *>(operand, op)) {
    return getUniqueResult(op->getOperation());
  }
  PyOpResultList *opResultList;
  if (nb::try_cast<PyOpResultList *>(operand, opResultList)) {
    return getUniqueResult(opResultList->getOperation()->get());
  }
  PyValue *value;
  if (nb::try_cast<PyValue *>(operand, value)) {
    return value->get();
  }
  throw nb::value_error("is not a Value");
}

nb::typed<nb::object, PyOperation> PyOpView::buildGeneric(
    std::string_view name, std::tuple<int, bool> opRegionSpec,
    nb::object operandSegmentSpecObj, nb::object resultSegmentSpecObj,
    std::optional<nb::sequence> resultTypeList, nb::sequence operandList,
    std::optional<nb::dict> attributes,
    std::optional<std::vector<PyBlock *>> successors,
    std::optional<int> regions, PyLocation &location,
    const nb::object &maybeIp) {
  PyAiirContextRef context = location.getContext();

  // Class level operation construction metadata.
  // Operand and result segment specs are either none, which does no
  // variadic unpacking, or a list of ints with segment sizes, where each
  // element is either a positive number (typically 1 for a scalar) or -1 to
  // indicate that it is derived from the length of the same-indexed operand
  // or result (implying that it is a list at that position).
  std::vector<int32_t> operandSegmentLengths;
  std::vector<int32_t> resultSegmentLengths;

  // Validate/determine region count.
  int opMinRegionCount = std::get<0>(opRegionSpec);
  bool opHasNoVariadicRegions = std::get<1>(opRegionSpec);
  if (!regions) {
    regions = opMinRegionCount;
  }
  if (*regions < opMinRegionCount) {
    throw nb::value_error(join("Operation \"", name,
                               "\" requires a minimum of ", opMinRegionCount,
                               " regions but was built with regions=", *regions)
                              .c_str());
  }
  if (opHasNoVariadicRegions && *regions > opMinRegionCount) {
    throw nb::value_error(join("Operation \"", name,
                               "\" requires a maximum of ", opMinRegionCount,
                               " regions but was built with regions=", *regions)
                              .c_str());
  }

  // Unpack results.
  std::vector<PyType *> resultTypes;
  if (resultTypeList.has_value()) {
    populateResultTypes(name, *resultTypeList, resultSegmentSpecObj,
                        resultSegmentLengths, resultTypes);
  }

  // Unpack operands.
  std::vector<AiirValue> operands;
  operands.reserve(operands.size());
  size_t index = 0;
  if (operandSegmentSpecObj.is_none()) {
    // Non-sized operand unpacking.
    for (nb::handle operand : operandList) {
      try {
        operands.push_back(getOpResultOrValue(operand));
      } catch (nb::builtin_exception &err) {
        throw nb::value_error(join("Operand ", index, " of operation \"", name,
                                   "\" must be a Value (", err.what(), ")")
                                  .c_str());
      }
      ++index;
    }
  } else {
    // Sized operand unpacking.
    auto operandSegmentSpec = nb::cast<std::vector<int>>(operandSegmentSpecObj);
    if (operandSegmentSpec.size() != nb::len(operandList)) {
      throw nb::value_error(
          join("Operation \"", name, "\" requires ", operandSegmentSpec.size(),
               "operand segments but was provided ", nb::len(operandList))
              .c_str());
    }
    operandSegmentLengths.reserve(nb::len(operandList));
    for (size_t i = 0, e = operandSegmentSpec.size(); i < e; ++i) {
      int segmentSpec = operandSegmentSpec[i];
      if (segmentSpec == 1 || segmentSpec == 0) {
        // Unpack unary element.
        const nanobind::handle operand = operandList[i];
        if (!operand.is_none()) {
          try {
            operands.push_back(getOpResultOrValue(operand));
          } catch (nb::builtin_exception &err) {
            throw nb::value_error(join("Operand ", i, " of operation \"", name,
                                       "\" must be a Value (", err.what(), ")")
                                      .c_str());
          }

          operandSegmentLengths.push_back(1);
        } else if (segmentSpec == 0) {
          // Allowed to be optional.
          operandSegmentLengths.push_back(0);
        } else {
          throw nb::value_error(
              join("Operand ", i, " of operation \"", name,
                   "\" must be a Value (was None and operand is not optional)")
                  .c_str());
        }
      } else if (segmentSpec == -1) {
        // Unpack sequence by appending.
        try {
          if (operandList[i].is_none()) {
            // Treat it as an empty list.
            operandSegmentLengths.push_back(0);
          } else {
            // Unpack the list.
            auto segment = nb::cast<nb::sequence>(operandList[i]);
            for (nb::handle segmentItem : segment) {
              operands.push_back(getOpResultOrValue(segmentItem));
            }
            operandSegmentLengths.push_back(nb::len(segment));
          }
        } catch (std::exception &err) {
          // NOTE: Sloppy to be using a catch-all here, but there are at least
          // three different unrelated exceptions that can be thrown in the
          // above "casts". Just keep the scope above small and catch them all.
          throw nb::value_error(join("Operand ", i, " of operation \"", name,
                                     "\" must be a Sequence of Values (",
                                     err.what(), ")")
                                    .c_str());
        }
      } else {
        throw nb::value_error("Unexpected segment spec");
      }
    }
  }

  // Merge operand/result segment lengths into attributes if needed.
  if (!operandSegmentLengths.empty() || !resultSegmentLengths.empty()) {
    // Dup.
    if (attributes) {
      attributes = nb::dict(*attributes);
    } else {
      attributes = nb::dict();
    }
    if (attributes->contains("resultSegmentSizes") ||
        attributes->contains("operandSegmentSizes")) {
      throw nb::value_error("Manually setting a 'resultSegmentSizes' or "
                            "'operandSegmentSizes' attribute is unsupported. "
                            "Use Operation.create for such low-level access.");
    }

    // Add resultSegmentSizes attribute.
    if (!resultSegmentLengths.empty()) {
      AiirAttribute segmentLengthAttr =
          aiirDenseI32ArrayGet(context->get(), resultSegmentLengths.size(),
                               resultSegmentLengths.data());
      (*attributes)["resultSegmentSizes"] =
          PyAttribute(context, segmentLengthAttr);
    }

    // Add operandSegmentSizes attribute.
    if (!operandSegmentLengths.empty()) {
      AiirAttribute segmentLengthAttr =
          aiirDenseI32ArrayGet(context->get(), operandSegmentLengths.size(),
                               operandSegmentLengths.data());
      (*attributes)["operandSegmentSizes"] =
          PyAttribute(context, segmentLengthAttr);
    }
  }

  // Delegate to create.
  return PyOperation::create(name,
                             /*results=*/std::move(resultTypes),
                             /*operands=*/operands.data(),
                             /*numOperands=*/operands.size(),
                             /*attributes=*/std::move(attributes),
                             /*successors=*/std::move(successors),
                             /*regions=*/*regions, location, maybeIp,
                             !resultTypeList);
}

nb::object PyOpView::constructDerived(const nb::object &cls,
                                      const nb::object &operation) {
  nb::handle opViewType = nb::type<PyOpView>();
  nb::object instance = cls.attr("__new__")(cls);
  opViewType.attr("__init__")(instance, operation);
  return instance;
}

PyOpView::PyOpView(const nb::object &operationObject)
    // Casting through the PyOperationBase base-class and then back to the
    // Operation lets us accept any PyOperationBase subclass.
    : operation(nb::cast<PyOperationBase &>(operationObject).getOperation()),
      operationObject(operation.getRef().getObject()) {}

//------------------------------------------------------------------------------
// PyAsmState
//------------------------------------------------------------------------------

PyAsmState::PyAsmState(AiirValue value, bool useLocalScope) {
  flags = aiirOpPrintingFlagsCreate();
  // The OpPrintingFlags are not exposed Python side, create locally and
  // associate lifetime with the state.
  if (useLocalScope)
    aiirOpPrintingFlagsUseLocalScope(flags);
  state = aiirAsmStateCreateForValue(value, flags);
}

PyAsmState::PyAsmState(PyOperationBase &operation, bool useLocalScope) {
  flags = aiirOpPrintingFlagsCreate();
  // The OpPrintingFlags are not exposed Python side, create locally and
  // associate lifetime with the state.
  if (useLocalScope)
    aiirOpPrintingFlagsUseLocalScope(flags);
  state = aiirAsmStateCreateForOperation(operation.getOperation().get(), flags);
}

//------------------------------------------------------------------------------
// PyInsertionPoint.
//------------------------------------------------------------------------------

PyInsertionPoint::PyInsertionPoint(const PyBlock &block) : block(block) {}

PyInsertionPoint::PyInsertionPoint(PyOperationBase &beforeOperationBase)
    : refOperation(beforeOperationBase.getOperation().getRef()),
      block((*refOperation)->getBlock()) {}

PyInsertionPoint::PyInsertionPoint(PyOperationRef beforeOperationRef)
    : refOperation(beforeOperationRef), block((*refOperation)->getBlock()) {}

void PyInsertionPoint::insert(PyOperationBase &operationBase) {
  PyOperation &operation = operationBase.getOperation();
  if (operation.isAttached())
    throw nb::value_error(
        "Attempt to insert operation that is already attached");
  block.getParentOperation()->checkValid();
  AiirOperation beforeOp = {nullptr};
  if (refOperation) {
    // Insert before operation.
    (*refOperation)->checkValid();
    beforeOp = (*refOperation)->get();
  } else {
    // Insert at end (before null) is only valid if the block does not
    // already end in a known terminator (violating this will cause assertion
    // failures later).
    if (!aiirOperationIsNull(aiirBlockGetTerminator(block.get()))) {
      throw nb::index_error("Cannot insert operation at the end of a block "
                            "that already has a terminator. Did you mean to "
                            "use 'InsertionPoint.at_block_terminator(block)' "
                            "versus 'InsertionPoint(block)'?");
    }
  }
  aiirBlockInsertOwnedOperationBefore(block.get(), beforeOp, operation);
  operation.setAttached();
}

PyInsertionPoint PyInsertionPoint::atBlockBegin(PyBlock &block) {
  AiirOperation firstOp = aiirBlockGetFirstOperation(block.get());
  if (aiirOperationIsNull(firstOp)) {
    // Just insert at end.
    return PyInsertionPoint(block);
  }

  // Insert before first op.
  PyOperationRef firstOpRef = PyOperation::forOperation(
      block.getParentOperation()->getContext(), firstOp);
  return PyInsertionPoint{block, std::move(firstOpRef)};
}

PyInsertionPoint PyInsertionPoint::atBlockTerminator(PyBlock &block) {
  AiirOperation terminator = aiirBlockGetTerminator(block.get());
  if (aiirOperationIsNull(terminator))
    throw nb::value_error("Block has no terminator");
  PyOperationRef terminatorOpRef = PyOperation::forOperation(
      block.getParentOperation()->getContext(), terminator);
  return PyInsertionPoint{block, std::move(terminatorOpRef)};
}

PyInsertionPoint PyInsertionPoint::after(PyOperationBase &op) {
  PyOperation &operation = op.getOperation();
  PyBlock block = operation.getBlock();
  AiirOperation nextOperation = aiirOperationGetNextInBlock(operation);
  if (aiirOperationIsNull(nextOperation))
    return PyInsertionPoint(block);
  PyOperationRef nextOpRef = PyOperation::forOperation(
      block.getParentOperation()->getContext(), nextOperation);
  return PyInsertionPoint{block, std::move(nextOpRef)};
}

size_t PyAiirContext::getLiveModuleCount() { return liveModules.size(); }

nb::object PyInsertionPoint::contextEnter(nb::object insertPoint) {
  return PyThreadContextEntry::pushInsertionPoint(std::move(insertPoint));
}

void PyInsertionPoint::contextExit(const nb::object &excType,
                                   const nb::object &excVal,
                                   const nb::object &excTb) {
  PyThreadContextEntry::popInsertionPoint(*this);
}

//------------------------------------------------------------------------------
// PyAttribute.
//------------------------------------------------------------------------------

bool PyAttribute::operator==(const PyAttribute &other) const {
  return aiirAttributeEqual(attr, other.attr);
}

nb::object PyAttribute::getCapsule() {
  return nb::steal<nb::object>(aiirPythonAttributeToCapsule(*this));
}

PyAttribute PyAttribute::createFromCapsule(const nb::object &capsule) {
  AiirAttribute rawAttr = aiirPythonCapsuleToAttribute(capsule.ptr());
  if (aiirAttributeIsNull(rawAttr))
    throw nb::python_error();
  return PyAttribute(
      PyAiirContext::forContext(aiirAttributeGetContext(rawAttr)), rawAttr);
}

nb::typed<nb::object, PyAttribute> PyAttribute::maybeDownCast() {
  AiirTypeID aiirTypeID = aiirAttributeGetTypeID(this->get());
  assert(!aiirTypeIDIsNull(aiirTypeID) &&
         "aiirTypeID was expected to be non-null.");
  std::optional<nb::callable> typeCaster = PyGlobals::get().lookupTypeCaster(
      aiirTypeID, aiirAttributeGetDialect(this->get()));
  // nb::rv_policy::move means use std::move to move the return value
  // contents into a new instance that will be owned by Python.
  nb::object thisObj = nb::cast(this, nb::rv_policy::move);
  if (!typeCaster)
    return thisObj;
  return typeCaster.value()(thisObj);
}

//------------------------------------------------------------------------------
// PyNamedAttribute.
//------------------------------------------------------------------------------

PyNamedAttribute::PyNamedAttribute(AiirAttribute attr, std::string ownedName)
    : ownedName(new std::string(std::move(ownedName))) {
  namedAttr = aiirNamedAttributeGet(
      aiirIdentifierGet(aiirAttributeGetContext(attr),
                        toAiirStringRef(*this->ownedName)),
      attr);
}

//------------------------------------------------------------------------------
// PyType.
//------------------------------------------------------------------------------

bool PyType::operator==(const PyType &other) const {
  return aiirTypeEqual(type, other.type);
}

nb::object PyType::getCapsule() {
  return nb::steal<nb::object>(aiirPythonTypeToCapsule(*this));
}

PyType PyType::createFromCapsule(nb::object capsule) {
  AiirType rawType = aiirPythonCapsuleToType(capsule.ptr());
  if (aiirTypeIsNull(rawType))
    throw nb::python_error();
  return PyType(PyAiirContext::forContext(aiirTypeGetContext(rawType)),
                rawType);
}

nb::typed<nb::object, PyType> PyType::maybeDownCast() {
  AiirTypeID aiirTypeID = aiirTypeGetTypeID(this->get());
  assert(!aiirTypeIDIsNull(aiirTypeID) &&
         "aiirTypeID was expected to be non-null.");
  std::optional<nb::callable> typeCaster = PyGlobals::get().lookupTypeCaster(
      aiirTypeID, aiirTypeGetDialect(this->get()));
  // nb::rv_policy::move means use std::move to move the return value
  // contents into a new instance that will be owned by Python.
  nb::object thisObj = nb::cast(this, nb::rv_policy::move);
  if (!typeCaster)
    return thisObj;
  return typeCaster.value()(thisObj);
}

//------------------------------------------------------------------------------
// PyTypeID.
//------------------------------------------------------------------------------

nb::object PyTypeID::getCapsule() {
  return nb::steal<nb::object>(aiirPythonTypeIDToCapsule(*this));
}

PyTypeID PyTypeID::createFromCapsule(nb::object capsule) {
  AiirTypeID aiirTypeID = aiirPythonCapsuleToTypeID(capsule.ptr());
  if (aiirTypeIDIsNull(aiirTypeID))
    throw nb::python_error();
  return PyTypeID(aiirTypeID);
}
bool PyTypeID::operator==(const PyTypeID &other) const {
  return aiirTypeIDEqual(typeID, other.typeID);
}

//------------------------------------------------------------------------------
// PyValue and subclasses.
//------------------------------------------------------------------------------

nb::object PyValue::getCapsule() {
  return nb::steal<nb::object>(aiirPythonValueToCapsule(get()));
}

static PyOperationRef getValueOwnerRef(AiirValue value) {
  AiirOperation owner;
  if (aiirValueIsAOpResult(value))
    owner = aiirOpResultGetOwner(value);
  else if (aiirValueIsABlockArgument(value))
    owner = aiirBlockGetParentOperation(aiirBlockArgumentGetOwner(value));
  else
    assert(false && "Value must be an block arg or op result.");
  if (aiirOperationIsNull(owner))
    throw nb::python_error();
  AiirContext ctx = aiirOperationGetContext(owner);
  return PyOperation::forOperation(PyAiirContext::forContext(ctx), owner);
}

nb::typed<nb::object, std::variant<PyBlockArgument, PyOpResult, PyValue>>
PyValue::maybeDownCast() {
  AiirType type = aiirValueGetType(get());
  AiirTypeID aiirTypeID = aiirTypeGetTypeID(type);
  assert(!aiirTypeIDIsNull(aiirTypeID) &&
         "aiirTypeID was expected to be non-null.");
  std::optional<nb::callable> valueCaster =
      PyGlobals::get().lookupValueCaster(aiirTypeID, aiirTypeGetDialect(type));
  // nb::rv_policy::move means use std::move to move the return value
  // contents into a new instance that will be owned by Python.
  nb::object thisObj;
  if (aiirValueIsAOpResult(value))
    thisObj = nb::cast<PyOpResult>(*this, nb::rv_policy::move);
  else if (aiirValueIsABlockArgument(value))
    thisObj = nb::cast<PyBlockArgument>(*this, nb::rv_policy::move);
  else
    assert(false && "Value must be an block arg or op result.");
  if (valueCaster)
    return valueCaster.value()(thisObj);
  return thisObj;
}

PyValue PyValue::createFromCapsule(nb::object capsule) {
  AiirValue value = aiirPythonCapsuleToValue(capsule.ptr());
  if (aiirValueIsNull(value))
    throw nb::python_error();
  PyOperationRef ownerRef = getValueOwnerRef(value);
  return PyValue(ownerRef, value);
}

//------------------------------------------------------------------------------
// PySymbolTable.
//------------------------------------------------------------------------------

PySymbolTable::PySymbolTable(PyOperationBase &operation)
    : operation(operation.getOperation().getRef()) {
  symbolTable = aiirSymbolTableCreate(operation.getOperation().get());
  if (aiirSymbolTableIsNull(symbolTable)) {
    throw nb::type_error("Operation is not a Symbol Table.");
  }
}

nb::object PySymbolTable::dunderGetItem(const std::string &name) {
  operation->checkValid();
  AiirOperation symbol = aiirSymbolTableLookup(
      symbolTable, aiirStringRefCreate(name.data(), name.length()));
  if (aiirOperationIsNull(symbol))
    throw nb::key_error(
        join("Symbol '", name, "' not in the symbol table.").c_str());

  return PyOperation::forOperation(operation->getContext(), symbol,
                                   operation.getObject())
      ->createOpView();
}

void PySymbolTable::erase(PyOperationBase &symbol) {
  operation->checkValid();
  symbol.getOperation().checkValid();
  aiirSymbolTableErase(symbolTable, symbol.getOperation().get());
  // The operation is also erased, so we must invalidate it. There may be Python
  // references to this operation so we don't want to delete it from the list of
  // live operations here.
  symbol.getOperation().valid = false;
}

void PySymbolTable::dunderDel(const std::string &name) {
  nb::object operation = dunderGetItem(name);
  erase(nb::cast<PyOperationBase &>(operation));
}

PyStringAttribute PySymbolTable::insert(PyOperationBase &symbol) {
  operation->checkValid();
  symbol.getOperation().checkValid();
  AiirAttribute symbolAttr = aiirOperationGetAttributeByName(
      symbol.getOperation().get(), aiirSymbolTableGetSymbolAttributeName());
  if (aiirAttributeIsNull(symbolAttr))
    throw nb::value_error("Expected operation to have a symbol name.");
  return PyStringAttribute(
      symbol.getOperation().getContext(),
      aiirSymbolTableInsert(symbolTable, symbol.getOperation().get()));
}

PyStringAttribute PySymbolTable::getSymbolName(PyOperationBase &symbol) {
  // Op must already be a symbol.
  PyOperation &operation = symbol.getOperation();
  operation.checkValid();
  AiirStringRef attrName = aiirSymbolTableGetSymbolAttributeName();
  AiirAttribute existingNameAttr =
      aiirOperationGetAttributeByName(operation.get(), attrName);
  if (aiirAttributeIsNull(existingNameAttr))
    throw nb::value_error("Expected operation to have a symbol name.");
  return PyStringAttribute(symbol.getOperation().getContext(),
                           existingNameAttr);
}

void PySymbolTable::setSymbolName(PyOperationBase &symbol,
                                  const std::string &name) {
  // Op must already be a symbol.
  PyOperation &operation = symbol.getOperation();
  operation.checkValid();
  AiirStringRef attrName = aiirSymbolTableGetSymbolAttributeName();
  AiirAttribute existingNameAttr =
      aiirOperationGetAttributeByName(operation.get(), attrName);
  if (aiirAttributeIsNull(existingNameAttr))
    throw nb::value_error("Expected operation to have a symbol name.");
  AiirAttribute newNameAttr =
      aiirStringAttrGet(operation.getContext()->get(), toAiirStringRef(name));
  aiirOperationSetAttributeByName(operation.get(), attrName, newNameAttr);
}

PyStringAttribute PySymbolTable::getVisibility(PyOperationBase &symbol) {
  PyOperation &operation = symbol.getOperation();
  operation.checkValid();
  AiirStringRef attrName = aiirSymbolTableGetVisibilityAttributeName();
  AiirAttribute existingVisAttr =
      aiirOperationGetAttributeByName(operation.get(), attrName);
  if (aiirAttributeIsNull(existingVisAttr))
    throw nb::value_error("Expected operation to have a symbol visibility.");
  return PyStringAttribute(symbol.getOperation().getContext(), existingVisAttr);
}

void PySymbolTable::setVisibility(PyOperationBase &symbol,
                                  const std::string &visibility) {
  if (visibility != "public" && visibility != "private" &&
      visibility != "nested")
    throw nb::value_error(
        "Expected visibility to be 'public', 'private' or 'nested'");
  PyOperation &operation = symbol.getOperation();
  operation.checkValid();
  AiirStringRef attrName = aiirSymbolTableGetVisibilityAttributeName();
  AiirAttribute existingVisAttr =
      aiirOperationGetAttributeByName(operation.get(), attrName);
  if (aiirAttributeIsNull(existingVisAttr))
    throw nb::value_error("Expected operation to have a symbol visibility.");
  AiirAttribute newVisAttr = aiirStringAttrGet(operation.getContext()->get(),
                                               toAiirStringRef(visibility));
  aiirOperationSetAttributeByName(operation.get(), attrName, newVisAttr);
}

void PySymbolTable::replaceAllSymbolUses(const std::string &oldSymbol,
                                         const std::string &newSymbol,
                                         PyOperationBase &from) {
  PyOperation &fromOperation = from.getOperation();
  fromOperation.checkValid();
  if (aiirLogicalResultIsFailure(aiirSymbolTableReplaceAllSymbolUses(
          toAiirStringRef(oldSymbol), toAiirStringRef(newSymbol),
          from.getOperation())))

    throw nb::value_error("Symbol rename failed");
}

void PySymbolTable::walkSymbolTables(PyOperationBase &from,
                                     bool allSymUsesVisible,
                                     nb::object callback) {
  PyOperation &fromOperation = from.getOperation();
  fromOperation.checkValid();
  struct UserData {
    PyAiirContextRef context;
    nb::object callback;
    bool gotException;
    std::string exceptionWhat;
    nb::object exceptionType;
  };
  UserData userData{
      fromOperation.getContext(), std::move(callback), false, {}, {}};
  aiirSymbolTableWalkSymbolTables(
      fromOperation.get(), allSymUsesVisible,
      [](AiirOperation foundOp, bool isVisible, void *calleeUserDataVoid) {
        UserData *calleeUserData = static_cast<UserData *>(calleeUserDataVoid);
        auto pyFoundOp =
            PyOperation::forOperation(calleeUserData->context, foundOp);
        if (calleeUserData->gotException)
          return;
        try {
          calleeUserData->callback(pyFoundOp.getObject(), isVisible);
        } catch (nb::python_error &e) {
          calleeUserData->gotException = true;
          calleeUserData->exceptionWhat = e.what();
          calleeUserData->exceptionType = nb::borrow(e.type());
        }
      },
      static_cast<void *>(&userData));
  if (userData.gotException) {
    std::string message("Exception raised in callback: ");
    message.append(userData.exceptionWhat);
    throw std::runtime_error(message);
  }
}

void PyBlockArgument::bindDerived(ClassTy &c) {
  c.def_prop_ro(
      "owner",
      [](PyBlockArgument &self) {
        return PyBlock(self.getParentOperation(),
                       aiirBlockArgumentGetOwner(self.get()));
      },
      "Returns the block that owns this argument.");
  c.def_prop_ro(
      "arg_number",
      [](PyBlockArgument &self) {
        return aiirBlockArgumentGetArgNumber(self.get());
      },
      "Returns the position of this argument in the block's argument list.");
  c.def(
      "set_type",
      [](PyBlockArgument &self, PyType type) {
        return aiirBlockArgumentSetType(self.get(), type);
      },
      "type"_a, "Sets the type of this block argument.");
  c.def(
      "set_location",
      [](PyBlockArgument &self, PyLocation loc) {
        return aiirBlockArgumentSetLocation(self.get(), loc);
      },
      "loc"_a, "Sets the location of this block argument.");
}

PyBlockArgumentList::PyBlockArgumentList(PyOperationRef operation,
                                         AiirBlock block, intptr_t startIndex,
                                         intptr_t length, intptr_t step)
    : Sliceable(startIndex,
                length == -1 ? aiirBlockGetNumArguments(block) : length, step),
      operation(std::move(operation)), block(block) {}

void PyBlockArgumentList::bindDerived(ClassTy &c) {
  c.def_prop_ro(
      "types",
      [](PyBlockArgumentList &self) {
        return getValueTypes(self, self.operation->getContext());
      },
      "Returns a list of types for all arguments in this argument list.");
}

intptr_t PyBlockArgumentList::getRawNumElements() {
  operation->checkValid();
  return aiirBlockGetNumArguments(block);
}

PyBlockArgument PyBlockArgumentList::getRawElement(intptr_t pos) const {
  AiirValue argument = aiirBlockGetArgument(block, pos);
  return PyBlockArgument(operation, argument);
}

PyBlockArgumentList PyBlockArgumentList::slice(intptr_t startIndex,
                                               intptr_t length,
                                               intptr_t step) const {
  return PyBlockArgumentList(operation, block, startIndex, length, step);
}

PyOpOperandList::PyOpOperandList(PyOperationRef operation, intptr_t startIndex,
                                 intptr_t length, intptr_t step)
    : Sliceable(startIndex,
                length == -1 ? aiirOperationGetNumOperands(operation->get())
                             : length,
                step),
      operation(operation) {}

void PyOpOperandList::dunderSetItem(intptr_t index, PyValue value) {
  index = wrapIndex(index);
  aiirOperationSetOperand(operation->get(), index, value.get());
}

void PyOpOperandList::bindDerived(ClassTy &c) {
  c.def("__setitem__", &PyOpOperandList::dunderSetItem, "index"_a, "value"_a,
        "Sets the operand at the specified index to a new value.");
}

intptr_t PyOpOperandList::getRawNumElements() {
  operation->checkValid();
  return aiirOperationGetNumOperands(operation->get());
}

PyValue PyOpOperandList::getRawElement(intptr_t pos) {
  AiirValue operand = aiirOperationGetOperand(operation->get(), pos);
  PyOperationRef pyOwner = getValueOwnerRef(operand);
  return PyValue(pyOwner, operand);
}

PyOpOperandList PyOpOperandList::slice(intptr_t startIndex, intptr_t length,
                                       intptr_t step) const {
  return PyOpOperandList(operation, startIndex, length, step);
}

/// A list of OpOperands. Internally, these are stored as consecutive elements,
/// random access is cheap. The (returned) OpOperand list is associated with the
/// operation whose operands these are, and thus extends the lifetime of this
/// operation.
class PyOpOperands : public Sliceable<PyOpOperands, PyOpOperand> {
public:
  static constexpr const char *pyClassName = "OpOperands";
  using SliceableT = Sliceable<PyOpOperandList, PyOpOperand>;

  PyOpOperands(PyOperationRef operation, intptr_t startIndex = 0,
               intptr_t length = -1, intptr_t step = 1)
      : Sliceable(startIndex,
                  length == -1 ? aiirOperationGetNumOperands(operation->get())
                               : length,
                  step),
        operation(operation) {}

private:
  /// Give the parent CRTP class access to hook implementations below.
  friend class Sliceable<PyOpOperands, PyOpOperand>;

  intptr_t getRawNumElements() {
    operation->checkValid();
    return aiirOperationGetNumOperands(operation->get());
  }

  PyOpOperand getRawElement(intptr_t pos) {
    AiirOpOperand opOperand = aiirOperationGetOpOperand(operation->get(), pos);
    return PyOpOperand(opOperand);
  }

  PyOpOperands slice(intptr_t startIndex, intptr_t length, intptr_t step) {
    return PyOpOperands(operation, startIndex, length, step);
  }

  PyOperationRef operation;
};

PyOpSuccessors::PyOpSuccessors(PyOperationRef operation, intptr_t startIndex,
                               intptr_t length, intptr_t step)
    : Sliceable(startIndex,
                length == -1 ? aiirOperationGetNumSuccessors(operation->get())
                             : length,
                step),
      operation(operation) {}

void PyOpSuccessors::dunderSetItem(intptr_t index, PyBlock block) {
  index = wrapIndex(index);
  aiirOperationSetSuccessor(operation->get(), index, block.get());
}

void PyOpSuccessors::bindDerived(ClassTy &c) {
  c.def("__setitem__", &PyOpSuccessors::dunderSetItem, "index"_a, "block"_a,
        "Sets the successor block at the specified index.");
}

intptr_t PyOpSuccessors::getRawNumElements() {
  operation->checkValid();
  return aiirOperationGetNumSuccessors(operation->get());
}

PyBlock PyOpSuccessors::getRawElement(intptr_t pos) {
  AiirBlock block = aiirOperationGetSuccessor(operation->get(), pos);
  return PyBlock(operation, block);
}

PyOpSuccessors PyOpSuccessors::slice(intptr_t startIndex, intptr_t length,
                                     intptr_t step) const {
  return PyOpSuccessors(operation, startIndex, length, step);
}

PyBlockSuccessors::PyBlockSuccessors(PyBlock block, PyOperationRef operation,
                                     intptr_t startIndex, intptr_t length,
                                     intptr_t step)
    : Sliceable(startIndex,
                length == -1 ? aiirBlockGetNumSuccessors(block.get()) : length,
                step),
      operation(operation), block(block) {}

intptr_t PyBlockSuccessors::getRawNumElements() {
  block.checkValid();
  return aiirBlockGetNumSuccessors(block.get());
}

PyBlock PyBlockSuccessors::getRawElement(intptr_t pos) {
  AiirBlock block = aiirBlockGetSuccessor(this->block.get(), pos);
  return PyBlock(operation, block);
}

PyBlockSuccessors PyBlockSuccessors::slice(intptr_t startIndex, intptr_t length,
                                           intptr_t step) const {
  return PyBlockSuccessors(block, operation, startIndex, length, step);
}

PyBlockPredecessors::PyBlockPredecessors(PyBlock block,
                                         PyOperationRef operation,
                                         intptr_t startIndex, intptr_t length,
                                         intptr_t step)
    : Sliceable(startIndex,
                length == -1 ? aiirBlockGetNumPredecessors(block.get())
                             : length,
                step),
      operation(operation), block(block) {}

intptr_t PyBlockPredecessors::getRawNumElements() {
  block.checkValid();
  return aiirBlockGetNumPredecessors(block.get());
}

PyBlock PyBlockPredecessors::getRawElement(intptr_t pos) {
  AiirBlock block = aiirBlockGetPredecessor(this->block.get(), pos);
  return PyBlock(operation, block);
}

PyBlockPredecessors PyBlockPredecessors::slice(intptr_t startIndex,
                                               intptr_t length,
                                               intptr_t step) const {
  return PyBlockPredecessors(block, operation, startIndex, length, step);
}

nb::typed<nb::object, PyAttribute>
PyOpAttributeMap::dunderGetItemNamed(const std::string &name) {
  AiirAttribute attr =
      aiirOperationGetAttributeByName(operation->get(), toAiirStringRef(name));
  if (aiirAttributeIsNull(attr)) {
    throw nb::key_error("attempt to access a non-existent attribute");
  }
  return PyAttribute(operation->getContext(), attr).maybeDownCast();
}

nb::typed<nb::object, std::optional<PyAttribute>>
PyOpAttributeMap::get(const std::string &key, nb::object defaultValue) {
  AiirAttribute attr =
      aiirOperationGetAttributeByName(operation->get(), toAiirStringRef(key));
  if (aiirAttributeIsNull(attr))
    return defaultValue;
  return PyAttribute(operation->getContext(), attr).maybeDownCast();
}

PyNamedAttribute PyOpAttributeMap::dunderGetItemIndexed(intptr_t index) {
  if (index < 0) {
    index += dunderLen();
  }
  if (index < 0 || index >= dunderLen()) {
    throw nb::index_error("attempt to access out of bounds attribute");
  }
  AiirNamedAttribute namedAttr =
      aiirOperationGetAttribute(operation->get(), index);
  return PyNamedAttribute(
      namedAttr.attribute,
      std::string(aiirIdentifierStr(namedAttr.name).data,
                  aiirIdentifierStr(namedAttr.name).length));
}

void PyOpAttributeMap::dunderSetItem(const std::string &name,
                                     const PyAttribute &attr) {
  aiirOperationSetAttributeByName(operation->get(), toAiirStringRef(name),
                                  attr);
}

void PyOpAttributeMap::dunderDelItem(const std::string &name) {
  int removed = aiirOperationRemoveAttributeByName(operation->get(),
                                                   toAiirStringRef(name));
  if (!removed)
    throw nb::key_error("attempt to delete a non-existent attribute");
}

intptr_t PyOpAttributeMap::dunderLen() {
  return aiirOperationGetNumAttributes(operation->get());
}

bool PyOpAttributeMap::dunderContains(const std::string &name) {
  return !aiirAttributeIsNull(
      aiirOperationGetAttributeByName(operation->get(), toAiirStringRef(name)));
}

void PyOpAttributeMap::forEachAttr(
    AiirOperation op, std::function<void(AiirStringRef, AiirAttribute)> fn) {
  intptr_t n = aiirOperationGetNumAttributes(op);
  for (intptr_t i = 0; i < n; ++i) {
    AiirNamedAttribute na = aiirOperationGetAttribute(op, i);
    AiirStringRef name = aiirIdentifierStr(na.name);
    fn(name, na.attribute);
  }
}

void PyOpAttributeMap::bind(nb::module_ &m) {
  nb::class_<PyOpAttributeMap>(m, "OpAttributeMap")
      .def("__contains__", &PyOpAttributeMap::dunderContains, "name"_a,
           "Checks if an attribute with the given name exists in the map.")
      .def("__len__", &PyOpAttributeMap::dunderLen,
           "Returns the number of attributes in the map.")
      .def("__getitem__", &PyOpAttributeMap::dunderGetItemNamed, "name"_a,
           "Gets an attribute by name.")
      .def("__getitem__", &PyOpAttributeMap::dunderGetItemIndexed, "index"_a,
           "Gets a named attribute by index.")
      .def("__setitem__", &PyOpAttributeMap::dunderSetItem, "name"_a, "attr"_a,
           "Sets an attribute with the given name.")
      .def("__delitem__", &PyOpAttributeMap::dunderDelItem, "name"_a,
           "Deletes an attribute with the given name.")
      .def("get", &PyOpAttributeMap::get, nb::arg("key"),
           nb::arg("default") = nb::none(),
           "Gets an attribute by name or the default value, if it does not "
           "exist.")
      .def(
          "__iter__",
          [](PyOpAttributeMap &self) -> nb::typed<nb::iterator, nb::str> {
            nb::list keys;
            PyOpAttributeMap::forEachAttr(
                self.operation->get(), [&](AiirStringRef name, AiirAttribute) {
                  keys.append(nb::str(name.data, name.length));
                });
            return nb::iter(keys);
          },
          "Iterates over attribute names.")
      .def(
          "keys",
          [](PyOpAttributeMap &self) -> nb::typed<nb::list, nb::str> {
            nb::list out;
            PyOpAttributeMap::forEachAttr(
                self.operation->get(), [&](AiirStringRef name, AiirAttribute) {
                  out.append(nb::str(name.data, name.length));
                });
            return out;
          },
          "Returns a list of attribute names.")
      .def(
          "values",
          [](PyOpAttributeMap &self) -> nb::typed<nb::list, PyAttribute> {
            nb::list out;
            PyOpAttributeMap::forEachAttr(
                self.operation->get(), [&](AiirStringRef, AiirAttribute attr) {
                  out.append(PyAttribute(self.operation->getContext(), attr)
                                 .maybeDownCast());
                });
            return out;
          },
          "Returns a list of attribute values.")
      .def(
          "items",
          [](PyOpAttributeMap &self)
              -> nb::typed<nb::list,
                           nb::typed<nb::tuple, nb::str, PyAttribute>> {
            nb::list out;
            PyOpAttributeMap::forEachAttr(
                self.operation->get(),
                [&](AiirStringRef name, AiirAttribute attr) {
                  out.append(nb::make_tuple(
                      nb::str(name.data, name.length),
                      PyAttribute(self.operation->getContext(), attr)
                          .maybeDownCast()));
                });
            return out;
          },
          "Returns a list of `(name, attribute)` tuples.");
}

void PyOpAdaptor::bind(nb::module_ &m) {
  nb::class_<PyOpAdaptor>(m, "OpAdaptor")
      .def(nb::init<nb::typed<nb::list, PyValue>, PyOpAttributeMap>(),
           "Creates an OpAdaptor with the given operands and attributes.",
           "operands"_a, "attributes"_a)
      .def(nb::init<nb::typed<nb::list, PyValue>, PyOpView &>(),
           "Creates an OpAdaptor with the given operands and operation view.",
           "operands"_a, "opview"_a)
      .def_prop_ro(
          "operands", [](PyOpAdaptor &self) { return self.operands; },
          "Returns the operands of the adaptor.")
      .def_prop_ro(
          "attributes", [](PyOpAdaptor &self) { return self.attributes; },
          "Returns the attributes of the adaptor.");
}

static AiirLogicalResult verifyTraitByMethod(AiirOperation op, void *userData,
                                             const char *methodName) {
  nb::handle targetObj(static_cast<PyObject *>(userData));
  if (!nb::hasattr(targetObj, methodName))
    return aiirLogicalResultSuccess();
  PyAiirContextRef ctx = PyAiirContext::forContext(aiirOperationGetContext(op));
  nb::object opView = PyOperation::forOperation(ctx, op)->createOpView();
  bool success = nb::cast<bool>(targetObj.attr(methodName)(opView));
  return success ? aiirLogicalResultSuccess() : aiirLogicalResultFailure();
};

static bool attachOpTrait(const nb::object &opName, AiirDynamicOpTrait trait,
                          PyAiirContext &context) {
  std::string opNameStr;
  if (opName.is_type()) {
    opNameStr = nb::cast<std::string>(opName.attr("OPERATION_NAME"));
  } else if (nb::isinstance<nb::str>(opName)) {
    opNameStr = nb::cast<std::string>(opName);
  } else {
    throw nb::type_error("the root argument must be a type or a string");
  }

  return aiirDynamicOpTraitAttach(
      trait, AiirStringRef{opNameStr.data(), opNameStr.size()}, context.get());
}

bool PyDynamicOpTrait::attach(const nb::object &opName,
                              const nb::object &target,
                              PyAiirContext &context) {
  if (!nb::hasattr(target, "verify_invariants") &&
      !nb::hasattr(target, "verify_region_invariants"))
    throw nb::type_error(
        "the target object must have at least one of 'verify_invariants' or "
        "'verify_region_invariants' methods");

  AiirDynamicOpTraitCallbacks callbacks;
  callbacks.construct = [](void *userData) {
    nb::handle(static_cast<PyObject *>(userData)).inc_ref();
  };
  callbacks.destruct = [](void *userData) {
    nb::handle(static_cast<PyObject *>(userData)).dec_ref();
  };

  callbacks.verifyTrait = [](AiirOperation op,
                             void *userData) -> AiirLogicalResult {
    return verifyTraitByMethod(op, userData, "verify_invariants");
  };
  callbacks.verifyRegionTrait = [](AiirOperation op,
                                   void *userData) -> AiirLogicalResult {
    return verifyTraitByMethod(op, userData, "verify_region_invariants");
  };

  // To ensure that the same dynamic trait gets the same TypeID despite how many
  // times `attach` is called, we store it as an attribute on the target class.
  if (!nb::hasattr(target, typeIDAttr)) {
    nb::setattr(target, typeIDAttr,
                nb::cast(PyTypeID(PyGlobals::get().allocateTypeID())));
  }
  AiirDynamicOpTrait trait = aiirDynamicOpTraitCreate(
      nb::cast<PyTypeID>(target.attr(typeIDAttr)).get(), callbacks,
      static_cast<void *>(target.ptr()));
  return attachOpTrait(opName, trait, context);
}

void PyDynamicOpTrait::bind(nb::module_ &m) {
  nb::class_<PyDynamicOpTrait> cls(m, "DynamicOpTrait");
  cls.attr("attach") = classmethod(
      [](const nb::object &cls, const nb::object &opName, nb::object target,
         DefaultingPyAiirContext context) {
        if (target.is_none())
          target = cls;
        return PyDynamicOpTrait::attach(opName, target, *context.get());
      },
      nb::arg("cls"), nb::arg("op_name"), nb::arg("target").none() = nb::none(),
      nb::arg("context").none() = nb::none(),
      "Attach the dynamic op trait subclass to the given operation name.");
}

bool PyDynamicOpTraits::IsTerminator::attach(const nb::object &opName,
                                             PyAiirContext &context) {
  AiirDynamicOpTrait trait = aiirDynamicOpTraitIsTerminatorCreate();
  return attachOpTrait(opName, trait, context);
}

void PyDynamicOpTraits::IsTerminator::bind(nb::module_ &m) {
  nb::class_<PyDynamicOpTraits::IsTerminator, PyDynamicOpTrait> cls(
      m, "IsTerminatorTrait");
  cls.attr(typeIDAttr) = PyTypeID(aiirDynamicOpTraitIsTerminatorGetTypeID());
  cls.attr("attach") = classmethod(
      [](const nb::object &cls, const nb::object &opName,
         DefaultingPyAiirContext context) {
        return PyDynamicOpTraits::IsTerminator::attach(opName, *context.get());
      },
      "Attach IsTerminator trait to the given operation name.", nb::arg("cls"),
      nb::arg("op_name"), nb::arg("context").none() = nb::none());
}

bool PyDynamicOpTraits::NoTerminator::attach(const nb::object &opName,
                                             PyAiirContext &context) {
  AiirDynamicOpTrait trait = aiirDynamicOpTraitNoTerminatorCreate();
  return attachOpTrait(opName, trait, context);
}

void PyDynamicOpTraits::NoTerminator::bind(nb::module_ &m) {
  nb::class_<PyDynamicOpTraits::NoTerminator, PyDynamicOpTrait> cls(
      m, "NoTerminatorTrait");
  cls.attr(typeIDAttr) = PyTypeID(aiirDynamicOpTraitNoTerminatorGetTypeID());
  cls.attr("attach") = classmethod(
      [](const nb::object &cls, const nb::object &opName,
         DefaultingPyAiirContext context) {
        return PyDynamicOpTraits::NoTerminator::attach(opName, *context.get());
      },
      "Attach NoTerminator trait to the given operation name.", nb::arg("cls"),
      nb::arg("op_name"), nb::arg("context").none() = nb::none());
}

} // namespace AIIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace aiir

namespace {

using namespace aiir::python::AIIR_BINDINGS_PYTHON_DOMAIN;

AiirLocation tracebackToLocation(AiirContext ctx) {
#if defined(Py_LIMITED_API)
  // Frame introspection C APIs are not available under the limited API.
  // Traceback-based auto-location is not supported; return unknown.
  return aiirLocationUnknownGet(ctx);
#else
  size_t framesLimit =
      PyGlobals::get().getTracebackLoc().locTracebackFramesLimit();
  // Use a thread_local here to avoid requiring a large amount of space.
  thread_local std::array<AiirLocation, PyGlobals::TracebackLoc::kMaxFrames>
      frames;
  size_t count = 0;

  nb::gil_scoped_acquire acquire;

  PyThreadState *tstate = PyThreadState_GET();
  PyFrameObject *next;
  PyFrameObject *pyFrame = PyThreadState_GetFrame(tstate);
  // In the increment expression:
  // 1. get the next prev frame;
  // 2. decrement the ref count on the current frame (in order that it can get
  //    gc'd, along with any objects in its closure and etc);
  // 3. set current = next.
  for (; pyFrame != nullptr && count < framesLimit;
       next = PyFrame_GetBack(pyFrame), Py_XDECREF(pyFrame), pyFrame = next) {
    PyCodeObject *code = PyFrame_GetCode(pyFrame);
    auto fileNameStr =
        nb::cast<std::string>(nb::borrow<nb::str>(code->co_filename));
    std::string_view fileName(fileNameStr);
    if (!PyGlobals::get().getTracebackLoc().isUserTracebackFilename(fileName))
      continue;

    // co_qualname and PyCode_Addr2Location added in py3.11
#if PY_VERSION_HEX < 0x030B00F0
    std::string name =
        nb::cast<std::string>(nb::borrow<nb::str>(code->co_name));
    std::string_view funcName(name);
    int startLine = PyFrame_GetLineNumber(pyFrame);
    AiirLocation loc = aiirLocationFileLineColGet(
        ctx, aiirStringRefCreate(fileName.data(), fileName.size()), startLine,
        0);
#else
    std::string name =
        nb::cast<std::string>(nb::borrow<nb::str>(code->co_qualname));
    std::string_view funcName(name);
    int startLine, startCol, endLine, endCol;
    int lasti = PyFrame_GetLasti(pyFrame);
    if (!PyCode_Addr2Location(code, lasti, &startLine, &startCol, &endLine,
                              &endCol)) {
      throw nb::python_error();
    }
    AiirLocation loc = aiirLocationFileLineColRangeGet(
        ctx, aiirStringRefCreate(fileName.data(), fileName.size()), startLine,
        startCol, endLine, endCol);
#endif

    frames[count] = aiirLocationNameGet(
        ctx, aiirStringRefCreate(funcName.data(), funcName.size()), loc);
    ++count;
  }
  // When the loop breaks (after the last iter), current frame (if non-null)
  // is leaked without this.
  Py_XDECREF(pyFrame);

  if (count == 0)
    return aiirLocationUnknownGet(ctx);

  AiirLocation callee = frames[0];
  assert(!aiirLocationIsNull(callee) && "expected non-null callee location");
  if (count == 1)
    return callee;

  AiirLocation caller = frames[count - 1];
  assert(!aiirLocationIsNull(caller) && "expected non-null caller location");
  for (int i = count - 2; i >= 1; i--)
    caller = aiirLocationCallSiteGet(frames[i], caller);

  return aiirLocationCallSiteGet(callee, caller);
#endif
}

PyLocation
maybeGetTracebackLocation(const std::optional<PyLocation> &location) {
  if (location.has_value())
    return location.value();
  if (!PyGlobals::get().getTracebackLoc().locTracebacksEnabled())
    return DefaultingPyLocation::resolve();

  PyAiirContext &ctx = DefaultingPyAiirContext::resolve();
  AiirLocation aiirLoc = tracebackToLocation(ctx.get());
  PyAiirContextRef ref = PyAiirContext::forContext(ctx.get());
  return {ref, aiirLoc};
}
} // namespace

namespace aiir {
namespace python {
namespace AIIR_BINDINGS_PYTHON_DOMAIN {

static std::string formatAIIRError(const AIIRError &e) {
  auto locStr = [](const PyLocation &loc) {
    PyPrintAccumulator accum;
    aiirLocationPrint(loc, accum.getCallback(), accum.getUserData());
    std::string s = nb::cast<std::string>(nb::str(accum.join()));
    std::string_view sv(s);
    if (sv.size() > 5) {
      sv.remove_prefix(4); // "loc("
      sv.remove_suffix(1); // ")"
    }
    return std::string(sv);
  };
  auto indent = [](std::string s) {
    size_t pos = 0;
    while ((pos = s.find('\n', pos)) != std::string::npos) {
      s.replace(pos, 1, "\n  ");
      pos += 3;
    }
    return s;
  };

  std::ostringstream os;
  os << e.message;
  if (!e.errorDiagnostics.empty())
    os << ":";
  for (const auto &diag : e.errorDiagnostics) {
    os << "\nerror: " << locStr(diag.location) << ": " << indent(diag.message);
    for (const auto &note : diag.notes) {
      os << "\n note: " << locStr(note.location) << ": "
         << indent(note.message);
    }
  }
  return os.str();
}

void AIIRError::bind(nb::module_ &m) {
  auto cls = nb::exception<AIIRError>(m, "AIIRError", PyExc_Exception);
  nb::register_exception_translator(
      [](const std::exception_ptr &p, void *payload) {
        try {
          if (p)
            std::rethrow_exception(p);
        } catch (AIIRError &e) {
          std::string formatted = formatAIIRError(e);
          nb::object ty = nb::borrow(static_cast<PyObject *>(payload));
          nb::object obj = ty(formatted);
          obj.attr("_message") = nb::cast(std::move(e.message));
          obj.attr("_error_diagnostics") =
              nb::cast(std::move(e.errorDiagnostics));
          PyErr_SetObject(static_cast<PyObject *>(payload), obj.ptr());
        }
      },
      cls.ptr());
  auto propertyType = nb::borrow<nb::type_object>(
      reinterpret_cast<PyObject *>(&PyProperty_Type));
  nb::setattr(
      cls, "message",
      propertyType(nb::cpp_function(
          [](nb::object self) -> nb::str { return self.attr("_message"); },
          nb::is_method())));
  nb::setattr(cls, "error_diagnostics",
              propertyType(nb::cpp_function(
                  [](nb::object self)
                      -> nb::typed<nb::list, PyDiagnostic::DiagnosticInfo> {
                    return self.attr("_error_diagnostics");
                  },
                  nb::is_method())));
}

void populateRoot(nb::module_ &m) {
  m.attr("T") = nb::type_var("T");
  m.attr("U") = nb::type_var("U");

  nb::class_<PyGlobals>(m, "_Globals")
      .def_prop_rw("dialect_search_modules",
                   &PyGlobals::getDialectSearchPrefixes,
                   &PyGlobals::setDialectSearchPrefixes)
      .def("append_dialect_search_prefix", &PyGlobals::addDialectSearchPrefix,
           "module_name"_a)
      .def(
          "_check_dialect_module_loaded",
          [](PyGlobals &self, const std::string &dialectNamespace) {
            return self.loadDialectModule(dialectNamespace);
          },
          "dialect_namespace"_a)
      .def("_register_dialect_impl", &PyGlobals::registerDialectImpl,
           "dialect_namespace"_a, "dialect_class"_a, nb::kw_only(),
           "replace"_a = false,
           "Testing hook for directly registering a dialect")
      .def("_register_operation_impl", &PyGlobals::registerOperationImpl,
           "operation_name"_a, "operation_class"_a, nb::kw_only(),
           "replace"_a = false,
           "Testing hook for directly registering an operation")
      .def("loc_tracebacks_enabled",
           [](PyGlobals &self) {
             return self.getTracebackLoc().locTracebacksEnabled();
           })
      .def("set_loc_tracebacks_enabled",
           [](PyGlobals &self, bool enabled) {
             self.getTracebackLoc().setLocTracebacksEnabled(enabled);
           })
      .def("loc_tracebacks_frame_limit",
           [](PyGlobals &self) {
             return self.getTracebackLoc().locTracebackFramesLimit();
           })
      .def("set_loc_tracebacks_frame_limit",
           [](PyGlobals &self, std::optional<int> n) {
             self.getTracebackLoc().setLocTracebackFramesLimit(
                 n.value_or(PyGlobals::TracebackLoc::kMaxFrames));
           })
      .def("register_traceback_file_inclusion",
           [](PyGlobals &self, const std::string &filename) {
             self.getTracebackLoc().registerTracebackFileInclusion(filename);
           })
      .def("register_traceback_file_exclusion",
           [](PyGlobals &self, const std::string &filename) {
             self.getTracebackLoc().registerTracebackFileExclusion(filename);
           });

  // Aside from making the globals accessible to python, having python manage
  // it is necessary to make sure it is destroyed (and releases its python
  // resources) properly.
  m.attr("globals") = nb::cast(new PyGlobals, nb::rv_policy::take_ownership);

  // Registration decorators.
  m.def(
      "register_dialect",
      [](nb::type_object pyClass) {
        std::string dialectNamespace =
            nb::cast<std::string>(pyClass.attr("DIALECT_NAMESPACE"));
        PyGlobals::get().registerDialectImpl(dialectNamespace, pyClass);
        return pyClass;
      },
      "dialect_class"_a,
      "Class decorator for registering a custom Dialect wrapper");
  m.def(
      "register_operation",
      [](const nb::type_object &dialectClass, bool replace) -> nb::object {
        return nb::cpp_function(
            [dialectClass,
             replace](nb::type_object opClass) -> nb::type_object {
              std::string operationName =
                  nb::cast<std::string>(opClass.attr("OPERATION_NAME"));
              PyGlobals::get().registerOperationImpl(operationName, opClass,
                                                     replace);
              // Dict-stuff the new opClass by name onto the dialect class.
              nb::object opClassName = opClass.attr("__name__");
              dialectClass.attr(opClassName) = opClass;
              return opClass;
            });
      },
      // clang-format off
      nb::sig("def register_operation(dialect_class: type, *, replace: bool = False) "
        "-> typing.Callable[[type[T]], type[T]]"),
      // clang-format on
      "dialect_class"_a, nb::kw_only(), "replace"_a = false,
      "Produce a class decorator for registering an Operation class as part of "
      "a dialect");
  m.def(
      "register_op_adaptor",
      [](const nb::type_object &opClass, bool replace) -> nb::object {
        return nb::cpp_function(
            [opClass,
             replace](nb::type_object adaptorClass) -> nb::type_object {
              std::string operationName =
                  nb::cast<std::string>(adaptorClass.attr("OPERATION_NAME"));
              PyGlobals::get().registerOpAdaptorImpl(operationName,
                                                     adaptorClass, replace);
              // Dict-stuff the new adaptorClass by name onto the opClass.
              opClass.attr("Adaptor") = adaptorClass;
              return adaptorClass;
            });
      },
      // clang-format off
      nb::sig("def register_op_adaptor(op_class: type, *, replace: bool = False) "
        "-> typing.Callable[[type[T]], type[T]]"),
      // clang-format on
      "op_class"_a, nb::kw_only(), "replace"_a = false,
      "Produce a class decorator for registering an OpAdaptor class for an "
      "operation.");
  m.def(
      AIIR_PYTHON_CAPI_TYPE_CASTER_REGISTER_ATTR,
      [](PyTypeID aiirTypeID, bool replace) -> nb::object {
        return nb::cpp_function([aiirTypeID, replace](
                                    nb::callable typeCaster) -> nb::object {
          PyGlobals::get().registerTypeCaster(aiirTypeID, typeCaster, replace);
          return typeCaster;
        });
      },
      // clang-format off
      nb::sig("def register_type_caster(typeid: _aiir.ir.TypeID, *, replace: bool = False) "
                        "-> typing.Callable[[typing.Callable[[T], U]], typing.Callable[[T], U]]"),
      // clang-format on
      "typeid"_a, nb::kw_only(), "replace"_a = false,
      "Register a type caster for casting AIIR types to custom user types.");
  m.def(
      AIIR_PYTHON_CAPI_VALUE_CASTER_REGISTER_ATTR,
      [](PyTypeID aiirTypeID, bool replace) -> nb::object {
        return nb::cpp_function(
            [aiirTypeID, replace](nb::callable valueCaster) -> nb::object {
              PyGlobals::get().registerValueCaster(aiirTypeID, valueCaster,
                                                   replace);
              return valueCaster;
            });
      },
      // clang-format off
      nb::sig("def register_value_caster(typeid: _aiir.ir.TypeID, *, replace: bool = False) "
                        "-> typing.Callable[[typing.Callable[[T], U]], typing.Callable[[T], U]]"),
      // clang-format on
      "typeid"_a, nb::kw_only(), "replace"_a = false,
      "Register a value caster for casting AIIR values to custom user values.");
}

//------------------------------------------------------------------------------
// Populates the core exports of the 'ir' submodule.
//------------------------------------------------------------------------------
void populateIRCore(nb::module_ &m) {
  //----------------------------------------------------------------------------
  // Enums.
  //----------------------------------------------------------------------------
  nb::enum_<PyDiagnosticSeverity>(m, "DiagnosticSeverity")
      .value("ERROR", PyDiagnosticSeverity::Error)
      .value("WARNING", PyDiagnosticSeverity::Warning)
      .value("NOTE", PyDiagnosticSeverity::Note)
      .value("REMARK", PyDiagnosticSeverity::Remark);

  nb::enum_<PyWalkOrder>(m, "WalkOrder")
      .value("PRE_ORDER", PyWalkOrder::PreOrder)
      .value("POST_ORDER", PyWalkOrder::PostOrder);
  nb::enum_<PyWalkResult>(m, "WalkResult")
      .value("ADVANCE", PyWalkResult::Advance)
      .value("INTERRUPT", PyWalkResult::Interrupt)
      .value("SKIP", PyWalkResult::Skip);

  //----------------------------------------------------------------------------
  // Mapping of Diagnostics.
  //----------------------------------------------------------------------------
  nb::class_<PyDiagnostic>(m, "Diagnostic")
      .def_prop_ro("severity", &PyDiagnostic::getSeverity,
                   "Returns the severity of the diagnostic.")
      .def_prop_ro("location", &PyDiagnostic::getLocation,
                   "Returns the location associated with the diagnostic.")
      .def_prop_ro("message", &PyDiagnostic::getMessage,
                   "Returns the message text of the diagnostic.")
      .def_prop_ro("notes", &PyDiagnostic::getNotes,
                   "Returns a tuple of attached note diagnostics.")
      .def(
          "__str__",
          [](PyDiagnostic &self) -> nb::str {
            if (!self.isValid())
              return nb::str("<Invalid Diagnostic>");
            return self.getMessage();
          },
          "Returns the diagnostic message as a string.");

  nb::class_<PyDiagnostic::DiagnosticInfo>(m, "DiagnosticInfo")
      .def(
          "__init__",
          [](PyDiagnostic::DiagnosticInfo &self, PyDiagnostic diag) {
            new (&self) PyDiagnostic::DiagnosticInfo(diag.getInfo());
          },
          "diag"_a, "Creates a DiagnosticInfo from a Diagnostic.")
      .def_ro("severity", &PyDiagnostic::DiagnosticInfo::severity,
              "The severity level of the diagnostic.")
      .def_ro("location", &PyDiagnostic::DiagnosticInfo::location,
              "The location associated with the diagnostic.")
      .def_ro("message", &PyDiagnostic::DiagnosticInfo::message,
              "The message text of the diagnostic.")
      .def_ro("notes", &PyDiagnostic::DiagnosticInfo::notes,
              "List of attached note diagnostics.")
      .def(
          "__str__",
          [](PyDiagnostic::DiagnosticInfo &self) { return self.message; },
          "Returns the diagnostic message as a string.");

  nb::class_<PyDiagnosticHandler>(m, "DiagnosticHandler")
      .def("detach", &PyDiagnosticHandler::detach,
           "Detaches the diagnostic handler from the context.")
      .def_prop_ro("attached", &PyDiagnosticHandler::isAttached,
                   "Returns True if the handler is attached to a context.")
      .def_prop_ro("had_error", &PyDiagnosticHandler::getHadError,
                   "Returns True if an error was encountered during diagnostic "
                   "handling.")
      .def("__enter__", &PyDiagnosticHandler::contextEnter,
           "Enters the diagnostic handler as a context manager.",
           nb::sig("def __enter__(self, /) -> DiagnosticHandler"))
      .def("__exit__", &PyDiagnosticHandler::contextExit, "exc_type"_a.none(),
           "exc_value"_a.none(), "traceback"_a.none(),
           "Exits the diagnostic handler context manager.");

  // Expose DefaultThreadPool to python
  nb::class_<PyThreadPool>(m, "ThreadPool")
      .def(
          "__init__", [](PyThreadPool &self) { new (&self) PyThreadPool(); },
          "Creates a new thread pool with default concurrency.")
      .def("get_max_concurrency", &PyThreadPool::getMaxConcurrency,
           "Returns the maximum number of threads in the pool.")
      .def("_aiir_thread_pool_ptr", &PyThreadPool::_aiir_thread_pool_ptr,
           "Returns the raw pointer to the LLVM thread pool as a string.");

  nb::class_<PyAiirContext>(m, "Context")
      .def(
          "__init__",
          [](PyAiirContext &self) {
            AiirContext context = aiirContextCreateWithThreading(false);
            new (&self) PyAiirContext(context);
          },
          R"(
            Creates a new AIIR context.

            The context is the top-level container for all AIIR objects. It owns the storage
            for types, attributes, locations, and other core IR objects. A context can be
            configured to allow or disallow unregistered dialects and can have dialects
            loaded on-demand.)")
      .def_static("_get_live_count", &PyAiirContext::getLiveCount,
                  "Gets the number of live Context objects.")
      .def(
          "_get_context_again",
          [](PyAiirContext &self) -> nb::typed<nb::object, PyAiirContext> {
            PyAiirContextRef ref = PyAiirContext::forContext(self.get());
            return ref.releaseObject();
          },
          "Gets another reference to the same context.")
      .def("_get_live_module_count", &PyAiirContext::getLiveModuleCount,
           "Gets the number of live modules owned by this context.")
      .def_prop_ro(AIIR_PYTHON_CAPI_PTR_ATTR, &PyAiirContext::getCapsule,
                   "Gets a capsule wrapping the AiirContext.")
      .def_static(AIIR_PYTHON_CAPI_FACTORY_ATTR,
                  &PyAiirContext::createFromCapsule,
                  "Creates a Context from a capsule wrapping AiirContext.")
      .def("__enter__", &PyAiirContext::contextEnter,
           "Enters the context as a context manager.",
           nb::sig("def __enter__(self, /) -> Context"))
      .def("__exit__", &PyAiirContext::contextExit, "exc_type"_a.none(),
           "exc_value"_a.none(), "traceback"_a.none(),
           "Exits the context manager.")
      .def_prop_ro_static(
          "current",
          [](nb::object & /*class*/)
              -> std::optional<nb::typed<nb::object, PyAiirContext>> {
            auto *context = PyThreadContextEntry::getDefaultContext();
            if (!context)
              return {};
            return nb::cast(context);
          },
          nb::sig("def current(/) -> Context | None"),
          "Gets the Context bound to the current thread or returns None if no "
          "context is set.")
      .def_prop_ro(
          "dialects",
          [](PyAiirContext &self) { return PyDialects(self.getRef()); },
          "Gets a container for accessing dialects by name.")
      .def_prop_ro(
          "d", [](PyAiirContext &self) { return PyDialects(self.getRef()); },
          "Alias for `dialects`.")
      .def(
          "get_dialect_descriptor",
          [=](PyAiirContext &self, std::string &name) {
            AiirDialect dialect = aiirContextGetOrLoadDialect(
                self.get(), {name.data(), name.size()});
            if (aiirDialectIsNull(dialect)) {
              throw nb::value_error(
                  join("Dialect '", name, "' not found").c_str());
            }
            return PyDialectDescriptor(self.getRef(), dialect);
          },
          "dialect_name"_a,
          "Gets or loads a dialect by name, returning its descriptor object.")
      .def_prop_rw(
          "allow_unregistered_dialects",
          [](PyAiirContext &self) -> bool {
            return aiirContextGetAllowUnregisteredDialects(self.get());
          },
          [](PyAiirContext &self, bool value) {
            aiirContextSetAllowUnregisteredDialects(self.get(), value);
          },
          "Controls whether unregistered dialects are allowed in this context.")
      .def("attach_diagnostic_handler", &PyAiirContext::attachDiagnosticHandler,
           "callback"_a,
           "Attaches a diagnostic handler that will receive callbacks.")
      .def(
          "enable_multithreading",
          [](PyAiirContext &self, bool enable) {
            aiirContextEnableMultithreading(self.get(), enable);
          },
          "enable"_a,
          R"(
            Enables or disables multi-threading support in the context.

            Args:
              enable: Whether to enable (True) or disable (False) multi-threading.
          )")
      .def(
          "set_thread_pool",
          [](PyAiirContext &self, PyThreadPool &pool) {
            // we should disable multi-threading first before setting
            // new thread pool otherwise the assert in
            // AIIRContext::setThreadPool will be raised.
            aiirContextEnableMultithreading(self.get(), false);
            aiirContextSetThreadPool(self.get(), pool.get());
          },
          R"(
            Sets a custom thread pool for the context to use.

            Args:
              pool: A ThreadPool object to use for parallel operations.

            Note:
              Multi-threading is automatically disabled before setting the thread pool.)")
      .def(
          "get_num_threads",
          [](PyAiirContext &self) {
            return aiirContextGetNumThreads(self.get());
          },
          "Gets the number of threads in the context's thread pool.")
      .def(
          "_aiir_thread_pool_ptr",
          [](PyAiirContext &self) {
            AiirLlvmThreadPool pool = aiirContextGetThreadPool(self.get());
            std::stringstream ss;
            ss << pool.ptr;
            return ss.str();
          },
          "Gets the raw pointer to the LLVM thread pool as a string.")
      .def(
          "is_registered_operation",
          [](PyAiirContext &self, std::string &name) {
            return aiirContextIsRegisteredOperation(
                self.get(), AiirStringRef{name.data(), name.size()});
          },
          "operation_name"_a,
          R"(
            Checks whether an operation with the given name is registered.

            Args:
              operation_name: The fully qualified name of the operation (e.g., `arith.addf`).

            Returns:
              True if the operation is registered, False otherwise.)")
      .def(
          "append_dialect_registry",
          [](PyAiirContext &self, PyDialectRegistry &registry) {
            aiirContextAppendDialectRegistry(self.get(), registry);
          },
          "registry"_a,
          R"(
            Appends the contents of a dialect registry to the context.

            Args:
              registry: A DialectRegistry containing dialects to append.)")
      .def_prop_rw("emit_error_diagnostics",
                   &PyAiirContext::getEmitErrorDiagnostics,
                   &PyAiirContext::setEmitErrorDiagnostics,
                   R"(
            Controls whether error diagnostics are emitted to diagnostic handlers.

            By default, error diagnostics are captured and reported through AIIRError exceptions.)")
      .def(
          "load_all_available_dialects",
          [](PyAiirContext &self) {
            aiirContextLoadAllAvailableDialects(self.get());
          },
          R"(
            Loads all dialects available in the registry into the context.

            This eagerly loads all dialects that have been registered, making them
            immediately available for use.)");

  //----------------------------------------------------------------------------
  // Mapping of PyDialectDescriptor
  //----------------------------------------------------------------------------
  nb::class_<PyDialectDescriptor>(m, "DialectDescriptor")
      .def_prop_ro(
          "namespace",
          [](PyDialectDescriptor &self) {
            AiirStringRef ns = aiirDialectGetNamespace(self.get());
            return nb::str(ns.data, ns.length);
          },
          "Returns the namespace of the dialect.")
      .def(
          "__repr__",
          [](PyDialectDescriptor &self) {
            AiirStringRef ns = aiirDialectGetNamespace(self.get());
            std::string repr("<DialectDescriptor ");
            repr.append(ns.data, ns.length);
            repr.append(">");
            return repr;
          },
          nb::sig("def __repr__(self) -> str"),
          "Returns a string representation of the dialect descriptor.");

  //----------------------------------------------------------------------------
  // Mapping of PyDialects
  //----------------------------------------------------------------------------
  nb::class_<PyDialects>(m, "Dialects")
      .def(
          "__getitem__",
          [=](PyDialects &self, std::string keyName) {
            AiirDialect dialect =
                self.getDialectForKey(keyName, /*attrError=*/false);
            nb::object descriptor =
                nb::cast(PyDialectDescriptor{self.getContext(), dialect});
            return createCustomDialectWrapper(keyName, std::move(descriptor));
          },
          "Gets a dialect by name using subscript notation.")
      .def(
          "__getattr__",
          [=](PyDialects &self, std::string attrName) {
            AiirDialect dialect =
                self.getDialectForKey(attrName, /*attrError=*/true);
            nb::object descriptor =
                nb::cast(PyDialectDescriptor{self.getContext(), dialect});
            return createCustomDialectWrapper(attrName, std::move(descriptor));
          },
          "Gets a dialect by name using attribute notation.");

  //----------------------------------------------------------------------------
  // Mapping of PyDialect
  //----------------------------------------------------------------------------
  nb::class_<PyDialect>(m, "Dialect")
      .def(nb::init<nb::object>(), "descriptor"_a,
           "Creates a Dialect from a DialectDescriptor.")
      .def_prop_ro(
          "descriptor", [](PyDialect &self) { return self.getDescriptor(); },
          "Returns the DialectDescriptor for this dialect.")
      .def(
          "__repr__",
          [](const nb::object &self) {
            auto clazz = self.attr("__class__");
            return nb::str("<Dialect ") +
                   self.attr("descriptor").attr("namespace") +
                   nb::str(" (class ") + clazz.attr("__module__") +
                   nb::str(".") + clazz.attr("__name__") + nb::str(")>");
          },
          nb::sig("def __repr__(self) -> str"),
          "Returns a string representation of the dialect.");

  //----------------------------------------------------------------------------
  // Mapping of PyDialectRegistry
  //----------------------------------------------------------------------------
  nb::class_<PyDialectRegistry>(m, "DialectRegistry")
      .def_prop_ro(AIIR_PYTHON_CAPI_PTR_ATTR, &PyDialectRegistry::getCapsule,
                   "Gets a capsule wrapping the AiirDialectRegistry.")
      .def_static(AIIR_PYTHON_CAPI_FACTORY_ATTR,
                  &PyDialectRegistry::createFromCapsule,
                  "Creates a DialectRegistry from a capsule wrapping "
                  "`AiirDialectRegistry`.")
      .def(nb::init<>(), "Creates a new empty dialect registry.");

  //----------------------------------------------------------------------------
  // Mapping of Location
  //----------------------------------------------------------------------------
  nb::class_<PyLocation>(m, "Location")
      .def_prop_ro(AIIR_PYTHON_CAPI_PTR_ATTR, &PyLocation::getCapsule,
                   "Gets a capsule wrapping the AiirLocation.")
      .def_static(AIIR_PYTHON_CAPI_FACTORY_ATTR, &PyLocation::createFromCapsule,
                  "Creates a Location from a capsule wrapping AiirLocation.")
      .def("__enter__", &PyLocation::contextEnter,
           "Enters the location as a context manager.",
           nb::sig("def __enter__(self, /) -> Location"))
      .def("__exit__", &PyLocation::contextExit, "exc_type"_a.none(),
           "exc_value"_a.none(), "traceback"_a.none(),
           "Exits the location context manager.")
      .def(
          "__eq__",
          [](PyLocation &self, PyLocation &other) -> bool {
            return aiirLocationEqual(self, other);
          },
          "Compares two locations for equality.")
      .def(
          "__eq__", [](PyLocation &self, nb::object other) { return false; },
          "Compares location with non-location object (always returns False).")
      .def_prop_ro_static(
          "current",
          [](nb::object & /*class*/) -> std::optional<PyLocation *> {
            auto *loc = PyThreadContextEntry::getDefaultLocation();
            if (!loc)
              return std::nullopt;
            return loc;
          },
          // clang-format off
          nb::sig("def current(/) -> Location | None"),
          // clang-format on
          "Gets the Location bound to the current thread or raises ValueError.")
      .def_static(
          "unknown",
          [](DefaultingPyAiirContext context) {
            return PyLocation(context->getRef(),
                              aiirLocationUnknownGet(context->get()));
          },
          "context"_a = nb::none(),
          "Gets a Location representing an unknown location.")
      .def_static(
          "callsite",
          [](PyLocation callee, const std::vector<PyLocation> &frames,
             DefaultingPyAiirContext context) {
            if (frames.empty())
              throw nb::value_error("No caller frames provided.");
            AiirLocation caller = frames.back().get();
            for (size_t index = frames.size() - 1; index-- > 0;) {
              caller = aiirLocationCallSiteGet(frames[index].get(), caller);
            }
            return PyLocation(context->getRef(),
                              aiirLocationCallSiteGet(callee.get(), caller));
          },
          "callee"_a, "frames"_a, "context"_a = nb::none(),
          "Gets a Location representing a caller and callsite.")
      .def("is_a_callsite", aiirLocationIsACallSite,
           "Returns True if this location is a CallSiteLoc.")
      .def_prop_ro(
          "callee",
          [](PyLocation &self) {
            return PyLocation(self.getContext(),
                              aiirLocationCallSiteGetCallee(self));
          },
          "Gets the callee location from a CallSiteLoc.")
      .def_prop_ro(
          "caller",
          [](PyLocation &self) {
            return PyLocation(self.getContext(),
                              aiirLocationCallSiteGetCaller(self));
          },
          "Gets the caller location from a CallSiteLoc.")
      .def_static(
          "file",
          [](std::string filename, int line, int col,
             DefaultingPyAiirContext context) {
            return PyLocation(
                context->getRef(),
                aiirLocationFileLineColGet(
                    context->get(), toAiirStringRef(filename), line, col));
          },
          "filename"_a, "line"_a, "col"_a, "context"_a = nb::none(),
          "Gets a Location representing a file, line and column.")
      .def_static(
          "file",
          [](std::string filename, int startLine, int startCol, int endLine,
             int endCol, DefaultingPyAiirContext context) {
            return PyLocation(context->getRef(),
                              aiirLocationFileLineColRangeGet(
                                  context->get(), toAiirStringRef(filename),
                                  startLine, startCol, endLine, endCol));
          },
          "filename"_a, "start_line"_a, "start_col"_a, "end_line"_a,
          "end_col"_a, "context"_a = nb::none(),
          "Gets a Location representing a file, line and column range.")
      .def("is_a_file", aiirLocationIsAFileLineColRange,
           "Returns True if this location is a FileLineColLoc.")
      .def_prop_ro(
          "filename",
          [](PyLocation loc) {
            return aiirIdentifierStr(
                aiirLocationFileLineColRangeGetFilename(loc));
          },
          "Gets the filename from a FileLineColLoc.")
      .def_prop_ro("start_line", aiirLocationFileLineColRangeGetStartLine,
                   "Gets the start line number from a `FileLineColLoc`.")
      .def_prop_ro("start_col", aiirLocationFileLineColRangeGetStartColumn,
                   "Gets the start column number from a `FileLineColLoc`.")
      .def_prop_ro("end_line", aiirLocationFileLineColRangeGetEndLine,
                   "Gets the end line number from a `FileLineColLoc`.")
      .def_prop_ro("end_col", aiirLocationFileLineColRangeGetEndColumn,
                   "Gets the end column number from a `FileLineColLoc`.")
      .def_static(
          "fused",
          [](const std::vector<PyLocation> &pyLocations,
             std::optional<PyAttribute> metadata,
             DefaultingPyAiirContext context) {
            std::vector<AiirLocation> locations;
            locations.reserve(pyLocations.size());
            for (const PyLocation &pyLocation : pyLocations)
              locations.push_back(pyLocation.get());
            AiirLocation location = aiirLocationFusedGet(
                context->get(), locations.size(), locations.data(),
                metadata ? metadata->get() : AiirAttribute{0});
            return PyLocation(context->getRef(), location);
          },
          "locations"_a, "metadata"_a = nb::none(), "context"_a = nb::none(),
          "Gets a Location representing a fused location with optional "
          "metadata.")
      .def("is_a_fused", aiirLocationIsAFused,
           "Returns True if this location is a `FusedLoc`.")
      .def_prop_ro(
          "locations",
          [](PyLocation &self) {
            unsigned numLocations = aiirLocationFusedGetNumLocations(self);
            std::vector<AiirLocation> locations(numLocations);
            if (numLocations)
              aiirLocationFusedGetLocations(self, locations.data());
            std::vector<PyLocation> pyLocations{};
            pyLocations.reserve(numLocations);
            for (unsigned i = 0; i < numLocations; ++i)
              pyLocations.emplace_back(self.getContext(), locations[i]);
            return pyLocations;
          },
          "Gets the list of locations from a `FusedLoc`.")
      .def_static(
          "name",
          [](std::string name, std::optional<PyLocation> childLoc,
             DefaultingPyAiirContext context) {
            return PyLocation(
                context->getRef(),
                aiirLocationNameGet(
                    context->get(), toAiirStringRef(name),
                    childLoc ? childLoc->get()
                             : aiirLocationUnknownGet(context->get())));
          },
          "name"_a, "childLoc"_a = nb::none(), "context"_a = nb::none(),
          "Gets a Location representing a named location with optional child "
          "location.")
      .def("is_a_name", aiirLocationIsAName,
           "Returns True if this location is a `NameLoc`.")
      .def_prop_ro(
          "name_str",
          [](PyLocation loc) {
            return aiirIdentifierStr(aiirLocationNameGetName(loc));
          },
          "Gets the name string from a `NameLoc`.")
      .def_prop_ro(
          "child_loc",
          [](PyLocation &self) {
            return PyLocation(self.getContext(),
                              aiirLocationNameGetChildLoc(self));
          },
          "Gets the child location from a `NameLoc`.")
      .def_static(
          "from_attr",
          [](PyAttribute &attribute, DefaultingPyAiirContext context) {
            return PyLocation(context->getRef(),
                              aiirLocationFromAttribute(attribute));
          },
          "attribute"_a, "context"_a = nb::none(),
          "Gets a Location from a `LocationAttr`.")
      .def_prop_ro(
          "context",
          [](PyLocation &self) -> nb::typed<nb::object, PyAiirContext> {
            return self.getContext().getObject();
          },
          "Context that owns the `Location`.")
      .def_prop_ro(
          "attr",
          [](PyLocation &self) {
            return PyAttribute(self.getContext(),
                               aiirLocationGetAttribute(self));
          },
          "Get the underlying `LocationAttr`.")
      .def(
          "emit_error",
          [](PyLocation &self, std::string message) {
            aiirEmitError(self, message.c_str());
          },
          "message"_a,
          R"(
            Emits an error diagnostic at this location.

            Args:
              message: The error message to emit.)")
      .def(
          "__repr__",
          [](PyLocation &self) {
            PyPrintAccumulator printAccum;
            aiirLocationPrint(self, printAccum.getCallback(),
                              printAccum.getUserData());
            return printAccum.join();
          },
          "Returns the assembly representation of the location.");

  //----------------------------------------------------------------------------
  // Mapping of Module
  //----------------------------------------------------------------------------
  nb::class_<PyModule>(m, "Module", nb::is_weak_referenceable())
      .def_prop_ro(AIIR_PYTHON_CAPI_PTR_ATTR, &PyModule::getCapsule,
                   "Gets a capsule wrapping the AiirModule.")
      .def_static(AIIR_PYTHON_CAPI_FACTORY_ATTR, &PyModule::createFromCapsule,
                  R"(
                    Creates a Module from a `AiirModule` wrapped by a capsule (i.e. `module._CAPIPtr`).

                    This returns a new object **BUT** `_clear_aiir_module(module)` must be called to
                    prevent double-frees (of the underlying `aiir::Module`).)")
      .def("_clear_aiir_module", &PyModule::clearAiirModule,
           R"(
             Clears the internal AIIR module reference.

             This is used internally to prevent double-free when ownership is transferred
             via the C API capsule mechanism. Not intended for normal use.)")
      .def_static(
          "parse",
          [](const std::string &moduleAsm, DefaultingPyAiirContext context)
              -> nb::typed<nb::object, PyModule> {
            PyAiirContext::ErrorCapture errors(context->getRef());
            AiirModule module = aiirModuleCreateParse(
                context->get(), toAiirStringRef(moduleAsm));
            if (aiirModuleIsNull(module))
              throw AIIRError("Unable to parse module assembly", errors.take());
            return PyModule::forModule(module).releaseObject();
          },
          "asm"_a, "context"_a = nb::none(), kModuleParseDocstring)
      .def_static(
          "parse",
          [](nb::bytes moduleAsm, DefaultingPyAiirContext context)
              -> nb::typed<nb::object, PyModule> {
            PyAiirContext::ErrorCapture errors(context->getRef());
            AiirModule module = aiirModuleCreateParse(
                context->get(), toAiirStringRef(moduleAsm));
            if (aiirModuleIsNull(module))
              throw AIIRError("Unable to parse module assembly", errors.take());
            return PyModule::forModule(module).releaseObject();
          },
          "asm"_a, "context"_a = nb::none(), kModuleParseDocstring)
      .def_static(
          "parseFile",
          [](const std::string &path, DefaultingPyAiirContext context)
              -> nb::typed<nb::object, PyModule> {
            PyAiirContext::ErrorCapture errors(context->getRef());
            AiirModule module = aiirModuleCreateParseFromFile(
                context->get(), toAiirStringRef(path));
            if (aiirModuleIsNull(module))
              throw AIIRError("Unable to parse module assembly", errors.take());
            return PyModule::forModule(module).releaseObject();
          },
          "path"_a, "context"_a = nb::none(), kModuleParseDocstring)
      .def_static(
          "create",
          [](const std::optional<PyLocation> &loc)
              -> nb::typed<nb::object, PyModule> {
            PyLocation pyLoc = maybeGetTracebackLocation(loc);
            AiirModule module = aiirModuleCreateEmpty(pyLoc.get());
            return PyModule::forModule(module).releaseObject();
          },
          "loc"_a = nb::none(), "Creates an empty module.")
      .def_prop_ro(
          "context",
          [](PyModule &self) -> nb::typed<nb::object, PyAiirContext> {
            return self.getContext().getObject();
          },
          "Context that created the `Module`.")
      .def_prop_ro(
          "operation",
          [](PyModule &self) -> nb::typed<nb::object, PyOperation> {
            return PyOperation::forOperation(self.getContext(),
                                             aiirModuleGetOperation(self.get()),
                                             self.getRef().releaseObject())
                .releaseObject();
          },
          "Accesses the module as an operation.")
      .def_prop_ro(
          "body",
          [](PyModule &self) {
            PyOperationRef moduleOp = PyOperation::forOperation(
                self.getContext(), aiirModuleGetOperation(self.get()),
                self.getRef().releaseObject());
            PyBlock returnBlock(moduleOp, aiirModuleGetBody(self.get()));
            return returnBlock;
          },
          "Return the block for this module.")
      .def(
          "dump",
          [](PyModule &self) {
            aiirOperationDump(aiirModuleGetOperation(self.get()));
          },
          kDumpDocstring)
      .def(
          "__str__",
          [](const nb::object &self) {
            // Defer to the operation's __str__.
            return self.attr("operation").attr("__str__")();
          },
          nb::sig("def __str__(self) -> str"),
          R"(
            Gets the assembly form of the operation with default options.

            If more advanced control over the assembly formatting or I/O options is needed,
            use the dedicated print or get_asm method, which supports keyword arguments to
            customize behavior.
          )")
      .def(
          "__eq__",
          [](PyModule &self, PyModule &other) {
            return aiirModuleEqual(self.get(), other.get());
          },
          "other"_a, "Compares two modules for equality.")
      .def(
          "__hash__",
          [](PyModule &self) { return aiirModuleHashValue(self.get()); },
          "Returns the hash value of the module.");

  //----------------------------------------------------------------------------
  // Mapping of Operation.
  //----------------------------------------------------------------------------
  nb::class_<PyOperationBase>(m, "_OperationBase")
      .def_prop_ro(
          AIIR_PYTHON_CAPI_PTR_ATTR,
          [](PyOperationBase &self) {
            return self.getOperation().getCapsule();
          },
          "Gets a capsule wrapping the `AiirOperation`.")
      .def(
          "__eq__",
          [](PyOperationBase &self, PyOperationBase &other) {
            return aiirOperationEqual(self.getOperation().get(),
                                      other.getOperation().get());
          },
          "Compares two operations for equality.")
      .def(
          "__eq__",
          [](PyOperationBase &self, nb::object other) { return false; },
          "Compares operation with non-operation object (always returns "
          "False).")
      .def(
          "__hash__",
          [](PyOperationBase &self) {
            return aiirOperationHashValue(self.getOperation().get());
          },
          "Returns the hash value of the operation.")
      .def_prop_ro(
          "attributes",
          [](PyOperationBase &self) {
            return PyOpAttributeMap(self.getOperation().getRef());
          },
          "Returns a dictionary-like map of operation attributes.")
      .def_prop_ro(
          "context",
          [](PyOperationBase &self) -> nb::typed<nb::object, PyAiirContext> {
            PyOperation &concreteOperation = self.getOperation();
            concreteOperation.checkValid();
            return concreteOperation.getContext().getObject();
          },
          "Context that owns the operation.")
      .def_prop_ro(
          "name",
          [](PyOperationBase &self) {
            auto &concreteOperation = self.getOperation();
            concreteOperation.checkValid();
            AiirOperation operation = concreteOperation.get();
            return aiirIdentifierStr(aiirOperationGetName(operation));
          },
          "Returns the fully qualified name of the operation.")
      .def_prop_ro(
          "operands",
          [](PyOperationBase &self) {
            return PyOpOperandList(self.getOperation().getRef());
          },
          "Returns the list of operation operands.")
      .def_prop_ro(
          "op_operands",
          [](PyOperationBase &self) {
            return PyOpOperands(self.getOperation().getRef());
          },
          "Returns the list of op operands.")
      .def_prop_ro(
          "regions",
          [](PyOperationBase &self) {
            return PyRegionList(self.getOperation().getRef());
          },
          "Returns the list of operation regions.")
      .def_prop_ro(
          "results",
          [](PyOperationBase &self) {
            return PyOpResultList(self.getOperation().getRef());
          },
          "Returns the list of Operation results.")
      .def_prop_ro(
          "result",
          [](PyOperationBase &self) -> nb::typed<nb::object, PyOpResult> {
            auto &operation = self.getOperation();
            return PyOpResult(operation.getRef(), getUniqueResult(operation))
                .maybeDownCast();
          },
          "Shortcut to get an op result if it has only one (throws an error "
          "otherwise).")
      .def_prop_rw(
          "location",
          [](PyOperationBase &self) {
            PyOperation &operation = self.getOperation();
            return PyLocation(operation.getContext(),
                              aiirOperationGetLocation(operation.get()));
          },
          [](PyOperationBase &self, const PyLocation &location) {
            PyOperation &operation = self.getOperation();
            aiirOperationSetLocation(operation.get(), location.get());
          },
          nb::for_getter("Returns the source location the operation was "
                         "defined or derived from."),
          nb::for_setter("Sets the source location the operation was defined "
                         "or derived from."))
      .def_prop_ro(
          "parent",
          [](PyOperationBase &self)
              -> std::optional<nb::typed<nb::object, PyOperation>> {
            auto parent = self.getOperation().getParentOperation();
            if (parent)
              return parent->getObject();
            return {};
          },
          "Returns the parent operation, or `None` if at top level.")
      .def(
          "__str__",
          [](PyOperationBase &self) {
            return self.getAsm(/*binary=*/false,
                               /*largeElementsLimit=*/std::nullopt,
                               /*largeResourceLimit=*/std::nullopt,
                               /*enableDebugInfo=*/false,
                               /*prettyDebugInfo=*/false,
                               /*printGenericOpForm=*/false,
                               /*useLocalScope=*/false,
                               /*useNameLocAsPrefix=*/false,
                               /*assumeVerified=*/false,
                               /*skipRegions=*/false);
          },
          nb::sig("def __str__(self) -> str"),
          "Returns the assembly form of the operation.")
      .def("print",
           nb::overload_cast<PyAsmState &, nb::object, bool>(
               &PyOperationBase::print),
           "state"_a, "file"_a = nb::none(), "binary"_a = false,
           R"(
             Prints the assembly form of the operation to a file like object.

             Args:
               state: `AsmState` capturing the operation numbering and flags.
               file: Optional file like object to write to. Defaults to sys.stdout.
               binary: Whether to write `bytes` (True) or `str` (False). Defaults to False.)")
      .def("print",
           nb::overload_cast<std::optional<int64_t>, std::optional<int64_t>,
                             bool, bool, bool, bool, bool, bool, nb::object,
                             bool, bool>(&PyOperationBase::print),
           // Careful: Lots of arguments must match up with print method.
           "large_elements_limit"_a = nb::none(),
           "large_resource_limit"_a = nb::none(), "enable_debug_info"_a = false,
           "pretty_debug_info"_a = false, "print_generic_op_form"_a = false,
           "use_local_scope"_a = false, "use_name_loc_as_prefix"_a = false,
           "assume_verified"_a = false, "file"_a = nb::none(),
           "binary"_a = false, "skip_regions"_a = false,
           R"(
             Prints the assembly form of the operation to a file like object.

             Args:
               large_elements_limit: Whether to elide elements attributes above this
                 number of elements. Defaults to None (no limit).
               large_resource_limit: Whether to elide resource attributes above this
                 number of characters. Defaults to None (no limit). If large_elements_limit
                 is set and this is None, the behavior will be to use large_elements_limit
                 as large_resource_limit.
               enable_debug_info: Whether to print debug/location information. Defaults
                 to False.
               pretty_debug_info: Whether to format debug information for easier reading
                 by a human (warning: the result is unparseable). Defaults to False.
               print_generic_op_form: Whether to print the generic assembly forms of all
                 ops. Defaults to False.
               use_local_scope: Whether to print in a way that is more optimized for
                 multi-threaded access but may not be consistent with how the overall
                 module prints.
               use_name_loc_as_prefix: Whether to use location attributes (NameLoc) as
                 prefixes for the SSA identifiers. Defaults to False.
               assume_verified: By default, if not printing generic form, the verifier
                 will be run and if it fails, generic form will be printed with a comment
                 about failed verification. While a reasonable default for interactive use,
                 for systematic use, it is often better for the caller to verify explicitly
                 and report failures in a more robust fashion. Set this to True if doing this
                 in order to avoid running a redundant verification. If the IR is actually
                 invalid, behavior is undefined.
               file: The file like object to write to. Defaults to sys.stdout.
               binary: Whether to write bytes (True) or str (False). Defaults to False.
               skip_regions: Whether to skip printing regions. Defaults to False.)")
      .def("write_bytecode", &PyOperationBase::writeBytecode, "file"_a,
           "desired_version"_a = nb::none(),
           R"(
             Write the bytecode form of the operation to a file like object.

             Args:
               file: The file like object to write to.
               desired_version: Optional version of bytecode to emit.
             Returns:
               The bytecode writer status.)")
      .def("get_asm", &PyOperationBase::getAsm,
           // Careful: Lots of arguments must match up with get_asm method.
           "binary"_a = false, "large_elements_limit"_a = nb::none(),
           "large_resource_limit"_a = nb::none(), "enable_debug_info"_a = false,
           "pretty_debug_info"_a = false, "print_generic_op_form"_a = false,
           "use_local_scope"_a = false, "use_name_loc_as_prefix"_a = false,
           "assume_verified"_a = false, "skip_regions"_a = false,
           R"(
            Gets the assembly form of the operation with all options available.

            Args:
              binary: Whether to return a bytes (True) or str (False) object. Defaults to
                False.
              ... others ...: See the print() method for common keyword arguments for
                configuring the printout.
            Returns:
              Either a bytes or str object, depending on the setting of the `binary`
              argument.)")
      .def("verify", &PyOperationBase::verify,
           "Verify the operation. Raises AIIRError if verification fails, and "
           "returns true otherwise.")
      .def("move_after", &PyOperationBase::moveAfter, "other"_a,
           "Puts self immediately after the other operation in its parent "
           "block.")
      .def("move_before", &PyOperationBase::moveBefore, "other"_a,
           "Puts self immediately before the other operation in its parent "
           "block.")
      .def("is_before_in_block", &PyOperationBase::isBeforeInBlock, "other"_a,
           R"(
             Checks if this operation is before another in the same block.

             Args:
               other: Another operation in the same parent block.

             Returns:
               True if this operation is before `other` in the operation list of the parent block.)")
      .def(
          "clone",
          [](PyOperationBase &self,
             const nb::object &ip) -> nb::typed<nb::object, PyOperation> {
            return self.getOperation().clone(ip);
          },
          "ip"_a = nb::none(),
          R"(
            Creates a deep copy of the operation.

            Args:
              ip: Optional insertion point where the cloned operation should be inserted.
                If None, the current insertion point is used. If False, the operation
                remains detached.

            Returns:
              A new Operation that is a clone of this operation.)")
      .def(
          "detach_from_parent",
          [](PyOperationBase &self) -> nb::typed<nb::object, PyOpView> {
            PyOperation &operation = self.getOperation();
            operation.checkValid();
            if (!operation.isAttached())
              throw nb::value_error("Detached operation has no parent.");

            operation.detachFromParent();
            return operation.createOpView();
          },
          "Detaches the operation from its parent block.")
      .def_prop_ro(
          "attached",
          [](PyOperationBase &self) {
            PyOperation &operation = self.getOperation();
            operation.checkValid();
            return operation.isAttached();
          },
          "Reports if the operation is attached to its parent block.")
      .def(
          "erase", [](PyOperationBase &self) { self.getOperation().erase(); },
          R"(
            Erases the operation and frees its memory.

            Note:
              After erasing, any Python references to the operation become invalid.)")
      .def(
          "walk",
          [](PyOperationBase &self,
             std::function<PyWalkResult(AiirOperation)> callback,
             PyWalkOrder walkOrder, std::optional<nb::object> opClass) {
            if (!opClass)
              return self.walk(callback, walkOrder);
            self.walk(
                [&](AiirOperation aiirOp) -> PyWalkResult {
                  nb::object opview =
                      PyOperation::forOperation(
                          self.getOperation().getContext(), aiirOp)
                          ->createOpView();
                  if (nb::isinstance(opview, *opClass))
                    return callback(aiirOp);
                  return PyWalkResult::Advance;
                },
                walkOrder);
          },
          "callback"_a, "walk_order"_a = PyWalkOrder::PostOrder,
          "op_class"_a = nb::none(),
          // clang-format off
           nb::sig("def walk(self, callback: Callable[[Operation], WalkResult], walk_order: WalkOrder = ..., op_class: type[OpView] | None = None) -> None"),
          // clang-format on
          R"(
             Walks the operation tree with a callback function.

             If op_class is provided, the callback is only invoked on operations
             of that type; all other operations are skipped silently.

             Args:
               callback: A callable that takes an Operation and returns a WalkResult.
               walk_order: The order of traversal (PRE_ORDER or POST_ORDER).
               op_class: If provided, only operations of this type are passed to the callback.)")
      .def(
          "has_trait",
          [](PyOperationBase &self, nb::type_object &traitCls) {
            PyTypeID traitTypeID =
                nb::cast<PyTypeID>(traitCls.attr(PyDynamicOpTrait::typeIDAttr));
            AiirIdentifier opName =
                aiirOperationGetName(self.getOperation().get());
            return aiirOperationNameHasTrait(
                aiirIdentifierStr(opName), traitTypeID.get(),
                self.getOperation().getContext()->get());
          },
          "trait_cls"_a, "Checks if the operation has a given trait.");

  nb::class_<PyOperation, PyOperationBase>(m, "Operation")
      .def_static(
          "create",
          [](std::string_view name,
             std::optional<std::vector<PyType *>> results,
             std::optional<std::vector<PyValue *>> operands,
             std::optional<nb::typed<nb::dict, nb::str, PyAttribute>>
                 attributes,
             std::optional<std::vector<PyBlock *>> successors, int regions,
             const std::optional<PyLocation> &location,
             const nb::object &maybeIp,
             bool inferType) -> nb::typed<nb::object, PyOperation> {
            // Unpack/validate operands.
            std::vector<AiirValue> aiirOperands;
            if (operands) {
              aiirOperands.reserve(operands->size());
              for (PyValue *operand : *operands) {
                if (!operand)
                  throw nb::value_error("operand value cannot be None");
                aiirOperands.push_back(operand->get());
              }
            }

            PyLocation pyLoc = maybeGetTracebackLocation(location);
            return PyOperation::create(
                name, results, aiirOperands.data(), aiirOperands.size(),
                attributes, successors, regions, pyLoc, maybeIp, inferType);
          },
          "name"_a, "results"_a = nb::none(), "operands"_a = nb::none(),
          "attributes"_a = nb::none(), "successors"_a = nb::none(),
          "regions"_a = 0, "loc"_a = nb::none(), "ip"_a = nb::none(),
          "infer_type"_a = false,
          R"(
            Creates a new operation.

            Args:
              name: Operation name (e.g. `dialect.operation`).
              results: Optional sequence of Type representing op result types.
              operands: Optional operands of the operation.
              attributes: Optional Dict of {str: Attribute}.
              successors: Optional List of Block for the operation's successors.
              regions: Number of regions to create (default = 0).
              location: Optional Location object (defaults to resolve from context manager).
              ip: Optional InsertionPoint (defaults to resolve from context manager or set to False to disable insertion, even with an insertion point set in the context manager).
              infer_type: Whether to infer result types (default = False).
            Returns:
              A new detached Operation object. Detached operations can be added to blocks, which causes them to become attached.)")
      .def_static(
          "parse",
          [](const std::string &sourceStr, const std::string &sourceName,
             DefaultingPyAiirContext context)
              -> nb::typed<nb::object, PyOpView> {
            return PyOperation::parse(context->getRef(), sourceStr, sourceName)
                ->createOpView();
          },
          "source"_a, nb::kw_only(), "source_name"_a = "",
          "context"_a = nb::none(),
          "Parses an operation. Supports both text assembly format and binary "
          "bytecode format.")
      .def_prop_ro(AIIR_PYTHON_CAPI_PTR_ATTR, &PyOperation::getCapsule,
                   "Gets a capsule wrapping the AiirOperation.")
      .def_static(AIIR_PYTHON_CAPI_FACTORY_ATTR,
                  &PyOperation::createFromCapsule,
                  "Creates an Operation from a capsule wrapping AiirOperation.")
      .def_prop_ro(
          "operation",
          [](nb::object self) -> nb::typed<nb::object, PyOperation> {
            return self;
          },
          "Returns self (the operation).")
      .def_prop_ro(
          "opview",
          [](PyOperation &self) -> nb::typed<nb::object, PyOpView> {
            return self.createOpView();
          },
          R"(
            Returns an OpView of this operation.

            Note:
              If the operation has a registered and loaded dialect then this OpView will
              be concrete wrapper class.)")
      .def_prop_ro("block", &PyOperation::getBlock,
                   "Returns the block containing this operation.")
      .def_prop_ro(
          "successors",
          [](PyOperationBase &self) {
            return PyOpSuccessors(self.getOperation().getRef());
          },
          "Returns the list of Operation successors.")
      .def(
          "replace_uses_of_with",
          [](PyOperation &self, PyValue &of, PyValue &with) {
            aiirOperationReplaceUsesOfWith(self.get(), of.get(), with.get());
          },
          "of"_a, "with_"_a,
          "Replaces uses of the 'of' value with the 'with' value inside the "
          "operation.")
      .def("_set_invalid", &PyOperation::setInvalid,
           "Invalidate the operation.");

  auto opViewClass =
      nb::class_<PyOpView, PyOperationBase>(m, "OpView")
          .def(nb::init<nb::typed<nb::object, PyOperation>>(), "operation"_a)
          .def(
              "__init__",
              [](PyOpView *self, std::string_view name,
                 std::tuple<int, bool> opRegionSpec,
                 nb::object operandSegmentSpecObj,
                 nb::object resultSegmentSpecObj,
                 std::optional<nb::sequence> resultTypeList,
                 nb::sequence operandList,
                 std::optional<nb::typed<nb::dict, nb::str, PyAttribute>>
                     attributes,
                 std::optional<std::vector<PyBlock *>> successors,
                 std::optional<int> regions,
                 const std::optional<PyLocation> &location,
                 const nb::object &maybeIp) {
                PyLocation pyLoc = maybeGetTracebackLocation(location);
                new (self) PyOpView(PyOpView::buildGeneric(
                    name, opRegionSpec, operandSegmentSpecObj,
                    resultSegmentSpecObj, resultTypeList, operandList,
                    attributes, successors, regions, pyLoc, maybeIp));
              },
              "name"_a, "opRegionSpec"_a,
              "operandSegmentSpecObj"_a = nb::none(),
              "resultSegmentSpecObj"_a = nb::none(), "results"_a = nb::none(),
              "operands"_a = nb::none(), "attributes"_a = nb::none(),
              "successors"_a = nb::none(), "regions"_a = nb::none(),
              "loc"_a = nb::none(), "ip"_a = nb::none())
          .def_prop_ro(
              "operation",
              [](PyOpView &self) -> nb::typed<nb::object, PyOperation> {
                return self.getOperationObject();
              })
          .def_prop_ro("opview",
                       [](nb::object self) -> nb::typed<nb::object, PyOpView> {
                         return self;
                       })
          .def(
              "__str__",
              [](PyOpView &self) { return nb::str(self.getOperationObject()); })
          .def_prop_ro(
              "successors",
              [](PyOperationBase &self) {
                return PyOpSuccessors(self.getOperation().getRef());
              },
              "Returns the list of Operation successors.")
          .def(
              "_set_invalid",
              [](PyOpView &self) { self.getOperation().setInvalid(); },
              "Invalidate the operation.");
  opViewClass.attr("_ODS_REGIONS") = nb::make_tuple(0, true);
  opViewClass.attr("_ODS_OPERAND_SEGMENTS") = nb::none();
  opViewClass.attr("_ODS_RESULT_SEGMENTS") = nb::none();
  // It is faster to pass the operation_name, ods_regions, and
  // ods_operand_segments/ods_result_segments as arguments to the constructor,
  // rather than to access them as attributes.
  opViewClass.attr("build_generic") = classmethod(
      [](nb::handle cls, std::optional<nb::sequence> resultTypeList,
         nb::sequence operandList,
         std::optional<nb::typed<nb::dict, nb::str, PyAttribute>> attributes,
         std::optional<std::vector<PyBlock *>> successors,
         std::optional<int> regions, std::optional<PyLocation> location,
         const nb::object &maybeIp) {
        std::string name = nb::cast<std::string>(cls.attr("OPERATION_NAME"));
        std::tuple<int, bool> opRegionSpec =
            nb::cast<std::tuple<int, bool>>(cls.attr("_ODS_REGIONS"));
        nb::object operandSegmentSpec = cls.attr("_ODS_OPERAND_SEGMENTS");
        nb::object resultSegmentSpec = cls.attr("_ODS_RESULT_SEGMENTS");
        PyLocation pyLoc = maybeGetTracebackLocation(location);
        return PyOpView::buildGeneric(name, opRegionSpec, operandSegmentSpec,
                                      resultSegmentSpec, resultTypeList,
                                      operandList, attributes, successors,
                                      regions, pyLoc, maybeIp);
      },
      "cls"_a, "results"_a = nb::none(), "operands"_a = nb::none(),
      "attributes"_a = nb::none(), "successors"_a = nb::none(),
      "regions"_a = nb::none(), "loc"_a = nb::none(), "ip"_a = nb::none(),
      // clang-format off
      nb::sig("def build_generic(cls, results: Sequence[Type] | None = None, operands: Sequence[Value] | None = None, attributes: dict[str, Attribute] | None = None, successors: Sequence[Block] | None = None, regions: int | None = None, loc: Location | None = None, ip: InsertionPoint | None = None) -> typing.Self"),
      // clang-format on
      "Builds a specific, generated OpView based on class level attributes.");
  opViewClass.attr("parse") = classmethod(
      [](const nb::object &cls, const std::string &sourceStr,
         const std::string &sourceName,
         DefaultingPyAiirContext context) -> nb::typed<nb::object, PyOpView> {
        PyOperationRef parsed =
            PyOperation::parse(context->getRef(), sourceStr, sourceName);

        // Check if the expected operation was parsed, and cast to to the
        // appropriate `OpView` subclass if successful.
        // NOTE: This accesses attributes that have been automatically added to
        // `OpView` subclasses, and is not intended to be used on `OpView`
        // directly.
        std::string clsOpName =
            nb::cast<std::string>(cls.attr("OPERATION_NAME"));
        AiirStringRef identifier =
            aiirIdentifierStr(aiirOperationGetName(*parsed.get()));
        std::string_view parsedOpName(identifier.data, identifier.length);
        if (clsOpName != parsedOpName)
          throw AIIRError(join("Expected a '", clsOpName, "' op, got: '",
                               parsedOpName, "'"));
        return PyOpView::constructDerived(cls, parsed.getObject());
      },
      "cls"_a, "source"_a, nb::kw_only(), "source_name"_a = "",
      "context"_a = nb::none(),
      // clang-format off
      nb::sig("def parse(cls, source: str, *, source_name: str = '', context: Context | None = None) -> typing.Self"),
      // clang-format on
      "Parses a specific, generated OpView based on class level attributes.");
  opViewClass.attr("has_trait") = classmethod(
      [](nb::object &self, nb::type_object &traitCls,
         DefaultingPyAiirContext &context) {
        PyTypeID traitTypeID =
            nb::cast<PyTypeID>(traitCls.attr(PyDynamicOpTrait::typeIDAttr));
        std::string opName = nb::cast<std::string>(self.attr("OPERATION_NAME"));
        return aiirOperationNameHasTrait(
            aiirStringRefCreate(opName.data(), opName.size()),
            traitTypeID.get(), context->get());
      },
      "cls"_a, "trait_cls"_a, "context"_a = nb::none(),
      "Checks if the operation has a given trait.");

  PyOpAdaptor::bind(m);

  //----------------------------------------------------------------------------
  // Mapping of PyRegion.
  //----------------------------------------------------------------------------
  nb::class_<PyRegion>(m, "Region")
      .def_prop_ro(
          "blocks",
          [](PyRegion &self) {
            return PyBlockList(self.getParentOperation(), self.get());
          },
          "Returns a forward-optimized sequence of blocks.")
      .def_prop_ro(
          "owner",
          [](PyRegion &self) -> nb::typed<nb::object, PyOpView> {
            return self.getParentOperation()->createOpView();
          },
          "Returns the operation owning this region.")
      .def(
          "__iter__",
          [](PyRegion &self) {
            self.checkValid();
            AiirBlock firstBlock = aiirRegionGetFirstBlock(self.get());
            return PyBlockIterator(self.getParentOperation(), firstBlock);
          },
          "Iterates over blocks in the region.")
      .def(
          "__eq__",
          [](PyRegion &self, PyRegion &other) {
            return self.get().ptr == other.get().ptr;
          },
          "Compares two regions for pointer equality.")
      .def(
          "__eq__", [](PyRegion &self, nb::object &other) { return false; },
          "Compares region with non-region object (always returns False).");

  //----------------------------------------------------------------------------
  // Mapping of PyBlock.
  //----------------------------------------------------------------------------
  nb::class_<PyBlock>(m, "Block")
      .def_prop_ro(AIIR_PYTHON_CAPI_PTR_ATTR, &PyBlock::getCapsule,
                   "Gets a capsule wrapping the AiirBlock.")
      .def_prop_ro(
          "owner",
          [](PyBlock &self) -> nb::typed<nb::object, PyOpView> {
            return self.getParentOperation()->createOpView();
          },
          "Returns the owning operation of this block.")
      .def_prop_ro(
          "region",
          [](PyBlock &self) {
            AiirRegion region = aiirBlockGetParentRegion(self.get());
            return PyRegion(self.getParentOperation(), region);
          },
          "Returns the owning region of this block.")
      .def_prop_ro(
          "arguments",
          [](PyBlock &self) {
            return PyBlockArgumentList(self.getParentOperation(), self.get());
          },
          "Returns a list of block arguments.")
      .def(
          "add_argument",
          [](PyBlock &self, const PyType &type, const PyLocation &loc) {
            return PyBlockArgument(self.getParentOperation(),
                                   aiirBlockAddArgument(self.get(), type, loc));
          },
          "type"_a, "loc"_a,
          R"(
            Appends an argument of the specified type to the block.

            Args:
              type: The type of the argument to add.
              loc: The source location for the argument.

            Returns:
              The newly added block argument.)")
      .def(
          "erase_argument",
          [](PyBlock &self, unsigned index) {
            return aiirBlockEraseArgument(self.get(), index);
          },
          "index"_a,
          R"(
            Erases the argument at the specified index.

            Args:
              index: The index of the argument to erase.)")
      .def_prop_ro(
          "operations",
          [](PyBlock &self) {
            return PyOperationList(self.getParentOperation(), self.get());
          },
          "Returns a forward-optimized sequence of operations.")
      .def_static(
          "create_at_start",
          [](PyRegion &parent, nb::typed<nb::sequence, PyType> pyArgTypes,
             const std::optional<nb::typed<nb::sequence, PyLocation>>
                 &pyArgLocs) {
            parent.checkValid();
            AiirBlock block = createBlock(pyArgTypes, pyArgLocs);
            aiirRegionInsertOwnedBlock(parent, 0, block);
            return PyBlock(parent.getParentOperation(), block);
          },
          "parent"_a, "arg_types"_a = nb::list(), "arg_locs"_a = std::nullopt,
          "Creates and returns a new Block at the beginning of the given "
          "region (with given argument types and locations).")
      .def(
          "append_to",
          [](PyBlock &self, PyRegion &region) {
            AiirBlock b = self.get();
            if (!aiirRegionIsNull(aiirBlockGetParentRegion(b)))
              aiirBlockDetach(b);
            aiirRegionAppendOwnedBlock(region.get(), b);
          },
          "region"_a,
          R"(
            Appends this block to a region.

            Transfers ownership if the block is currently owned by another region.

            Args:
              region: The region to append the block to.)")
      .def(
          "create_before",
          [](PyBlock &self, const nb::args &pyArgTypes,
             const std::optional<nb::typed<nb::sequence, PyLocation>>
                 &pyArgLocs) {
            self.checkValid();
            AiirBlock block =
                createBlock(nb::cast<nb::sequence>(pyArgTypes), pyArgLocs);
            AiirRegion region = aiirBlockGetParentRegion(self.get());
            aiirRegionInsertOwnedBlockBefore(region, self.get(), block);
            return PyBlock(self.getParentOperation(), block);
          },
          "arg_types"_a, nb::kw_only(), "arg_locs"_a = std::nullopt,
          "Creates and returns a new Block before this block "
          "(with given argument types and locations).")
      .def(
          "create_after",
          [](PyBlock &self, const nb::args &pyArgTypes,
             const std::optional<nb::typed<nb::sequence, PyLocation>>
                 &pyArgLocs) {
            self.checkValid();
            AiirBlock block =
                createBlock(nb::cast<nb::sequence>(pyArgTypes), pyArgLocs);
            AiirRegion region = aiirBlockGetParentRegion(self.get());
            aiirRegionInsertOwnedBlockAfter(region, self.get(), block);
            return PyBlock(self.getParentOperation(), block);
          },
          "arg_types"_a, nb::kw_only(), "arg_locs"_a = std::nullopt,
          "Creates and returns a new Block after this block "
          "(with given argument types and locations).")
      .def(
          "__iter__",
          [](PyBlock &self) {
            self.checkValid();
            AiirOperation firstOperation =
                aiirBlockGetFirstOperation(self.get());
            return PyOperationIterator(self.getParentOperation(),
                                       firstOperation);
          },
          "Iterates over operations in the block.")
      .def(
          "__eq__",
          [](PyBlock &self, PyBlock &other) {
            return self.get().ptr == other.get().ptr;
          },
          "Compares two blocks for pointer equality.")
      .def(
          "__eq__", [](PyBlock &self, nb::object &other) { return false; },
          "Compares block with non-block object (always returns False).")
      .def(
          "__hash__", [](PyBlock &self) { return hash(self.get().ptr); },
          "Returns the hash value of the block.")
      .def(
          "__str__",
          [](PyBlock &self) {
            self.checkValid();
            PyPrintAccumulator printAccum;
            aiirBlockPrint(self.get(), printAccum.getCallback(),
                           printAccum.getUserData());
            return printAccum.join();
          },
          "Returns the assembly form of the block.")
      .def(
          "append",
          [](PyBlock &self, PyOperationBase &operation) {
            if (operation.getOperation().isAttached())
              operation.getOperation().detachFromParent();

            AiirOperation aiirOperation = operation.getOperation().get();
            aiirBlockAppendOwnedOperation(self.get(), aiirOperation);
            operation.getOperation().setAttached(
                self.getParentOperation().getObject());
          },
          "operation"_a,
          R"(
            Appends an operation to this block.

            If the operation is currently in another block, it will be moved.

            Args:
              operation: The operation to append to the block.)")
      .def_prop_ro(
          "successors",
          [](PyBlock &self) {
            return PyBlockSuccessors(self, self.getParentOperation());
          },
          "Returns the list of Block successors.")
      .def_prop_ro(
          "predecessors",
          [](PyBlock &self) {
            return PyBlockPredecessors(self, self.getParentOperation());
          },
          "Returns the list of Block predecessors.");

  //----------------------------------------------------------------------------
  // Mapping of PyInsertionPoint.
  //----------------------------------------------------------------------------

  nb::class_<PyInsertionPoint>(m, "InsertionPoint")
      .def(nb::init<PyBlock &>(), "block"_a,
           "Inserts after the last operation but still inside the block.")
      .def("__enter__", &PyInsertionPoint::contextEnter,
           "Enters the insertion point as a context manager.",
           nb::sig("def __enter__(self, /) -> InsertionPoint"))
      .def("__exit__", &PyInsertionPoint::contextExit, "exc_type"_a.none(),
           "exc_value"_a.none(), "traceback"_a.none(),
           "Exits the insertion point context manager.")
      .def_prop_ro_static(
          "current",
          [](nb::object & /*class*/) {
            auto *ip = PyThreadContextEntry::getDefaultInsertionPoint();
            if (!ip)
              throw nb::value_error("No current InsertionPoint");
            return ip;
          },
          nb::sig("def current(/) -> InsertionPoint"),
          "Gets the InsertionPoint bound to the current thread or raises "
          "ValueError if none has been set.")
      .def(nb::init<PyOperationBase &>(), "beforeOperation"_a,
           "Inserts before a referenced operation.")
      .def_static("at_block_begin", &PyInsertionPoint::atBlockBegin, "block"_a,
                  R"(
                    Creates an insertion point at the beginning of a block.

                    Args:
                      block: The block at whose beginning operations should be inserted.

                    Returns:
                      An InsertionPoint at the block's beginning.)")
      .def_static("at_block_terminator", &PyInsertionPoint::atBlockTerminator,
                  "block"_a,
                  R"(
                    Creates an insertion point before a block's terminator.

                    Args:
                      block: The block whose terminator to insert before.

                    Returns:
                      An InsertionPoint before the terminator.

                    Raises:
                      ValueError: If the block has no terminator.)")
      .def_static("after", &PyInsertionPoint::after, "operation"_a,
                  R"(
                    Creates an insertion point immediately after an operation.

                    Args:
                      operation: The operation after which to insert.

                    Returns:
                      An InsertionPoint after the operation.)")
      .def("insert", &PyInsertionPoint::insert, "operation"_a,
           R"(
             Inserts an operation at this insertion point.

             Args:
               operation: The operation to insert.)")
      .def_prop_ro(
          "block", [](PyInsertionPoint &self) { return self.getBlock(); },
          "Returns the block that this `InsertionPoint` points to.")
      .def_prop_ro(
          "ref_operation",
          [](PyInsertionPoint &self)
              -> std::optional<nb::typed<nb::object, PyOperation>> {
            auto refOperation = self.getRefOperation();
            if (refOperation)
              return refOperation->getObject();
            return {};
          },
          "The reference operation before which new operations are "
          "inserted, or None if the insertion point is at the end of "
          "the block.");

  //----------------------------------------------------------------------------
  // Mapping of PyAttribute.
  //----------------------------------------------------------------------------
  nb::class_<PyAttribute>(m, "Attribute")
      // Delegate to the PyAttribute copy constructor, which will also lifetime
      // extend the backing context which owns the AiirAttribute.
      .def(nb::init<PyAttribute &>(), "cast_from_type"_a,
           "Casts the passed attribute to the generic `Attribute`.")
      .def_prop_ro(AIIR_PYTHON_CAPI_PTR_ATTR, &PyAttribute::getCapsule,
                   "Gets a capsule wrapping the AiirAttribute.")
      .def_static(
          AIIR_PYTHON_CAPI_FACTORY_ATTR, &PyAttribute::createFromCapsule,
          "Creates an Attribute from a capsule wrapping `AiirAttribute`.")
      .def_static(
          "parse",
          [](const std::string &attrSpec, DefaultingPyAiirContext context)
              -> nb::typed<nb::object, PyAttribute> {
            PyAiirContext::ErrorCapture errors(context->getRef());
            AiirAttribute attr = aiirAttributeParseGet(
                context->get(), toAiirStringRef(attrSpec));
            if (aiirAttributeIsNull(attr))
              throw AIIRError("Unable to parse attribute", errors.take());
            return PyAttribute(context.get()->getRef(), attr).maybeDownCast();
          },
          "asm"_a, "context"_a = nb::none(),
          "Parses an attribute from an assembly form. Raises an `AIIRError` on "
          "failure.")
      .def_prop_ro(
          "context",
          [](PyAttribute &self) -> nb::typed<nb::object, PyAiirContext> {
            return self.getContext().getObject();
          },
          "Context that owns the `Attribute`.")
      .def_prop_ro(
          "type",
          [](PyAttribute &self) -> nb::typed<nb::object, PyType> {
            return PyType(self.getContext(), aiirAttributeGetType(self))
                .maybeDownCast();
          },
          "Returns the type of the `Attribute`.")
      .def(
          "get_named",
          [](PyAttribute &self, std::string name) {
            return PyNamedAttribute(self, std::move(name));
          },
          nb::keep_alive<0, 1>(),
          R"(
            Binds a name to the attribute, creating a `NamedAttribute`.

            Args:
              name: The name to bind to the `Attribute`.

            Returns:
              A `NamedAttribute` with the given name and this attribute.)")
      .def(
          "__eq__",
          [](PyAttribute &self, PyAttribute &other) { return self == other; },
          "Compares two attributes for equality.")
      .def(
          "__eq__", [](PyAttribute &self, nb::object &other) { return false; },
          "Compares attribute with non-attribute object (always returns "
          "False).")
      .def(
          "__hash__", [](PyAttribute &self) { return hash(self.get().ptr); },
          "Returns the hash value of the attribute.")
      .def(
          "dump", [](PyAttribute &self) { aiirAttributeDump(self); },
          kDumpDocstring)
      .def(
          "__str__",
          [](PyAttribute &self) {
            PyPrintAccumulator printAccum;
            aiirAttributePrint(self, printAccum.getCallback(),
                               printAccum.getUserData());
            return printAccum.join();
          },
          "Returns the assembly form of the Attribute.")
      .def(
          "__repr__",
          [](PyAttribute &self) {
            // Generally, assembly formats are not printed for __repr__ because
            // this can cause exceptionally long debug output and exceptions.
            // However, attribute values are generally considered useful and
            // are printed. This may need to be re-evaluated if debug dumps end
            // up being excessive.
            PyPrintAccumulator printAccum;
            printAccum.parts.append("Attribute(");
            aiirAttributePrint(self, printAccum.getCallback(),
                               printAccum.getUserData());
            printAccum.parts.append(")");
            return printAccum.join();
          },
          "Returns a string representation of the attribute.")
      .def_prop_ro(
          "typeid",
          [](PyAttribute &self) {
            AiirTypeID aiirTypeID = aiirAttributeGetTypeID(self);
            assert(!aiirTypeIDIsNull(aiirTypeID) &&
                   "aiirTypeID was expected to be non-null.");
            return PyTypeID(aiirTypeID);
          },
          "Returns the `TypeID` of the attribute.")
      .def(
          AIIR_PYTHON_MAYBE_DOWNCAST_ATTR,
          [](PyAttribute &self) -> nb::typed<nb::object, PyAttribute> {
            return self.maybeDownCast();
          },
          "Downcasts the attribute to a more specific attribute if possible.");

  //----------------------------------------------------------------------------
  // Mapping of PyNamedAttribute
  //----------------------------------------------------------------------------
  nb::class_<PyNamedAttribute>(m, "NamedAttribute")
      .def(
          "__repr__",
          [](PyNamedAttribute &self) {
            PyPrintAccumulator printAccum;
            printAccum.parts.append("NamedAttribute(");
            printAccum.parts.append(
                nb::str(aiirIdentifierStr(self.namedAttr.name).data,
                        aiirIdentifierStr(self.namedAttr.name).length));
            printAccum.parts.append("=");
            aiirAttributePrint(self.namedAttr.attribute,
                               printAccum.getCallback(),
                               printAccum.getUserData());
            printAccum.parts.append(")");
            return printAccum.join();
          },
          "Returns a string representation of the named attribute.")
      .def_prop_ro(
          "name",
          [](PyNamedAttribute &self) {
            return aiirIdentifierStr(self.namedAttr.name);
          },
          "The name of the `NamedAttribute` binding.")
      .def_prop_ro(
          "attr",
          [](PyNamedAttribute &self) { return self.namedAttr.attribute; },
          nb::keep_alive<0, 1>(), nb::sig("def attr(self) -> Attribute"),
          "The underlying generic attribute of the `NamedAttribute` binding.");

  //----------------------------------------------------------------------------
  // Mapping of PyType.
  //----------------------------------------------------------------------------
  nb::class_<PyType>(m, "Type")
      // Delegate to the PyType copy constructor, which will also lifetime
      // extend the backing context which owns the AiirType.
      .def(nb::init<PyType &>(), "cast_from_type"_a,
           "Casts the passed type to the generic `Type`.")
      .def_prop_ro(AIIR_PYTHON_CAPI_PTR_ATTR, &PyType::getCapsule,
                   "Gets a capsule wrapping the `AiirType`.")
      .def_static(AIIR_PYTHON_CAPI_FACTORY_ATTR, &PyType::createFromCapsule,
                  "Creates a Type from a capsule wrapping `AiirType`.")
      .def_static(
          "parse",
          [](std::string typeSpec,
             DefaultingPyAiirContext context) -> nb::typed<nb::object, PyType> {
            PyAiirContext::ErrorCapture errors(context->getRef());
            AiirType type =
                aiirTypeParseGet(context->get(), toAiirStringRef(typeSpec));
            if (aiirTypeIsNull(type))
              throw AIIRError("Unable to parse type", errors.take());
            return PyType(context.get()->getRef(), type).maybeDownCast();
          },
          "asm"_a, "context"_a = nb::none(),
          R"(
            Parses the assembly form of a type.

            Returns a Type object or raises an `AIIRError` if the type cannot be parsed.

            See also: https://aiir.llvm.org/docs/LangRef/#type-system)")
      .def_prop_ro(
          "context",
          [](PyType &self) -> nb::typed<nb::object, PyAiirContext> {
            return self.getContext().getObject();
          },
          "Context that owns the `Type`.")
      .def(
          "__eq__", [](PyType &self, PyType &other) { return self == other; },
          "Compares two types for equality.")
      .def(
          "__eq__", [](PyType &self, nb::object &other) { return false; },
          "other"_a.none(),
          "Compares type with non-type object (always returns False).")
      .def(
          "__hash__", [](PyType &self) { return hash(self.get().ptr); },
          "Returns the hash value of the `Type`.")
      .def(
          "dump", [](PyType &self) { aiirTypeDump(self); }, kDumpDocstring)
      .def(
          "__str__",
          [](PyType &self) {
            PyPrintAccumulator printAccum;
            aiirTypePrint(self, printAccum.getCallback(),
                          printAccum.getUserData());
            return printAccum.join();
          },
          "Returns the assembly form of the `Type`.")
      .def(
          "__repr__",
          [](PyType &self) {
            // Generally, assembly formats are not printed for __repr__ because
            // this can cause exceptionally long debug output and exceptions.
            // However, types are an exception as they typically have compact
            // assembly forms and printing them is useful.
            PyPrintAccumulator printAccum;
            printAccum.parts.append("Type(");
            aiirTypePrint(self, printAccum.getCallback(),
                          printAccum.getUserData());
            printAccum.parts.append(")");
            return printAccum.join();
          },
          "Returns a string representation of the `Type`.")
      .def(
          AIIR_PYTHON_MAYBE_DOWNCAST_ATTR,
          [](PyType &self) -> nb::typed<nb::object, PyType> {
            return self.maybeDownCast();
          },
          "Downcasts the Type to a more specific `Type` if possible.")
      .def_prop_ro(
          "typeid",
          [](PyType &self) {
            AiirTypeID aiirTypeID = aiirTypeGetTypeID(self);
            if (!aiirTypeIDIsNull(aiirTypeID))
              return PyTypeID(aiirTypeID);
            auto origRepr = nb::cast<std::string>(nb::repr(nb::cast(self)));
            throw nb::value_error(join(origRepr, " has no typeid.").c_str());
          },
          "Returns the `TypeID` of the `Type`, or raises `ValueError` if "
          "`Type` has no "
          "`TypeID`.");

  //----------------------------------------------------------------------------
  // Mapping of PyTypeID.
  //----------------------------------------------------------------------------
  nb::class_<PyTypeID>(m, "TypeID")
      .def_prop_ro(AIIR_PYTHON_CAPI_PTR_ATTR, &PyTypeID::getCapsule,
                   "Gets a capsule wrapping the `AiirTypeID`.")
      .def_static(AIIR_PYTHON_CAPI_FACTORY_ATTR, &PyTypeID::createFromCapsule,
                  "Creates a `TypeID` from a capsule wrapping `AiirTypeID`.")
      // Note, this tests whether the underlying TypeIDs are the same,
      // not whether the wrapper AiirTypeIDs are the same, nor whether
      // the Python objects are the same (i.e., PyTypeID is a value type).
      .def(
          "__eq__",
          [](PyTypeID &self, PyTypeID &other) { return self == other; },
          "Compares two `TypeID`s for equality.")
      .def(
          "__eq__",
          [](PyTypeID &self, const nb::object &other) { return false; },
          "Compares TypeID with non-TypeID object (always returns False).")
      // Note, this gives the hash value of the underlying TypeID, not the
      // hash value of the Python object, nor the hash value of the
      // AiirTypeID wrapper.
      .def(
          "__hash__",
          [](PyTypeID &self) {
            return static_cast<size_t>(aiirTypeIDHashValue(self));
          },
          "Returns the hash value of the `TypeID`.");

  //----------------------------------------------------------------------------
  // Mapping of Value.
  //----------------------------------------------------------------------------
  m.attr("_T") = nb::type_var("_T", "bound"_a = m.attr("Type"));

  nb::class_<PyValue>(m, "Value", nb::is_generic(),
                      nb::sig("class Value(typing.Generic[_T])"))
      .def(nb::init<PyValue &>(), nb::keep_alive<0, 1>(), "value"_a,
           "Creates a Value reference from another `Value`.")
      .def_prop_ro(AIIR_PYTHON_CAPI_PTR_ATTR, &PyValue::getCapsule,
                   "Gets a capsule wrapping the `AiirValue`.")
      .def_static(AIIR_PYTHON_CAPI_FACTORY_ATTR, &PyValue::createFromCapsule,
                  "Creates a `Value` from a capsule wrapping `AiirValue`.")
      .def_prop_ro(
          "context",
          [](PyValue &self) -> nb::typed<nb::object, PyAiirContext> {
            return self.getParentOperation()->getContext().getObject();
          },
          "Context in which the value lives.")
      .def(
          "dump", [](PyValue &self) { aiirValueDump(self.get()); },
          kDumpDocstring)
      .def_prop_ro(
          "owner",
          [](PyValue &self)
              -> nb::typed<nb::object, std::variant<PyOpView, PyBlock>> {
            AiirValue v = self.get();
            if (aiirValueIsAOpResult(v)) {
              assert(aiirOperationEqual(self.getParentOperation()->get(),
                                        aiirOpResultGetOwner(self.get())) &&
                     "expected the owner of the value in Python to match "
                     "that in "
                     "the IR");
              return self.getParentOperation()->createOpView();
            }

            if (aiirValueIsABlockArgument(v)) {
              AiirBlock block = aiirBlockArgumentGetOwner(self.get());
              return nb::cast(PyBlock(self.getParentOperation(), block));
            }

            assert(false && "Value must be a block argument or an op result");
            return nb::none();
          },
          "Returns the owner of the value (`Operation` for results, `Block` "
          "for "
          "arguments).")
      .def_prop_ro(
          "uses",
          [](PyValue &self) {
            return PyOpOperandIterator(aiirValueGetFirstUse(self.get()));
          },
          "Returns an iterator over uses of this value.")
      .def(
          "__eq__",
          [](PyValue &self, PyValue &other) {
            return self.get().ptr == other.get().ptr;
          },
          "Compares two values for pointer equality.")
      .def(
          "__eq__", [](PyValue &self, nb::object other) { return false; },
          "Compares value with non-value object (always returns False).")
      .def(
          "__hash__", [](PyValue &self) { return hash(self.get().ptr); },
          "Returns the hash value of the value.")
      .def(
          "__str__",
          [](PyValue &self) {
            PyPrintAccumulator printAccum;
            printAccum.parts.append("Value(");
            aiirValuePrint(self.get(), printAccum.getCallback(),
                           printAccum.getUserData());
            printAccum.parts.append(")");
            return printAccum.join();
          },
          R"(
            Returns the string form of the value.

            If the value is a block argument, this is the assembly form of its type and the
            position in the argument list. If the value is an operation result, this is
            equivalent to printing the operation that produced it.
          )")
      .def(
          "get_name",
          [](PyValue &self, bool useLocalScope, bool useNameLocAsPrefix) {
            PyPrintAccumulator printAccum;
            AiirOpPrintingFlags flags = aiirOpPrintingFlagsCreate();
            if (useLocalScope)
              aiirOpPrintingFlagsUseLocalScope(flags);
            if (useNameLocAsPrefix)
              aiirOpPrintingFlagsPrintNameLocAsPrefix(flags);
            AiirAsmState valueState =
                aiirAsmStateCreateForValue(self.get(), flags);
            aiirValuePrintAsOperand(self.get(), valueState,
                                    printAccum.getCallback(),
                                    printAccum.getUserData());
            aiirOpPrintingFlagsDestroy(flags);
            aiirAsmStateDestroy(valueState);
            return printAccum.join();
          },
          "use_local_scope"_a = false, "use_name_loc_as_prefix"_a = false,
          R"(
            Returns the string form of value as an operand.

            Args:
              use_local_scope: Whether to use local scope for naming.
              use_name_loc_as_prefix: Whether to use the location attribute (NameLoc) as prefix.

            Returns:
              The value's name as it appears in IR (e.g., `%0`, `%arg0`).)")
      .def(
          "get_name",
          [](PyValue &self, PyAsmState &state) {
            PyPrintAccumulator printAccum;
            AiirAsmState valueState = state.get();
            aiirValuePrintAsOperand(self.get(), valueState,
                                    printAccum.getCallback(),
                                    printAccum.getUserData());
            return printAccum.join();
          },
          "state"_a,
          "Returns the string form of value as an operand (i.e., the ValueID).")
      .def_prop_ro(
          "type",
          [](PyValue &self) {
            return PyType(self.getParentOperation()->getContext(),
                          aiirValueGetType(self.get()))
                .maybeDownCast();
          },
          "Returns the type of the value.", nb::sig("def type(self) -> _T"))
      .def(
          "set_type",
          [](PyValue &self, const PyType &type) {
            aiirValueSetType(self.get(), type);
          },
          "type"_a, "Sets the type of the value.",
          nb::sig("def set_type(self, type: _T)"))
      .def(
          "replace_all_uses_with",
          [](PyValue &self, PyValue &with) {
            aiirValueReplaceAllUsesOfWith(self.get(), with.get());
          },
          "Replace all uses of value with the new value, updating anything in "
          "the IR that uses `self` to use the other value instead.")
      .def(
          "replace_all_uses_except",
          [](PyValue &self, PyValue &with, PyOperation &exception) {
            AiirOperation exceptedUser = exception.get();
            aiirValueReplaceAllUsesExcept(self, with, 1, &exceptedUser);
          },
          "with_"_a, "exceptions"_a, kValueReplaceAllUsesExceptDocstring)
      .def(
          "replace_all_uses_except",
          [](PyValue &self, PyValue &with,
             std::vector<PyOperation> &exceptions) {
            // Convert Python list to a std::vector of AiirOperations
            std::vector<AiirOperation> exceptionOps;
            for (PyOperation &exception : exceptions)
              exceptionOps.push_back(exception);
            aiirValueReplaceAllUsesExcept(
                self, with, static_cast<intptr_t>(exceptionOps.size()),
                exceptionOps.data());
          },
          "with_"_a, "exceptions"_a, kValueReplaceAllUsesExceptDocstring)
      .def(
          AIIR_PYTHON_MAYBE_DOWNCAST_ATTR,
          [](PyValue &self) { return self.maybeDownCast(); },
          "Downcasts the `Value` to a more specific kind if possible.")
      .def_prop_ro(
          "location",
          [](PyValue self) {
            return PyLocation(
                PyAiirContext::forContext(aiirValueGetContext(self)),
                aiirValueGetLocation(self));
          },
          "Returns the source location of the value.");

  PyBlockArgument::bind(m);
  PyOpResult::bind(m);
  PyOpOperand::bind(m);

  nb::class_<PyAsmState>(m, "AsmState")
      .def(nb::init<PyValue &, bool>(), "value"_a, "use_local_scope"_a = false,
           R"(
             Creates an `AsmState` for consistent SSA value naming.

             Args:
               value: The value to create state for.
               use_local_scope: Whether to use local scope for naming.)")
      .def(nb::init<PyOperationBase &, bool>(), "op"_a,
           "use_local_scope"_a = false,
           R"(
             Creates an AsmState for consistent SSA value naming.

             Args:
               op: The operation to create state for.
               use_local_scope: Whether to use local scope for naming.)");

  //----------------------------------------------------------------------------
  // Mapping of SymbolTable.
  //----------------------------------------------------------------------------
  nb::class_<PySymbolTable>(m, "SymbolTable")
      .def(nb::init<PyOperationBase &>(),
           R"(
             Creates a symbol table for an operation.

             Args:
               operation: The `Operation` that defines a symbol table (e.g., a `ModuleOp`).

             Raises:
               TypeError: If the operation is not a symbol table.)")
      .def(
          "__getitem__",
          [](PySymbolTable &self,
             const std::string &name) -> nb::typed<nb::object, PyOpView> {
            return self.dunderGetItem(name);
          },
          R"(
            Looks up a symbol by name in the symbol table.

            Args:
              name: The name of the symbol to look up.

            Returns:
              The operation defining the symbol.

            Raises:
              KeyError: If the symbol is not found.)")
      .def("insert", &PySymbolTable::insert, "operation"_a,
           R"(
             Inserts a symbol operation into the symbol table.

             Args:
               operation: An operation with a symbol name to insert.

             Returns:
               The symbol name attribute of the inserted operation.

             Raises:
               ValueError: If the operation does not have a symbol name.)")
      .def("erase", &PySymbolTable::erase, "operation"_a,
           R"(
             Erases a symbol operation from the symbol table.

             Args:
               operation: The symbol operation to erase.

             Note:
               The operation is also erased from the IR and invalidated.)")
      .def("__delitem__", &PySymbolTable::dunderDel,
           "Deletes a symbol by name from the symbol table.")
      .def(
          "__contains__",
          [](PySymbolTable &table, const std::string &name) {
            return !aiirOperationIsNull(aiirSymbolTableLookup(
                table, aiirStringRefCreate(name.data(), name.length())));
          },
          "Checks if a symbol with the given name exists in the table.")
      // Static helpers.
      .def_static("set_symbol_name", &PySymbolTable::setSymbolName, "symbol"_a,
                  "name"_a, "Sets the symbol name for a symbol operation.")
      .def_static("get_symbol_name", &PySymbolTable::getSymbolName, "symbol"_a,
                  "Gets the symbol name from a symbol operation.")
      .def_static("get_visibility", &PySymbolTable::getVisibility, "symbol"_a,
                  "Gets the visibility attribute of a symbol operation.")
      .def_static("set_visibility", &PySymbolTable::setVisibility, "symbol"_a,
                  "visibility"_a,
                  "Sets the visibility attribute of a symbol operation.")
      .def_static("replace_all_symbol_uses",
                  &PySymbolTable::replaceAllSymbolUses, "old_symbol"_a,
                  "new_symbol"_a, "from_op"_a,
                  "Replaces all uses of a symbol with a new symbol name within "
                  "the given operation.")
      .def_static("walk_symbol_tables", &PySymbolTable::walkSymbolTables,
                  "from_op"_a, "all_sym_uses_visible"_a, "callback"_a,
                  "Walks symbol tables starting from an operation with a "
                  "callback function.");

  // Container bindings.
  PyBlockArgumentList::bind(m);
  PyBlockIterator::bind(m);
  PyBlockList::bind(m);
  PyBlockSuccessors::bind(m);
  PyBlockPredecessors::bind(m);
  PyOperationIterator::bind(m);
  PyOperationList::bind(m);
  PyOpAttributeMap::bind(m);
  PyOpOperandIterator::bind(m);
  PyOpOperandList::bind(m);
  PyOpOperands::bind(m);
  PyOpResultList::bind(m);
  PyOpSuccessors::bind(m);
  PyRegionList::bind(m);

  // Debug bindings.
  PyGlobalDebugFlag::bind(m);

  // Attribute builder getter.
  PyAttrBuilderMap::bind(m);

  // Extensible Dialect
  PyDynamicOpTrait::bind(m);
  PyDynamicOpTraits::IsTerminator::bind(m);
  PyDynamicOpTraits::NoTerminator::bind(m);

  // AIIRError exception.
  AIIRError::bind(m);
}
} // namespace AIIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace aiir
