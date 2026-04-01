//===- Rewrite.cpp - Rewrite ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Rewrite.h"

#include "aiir-c/Bindings/Python/Interop.h"
#include "aiir-c/IR.h"
#include "aiir-c/Rewrite.h"
#include "aiir-c/Support.h"
#include "aiir/Bindings/Python/Globals.h"
#include "aiir/Bindings/Python/IRCore.h"
#include "aiir/Config/aiir-config.h"
#include "nanobind/nanobind.h"
#include <type_traits>

namespace nb = nanobind;
using namespace aiir;
using namespace nb::literals;
using namespace aiir::python::AIIR_BINDINGS_PYTHON_DOMAIN;

namespace aiir {
namespace python {
namespace AIIR_BINDINGS_PYTHON_DOMAIN {

// Convert the Python object to a boolean.
// If it evaluates to False, treat it as success;
// otherwise, treat it as failure.
// Note that None is considered success.
static AiirLogicalResult logicalResultFromObject(const nb::object &obj) {
  if (obj.is_none())
    return aiirLogicalResultSuccess();

  return nb::cast<bool>(obj) ? aiirLogicalResultFailure()
                             : aiirLogicalResultSuccess();
}

static std::string operationNameFromObject(nb::handle root) {
  if (root.is_type())
    return nb::cast<std::string>(root.attr("OPERATION_NAME"));
  if (nb::isinstance<nb::str>(root))
    return nb::cast<std::string>(root);

  throw nb::type_error("the root argument must be a type or a string");
}

static std::string dialectNameFromObject(nb::handle root) {
  if (root.is_type())
    return nb::cast<std::string>(root.attr("DIALECT_NAMESPACE"));
  if (nb::isinstance<nb::str>(root))
    return nb::cast<std::string>(root);

  throw nb::type_error("the root argument must be a type or a string");
}

class PyPatternRewriter : public PyRewriterBase<PyPatternRewriter> {
public:
  static constexpr const char *pyClassName = "PatternRewriter";

  PyPatternRewriter(AiirPatternRewriter rewriter)
      : PyRewriterBase(aiirPatternRewriterAsBase(rewriter)) {}
};

//===----------------------------------------------------------------------===//
// PyRewritePatternSet
//===----------------------------------------------------------------------===//

PyRewritePatternSet::PyRewritePatternSet(AiirContext ctx)
    : patterns(aiirRewritePatternSetCreate(ctx)), owned(true) {}

PyRewritePatternSet::PyRewritePatternSet(AiirRewritePatternSet patterns)
    : patterns(patterns), owned(false) {}

PyRewritePatternSet::~PyRewritePatternSet() {
  if (owned && patterns.ptr)
    aiirRewritePatternSetDestroy(patterns);
}

AiirRewritePatternSet PyRewritePatternSet::get() const { return patterns; }

bool PyRewritePatternSet::isOwned() const { return owned; }

void PyRewritePatternSet::add(nb::handle root,
                              const nb::callable &matchAndRewrite,
                              unsigned benefit) {
  std::string opName = operationNameFromObject(root);
  AiirStringRef rootName = aiirStringRefCreate(opName.data(), opName.size());

  AiirRewritePatternCallbacks callbacks;
  callbacks.construct = [](void *userData) {
    nb::handle(static_cast<PyObject *>(userData)).inc_ref();
  };
  callbacks.destruct = [](void *userData) {
    nb::handle(static_cast<PyObject *>(userData)).dec_ref();
  };
  callbacks.matchAndRewrite = [](AiirRewritePattern, AiirOperation op,
                                 AiirPatternRewriter rewriter,
                                 void *userData) -> AiirLogicalResult {
    nb::handle f(static_cast<PyObject *>(userData));

    PyAiirContextRef context =
        PyAiirContext::forContext(aiirOperationGetContext(op));
    nb::object opView = PyOperation::forOperation(context, op)->createOpView();

    nb::object res = f(opView, PyPatternRewriter(rewriter));
    return logicalResultFromObject(res);
  };

  AiirRewritePattern pattern = aiirOpRewritePatternCreate(
      rootName, benefit, aiirRewritePatternSetGetContext(patterns), callbacks,
      matchAndRewrite.ptr(),
      /* nGeneratedNames */ 0,
      /* generatedNames */ nullptr);
  aiirRewritePatternSetAdd(patterns, pattern);
}

//===----------------------------------------------------------------------===//
// PyConversionPatternRewriter
//===----------------------------------------------------------------------===//

class PyConversionPatternRewriter : public PyPatternRewriter {
public:
  PyConversionPatternRewriter(AiirConversionPatternRewriter rewriter)
      : PyPatternRewriter(
            aiirConversionPatternRewriterAsPatternRewriter(rewriter)),
        rewriter(rewriter) {}

  AiirConversionPatternRewriter rewriter;
};

class PyConversionTarget {
public:
  PyConversionTarget(AiirContext context)
      : target(aiirConversionTargetCreate(context)) {}
  ~PyConversionTarget() { aiirConversionTargetDestroy(target); }

  void addLegalOp(const std::string &opName) {
    aiirConversionTargetAddLegalOp(
        target, aiirStringRefCreate(opName.data(), opName.size()));
  }

  void addIllegalOp(const std::string &opName) {
    aiirConversionTargetAddIllegalOp(
        target, aiirStringRefCreate(opName.data(), opName.size()));
  }

  void addLegalDialect(const std::string &dialectName) {
    aiirConversionTargetAddLegalDialect(
        target, aiirStringRefCreate(dialectName.data(), dialectName.size()));
  }

  void addIllegalDialect(const std::string &dialectName) {
    aiirConversionTargetAddIllegalDialect(
        target, aiirStringRefCreate(dialectName.data(), dialectName.size()));
  }

  AiirConversionTarget get() { return target; }

private:
  AiirConversionTarget target;
};

class PyTypeConverter {
public:
  PyTypeConverter() : typeConverter(aiirTypeConverterCreate()), owner(true) {}
  PyTypeConverter(AiirTypeConverter typeConverter)
      : typeConverter(typeConverter), owner(false) {}
  ~PyTypeConverter() {
    if (owner)
      aiirTypeConverterDestroy(typeConverter);
  }

  void addConversion(const nb::callable &convert) {
    aiirTypeConverterAddConversion(
        typeConverter,
        [](AiirType type, AiirType *converted,
           void *userData) -> AiirLogicalResult {
          nb::handle f = nb::handle(static_cast<PyObject *>(userData));
          auto ctx = PyAiirContext::forContext(aiirTypeGetContext(type));
          nb::object res = f(PyType(ctx, type).maybeDownCast());
          if (res.is_none())
            return aiirLogicalResultFailure();

          *converted = nb::cast<PyType>(res).get();
          return aiirLogicalResultSuccess();
        },
        convert.ptr());
  }

  nb::typed<nb::object, std::optional<PyType>> convertType(PyType &type) {
    AiirType converted = aiirTypeConverterConvertType(typeConverter, type);
    if (aiirTypeIsNull(converted))
      return nb::none();
    return PyType(PyAiirContext::forContext(aiirTypeGetContext(converted)),
                  converted)
        .maybeDownCast();
  }

  AiirTypeConverter get() { return typeConverter; }

private:
  AiirTypeConverter typeConverter;
  bool owner;
};

class PyConversionPattern {
public:
  PyConversionPattern(AiirConversionPattern pattern) : pattern(pattern) {}

  PyTypeConverter getTypeConverter() {
    return PyTypeConverter(aiirConversionPatternGetTypeConverter(pattern));
  }

private:
  AiirConversionPattern pattern;
};

void PyRewritePatternSet::addConversion(nb::handle root,
                                        const nb::callable &matchAndRewrite,
                                        PyTypeConverter &typeConverter,
                                        unsigned benefit) {
  std::string opName = operationNameFromObject(root);
  AiirStringRef rootName = aiirStringRefCreate(opName.data(), opName.size());

  AiirConversionPatternCallbacks callbacks;
  callbacks.construct = [](void *userData) {
    nb::handle(static_cast<PyObject *>(userData)).inc_ref();
  };
  callbacks.destruct = [](void *userData) {
    nb::handle(static_cast<PyObject *>(userData)).dec_ref();
  };
  callbacks.matchAndRewrite =
      [](AiirConversionPattern pattern, AiirOperation op, intptr_t nOperands,
         AiirValue *operands, AiirConversionPatternRewriter rewriter,
         void *userData) -> AiirLogicalResult {
    nb::handle f(static_cast<PyObject *>(userData));

    PyAiirContextRef ctx =
        PyAiirContext::forContext(aiirOperationGetContext(op));
    nb::object opView = PyOperation::forOperation(ctx, op)->createOpView();

    std::vector<AiirValue> operandsVec(operands, operands + nOperands);
    nb::object adaptorCls =
        PyGlobals::get()
            .lookupOpAdaptorClass([&] {
              AiirStringRef ref = aiirIdentifierStr(aiirOperationGetName(op));
              return std::string_view(ref.data, ref.length);
            }())
            .value_or(nb::borrow(nb::type<PyOpAdaptor>()));

    nb::object res = f(opView, adaptorCls(operandsVec, opView),
                       PyConversionPattern(pattern).getTypeConverter(),
                       PyConversionPatternRewriter(rewriter));
    return logicalResultFromObject(res);
  };
  AiirConversionPattern pattern = aiirOpConversionPatternCreate(
      rootName, benefit, aiirRewritePatternSetGetContext(patterns),
      typeConverter.get(), callbacks, matchAndRewrite.ptr(),
      /* nGeneratedNames */ 0,
      /* generatedNames */ nullptr);
  aiirRewritePatternSetAdd(patterns,
                           aiirConversionPatternAsRewritePattern(pattern));
}

#if AIIR_ENABLE_PDL_IN_PATTERNMATCH
struct PyAiirPDLResultList : AiirPDLResultList {};

static nb::object objectFromPDLValue(AiirPDLValue value) {
  if (AiirValue v = aiirPDLValueAsValue(value); !aiirValueIsNull(v))
    return nb::cast(v);
  if (AiirOperation v = aiirPDLValueAsOperation(value); !aiirOperationIsNull(v))
    return nb::cast(v);
  if (AiirAttribute v = aiirPDLValueAsAttribute(value); !aiirAttributeIsNull(v))
    return nb::cast(v);
  if (AiirType v = aiirPDLValueAsType(value); !aiirTypeIsNull(v))
    return nb::cast(v);

  throw std::runtime_error("unsupported PDL value type");
}

static std::vector<nb::object> objectsFromPDLValues(size_t nValues,
                                                    AiirPDLValue *values) {
  std::vector<nb::object> args;
  args.reserve(nValues);
  for (size_t i = 0; i < nValues; ++i)
    args.push_back(objectFromPDLValue(values[i]));
  return args;
}

/// Owning Wrapper around a PDLPatternModule.
class PyPDLPatternModule {
public:
  PyPDLPatternModule(AiirPDLPatternModule module) : module(module) {}
  PyPDLPatternModule(PyPDLPatternModule &&other) noexcept
      : module(other.module) {
    other.module.ptr = nullptr;
  }
  ~PyPDLPatternModule() {
    if (module.ptr != nullptr)
      aiirPDLPatternModuleDestroy(module);
  }
  AiirPDLPatternModule get() { return module; }

  void registerRewriteFunction(const std::string &name,
                               const nb::callable &fn) {
    aiirPDLPatternModuleRegisterRewriteFunction(
        get(), aiirStringRefCreate(name.data(), name.size()),
        [](AiirPatternRewriter rewriter, AiirPDLResultList results,
           size_t nValues, AiirPDLValue *values,
           void *userData) -> AiirLogicalResult {
          nb::handle f = nb::handle(static_cast<PyObject *>(userData));
          return logicalResultFromObject(
              f(PyPatternRewriter(rewriter), PyAiirPDLResultList{results.ptr},
                objectsFromPDLValues(nValues, values)));
        },
        fn.ptr());
  }

  void registerConstraintFunction(const std::string &name,
                                  const nb::callable &fn) {
    aiirPDLPatternModuleRegisterConstraintFunction(
        get(), aiirStringRefCreate(name.data(), name.size()),
        [](AiirPatternRewriter rewriter, AiirPDLResultList results,
           size_t nValues, AiirPDLValue *values,
           void *userData) -> AiirLogicalResult {
          nb::handle f = nb::handle(static_cast<PyObject *>(userData));
          return logicalResultFromObject(
              f(PyPatternRewriter(rewriter), PyAiirPDLResultList{results.ptr},
                objectsFromPDLValues(nValues, values)));
        },
        fn.ptr());
  }

private:
  AiirPDLPatternModule module;
};
#endif // AIIR_ENABLE_PDL_IN_PATTERNMATCH

/// Owning Wrapper around a FrozenRewritePatternSet.
class PyFrozenRewritePatternSet {
public:
  PyFrozenRewritePatternSet(AiirFrozenRewritePatternSet set) : set(set) {}
  PyFrozenRewritePatternSet(PyFrozenRewritePatternSet &&other) noexcept
      : set(other.set) {
    other.set.ptr = nullptr;
  }
  ~PyFrozenRewritePatternSet() {
    if (set.ptr != nullptr)
      aiirFrozenRewritePatternSetDestroy(set);
  }
  AiirFrozenRewritePatternSet get() { return set; }

  nb::object getCapsule() {
    return nb::steal<nb::object>(
        aiirPythonFrozenRewritePatternSetToCapsule(get()));
  }

  static nb::object createFromCapsule(const nb::object &capsule) {
    AiirFrozenRewritePatternSet rawPm =
        aiirPythonCapsuleToFrozenRewritePatternSet(capsule.ptr());
    if (rawPm.ptr == nullptr)
      throw nb::python_error();
    return nb::cast(PyFrozenRewritePatternSet(rawPm), nb::rv_policy::move);
  }

private:
  AiirFrozenRewritePatternSet set;
};

void PyRewritePatternSet::bind(nb::module_ &m) {
  nb::class_<PyRewritePatternSet>(m, "RewritePatternSet")
      .def(
          "__init__",
          [](PyRewritePatternSet &self, DefaultingPyAiirContext context) {
            new (&self) PyRewritePatternSet(context.get()->get());
          },
          "context"_a = nb::none())
      .def("add", &PyRewritePatternSet::add, nb::arg("root"), nb::arg("fn"),
           nb::arg("benefit") = 1,
           R"(Add a new rewrite pattern on the specified root operation, using
              the provided callable for matching and rewriting, and assign it
              the given benefit.

              Args:
                root: The root operation to which this pattern applies. This may
                      be either an OpView subclass or an operation name.
                fn: The callable to use for matching and rewriting, which takes
                    an operation and a pattern rewriter. The match is considered
                    successful iff the callable returns a falsy value.
                benefit: The benefit of the pattern, defaulting to 1.)")
      .def("add_conversion", &PyRewritePatternSet::addConversion,
           nb::arg("root"), nb::arg("fn"), nb::arg("type_converter"),
           nb::arg("benefit") = 1,
           R"(
            Add a new conversion pattern on the specified root operation,
            using the provided callable for matching and rewriting,
            and assign it the given benefit.

            Args:
              root: The root operation to which this pattern applies.
                    This may be either an OpView subclass or an operation name.
              fn: The callable to use for matching and rewriting, which takes an
                  operation, its adaptor, the type converter and a pattern
                  rewriter. The match is considered successful iff the callable
                  returns a falsy value.
              type_converter: The type converter to convert types in the IR.
              benefit: The benefit of the pattern, defaulting to 1.)")
      .def(
          "freeze",
          [](PyRewritePatternSet &self) {
            if (!self.isOwned())
              throw std::runtime_error(
                  "cannot freeze a non-owning pattern set");
            AiirRewritePatternSet s = self.get();
            return PyFrozenRewritePatternSet(aiirFreezeRewritePattern(s));
          },
          "Freeze the pattern set into a frozen one.");
}

enum class PyGreedyRewriteStrictness : std::underlying_type_t<
    AiirGreedyRewriteStrictness> {
  ANY_OP = AIIR_GREEDY_REWRITE_STRICTNESS_ANY_OP,
  EXISTING_AND_NEW_OPS = AIIR_GREEDY_REWRITE_STRICTNESS_EXISTING_AND_NEW_OPS,
  EXISTING_OPS = AIIR_GREEDY_REWRITE_STRICTNESS_EXISTING_OPS,
};

enum class PyGreedySimplifyRegionLevel : std::underlying_type_t<
    AiirGreedySimplifyRegionLevel> {
  DISABLED = AIIR_GREEDY_SIMPLIFY_REGION_LEVEL_DISABLED,
  NORMAL = AIIR_GREEDY_SIMPLIFY_REGION_LEVEL_NORMAL,
  AGGRESSIVE = AIIR_GREEDY_SIMPLIFY_REGION_LEVEL_AGGRESSIVE
};

/// Owning Wrapper around a GreedyRewriteDriverConfig.
class PyGreedyRewriteConfig {
public:
  PyGreedyRewriteConfig()
      : config(aiirGreedyRewriteDriverConfigCreate().ptr,
               PyGreedyRewriteConfig::customDeleter) {}
  PyGreedyRewriteConfig(PyGreedyRewriteConfig &&other) noexcept
      : config(std::move(other.config)) {}
  PyGreedyRewriteConfig(const PyGreedyRewriteConfig &other) noexcept
      : config(other.config) {}

  AiirGreedyRewriteDriverConfig get() {
    return AiirGreedyRewriteDriverConfig{config.get()};
  }

  void setMaxIterations(int64_t maxIterations) {
    aiirGreedyRewriteDriverConfigSetMaxIterations(get(), maxIterations);
  }

  void setMaxNumRewrites(int64_t maxNumRewrites) {
    aiirGreedyRewriteDriverConfigSetMaxNumRewrites(get(), maxNumRewrites);
  }

  void setUseTopDownTraversal(bool useTopDownTraversal) {
    aiirGreedyRewriteDriverConfigSetUseTopDownTraversal(get(),
                                                        useTopDownTraversal);
  }

  void enableFolding(bool enable) {
    aiirGreedyRewriteDriverConfigEnableFolding(get(), enable);
  }

  void setStrictness(PyGreedyRewriteStrictness strictness) {
    aiirGreedyRewriteDriverConfigSetStrictness(
        get(), static_cast<AiirGreedyRewriteStrictness>(strictness));
  }

  void setRegionSimplificationLevel(PyGreedySimplifyRegionLevel level) {
    aiirGreedyRewriteDriverConfigSetRegionSimplificationLevel(
        get(), static_cast<AiirGreedySimplifyRegionLevel>(level));
  }

  void enableConstantCSE(bool enable) {
    aiirGreedyRewriteDriverConfigEnableConstantCSE(get(), enable);
  }

  int64_t getMaxIterations() {
    return aiirGreedyRewriteDriverConfigGetMaxIterations(get());
  }

  int64_t getMaxNumRewrites() {
    return aiirGreedyRewriteDriverConfigGetMaxNumRewrites(get());
  }

  bool getUseTopDownTraversal() {
    return aiirGreedyRewriteDriverConfigGetUseTopDownTraversal(get());
  }

  bool isFoldingEnabled() {
    return aiirGreedyRewriteDriverConfigIsFoldingEnabled(get());
  }

  PyGreedyRewriteStrictness getStrictness() {
    return static_cast<PyGreedyRewriteStrictness>(
        aiirGreedyRewriteDriverConfigGetStrictness(get()));
  }

  PyGreedySimplifyRegionLevel getRegionSimplificationLevel() {
    return static_cast<PyGreedySimplifyRegionLevel>(
        aiirGreedyRewriteDriverConfigGetRegionSimplificationLevel(get()));
  }

  bool isConstantCSEEnabled() {
    return aiirGreedyRewriteDriverConfigIsConstantCSEEnabled(get());
  }

private:
  std::shared_ptr<void> config;
  static void customDeleter(void *c) {
    aiirGreedyRewriteDriverConfigDestroy(AiirGreedyRewriteDriverConfig{c});
  }
};

enum class PyDialectConversionFoldingMode : std::underlying_type_t<
    AiirDialectConversionFoldingMode> {
  Never = AIIR_DIALECT_CONVERSION_FOLDING_MODE_NEVER,
  BeforePatterns = AIIR_DIALECT_CONVERSION_FOLDING_MODE_BEFORE_PATTERNS,
  AfterPatterns = AIIR_DIALECT_CONVERSION_FOLDING_MODE_AFTER_PATTERNS,
};

class PyConversionConfig {
public:
  PyConversionConfig()
      : config(aiirConversionConfigCreate().ptr,
               PyConversionConfig::customDeleter) {}

  AiirConversionConfig get() { return AiirConversionConfig{config.get()}; }

  void setFoldingMode(PyDialectConversionFoldingMode mode) {
    aiirConversionConfigSetFoldingMode(get(),
                                       AiirDialectConversionFoldingMode(mode));
  }

  PyDialectConversionFoldingMode getFoldingMode() {
    return PyDialectConversionFoldingMode(
        aiirConversionConfigGetFoldingMode(get()));
  }

  void enableBuildMaterializations(bool enabled) {
    aiirConversionConfigEnableBuildMaterializations(get(), enabled);
  }

  bool isBuildMaterializationsEnabled() {
    return aiirConversionConfigIsBuildMaterializationsEnabled(get());
  }

private:
  std::shared_ptr<void> config;
  static void customDeleter(void *c) {
    aiirConversionConfigDestroy(AiirConversionConfig{c});
  }
};

/// Create the `aiir.rewrite` here.
void populateRewriteSubmodule(nb::module_ &m) {
  // Enum definitions
  nb::enum_<PyGreedyRewriteStrictness>(m, "GreedyRewriteStrictness")
      .value("ANY_OP", PyGreedyRewriteStrictness::ANY_OP)
      .value("EXISTING_AND_NEW_OPS",
             PyGreedyRewriteStrictness::EXISTING_AND_NEW_OPS)
      .value("EXISTING_OPS", PyGreedyRewriteStrictness::EXISTING_OPS);

  nb::enum_<PyGreedySimplifyRegionLevel>(m, "GreedySimplifyRegionLevel")
      .value("DISABLED", PyGreedySimplifyRegionLevel::DISABLED)
      .value("NORMAL", PyGreedySimplifyRegionLevel::NORMAL)
      .value("AGGRESSIVE", PyGreedySimplifyRegionLevel::AGGRESSIVE);

  nb::enum_<PyDialectConversionFoldingMode>(m, "DialectConversionFoldingMode")
      .value("NEVER", PyDialectConversionFoldingMode::Never)
      .value("BEFORE_PATTERNS", PyDialectConversionFoldingMode::BeforePatterns)
      .value("AFTER_PATTERNS", PyDialectConversionFoldingMode::AfterPatterns);

  //----------------------------------------------------------------------------
  // Mapping of the PatternRewriter
  //----------------------------------------------------------------------------

  PyPatternRewriter::bind(m);

  //----------------------------------------------------------------------------
  // Mapping of the RewritePatternSet
  //----------------------------------------------------------------------------
  PyRewritePatternSet::bind(m);

  nb::class_<PyConversionPatternRewriter, PyPatternRewriter>(
      m, "ConversionPatternRewriter")
      .def("convert_region_types",
           [](PyConversionPatternRewriter &self, PyRegion &region,
              PyTypeConverter &typeConverter) {
             aiirConversionPatternRewriterConvertRegionTypes(
                 self.rewriter, region.get(), typeConverter.get());
           });

  nb::class_<PyConversionTarget>(m, "ConversionTarget")
      .def(
          "__init__",
          [](PyConversionTarget &self, DefaultingPyAiirContext context) {
            new (&self) PyConversionTarget(context.get()->get());
          },
          "context"_a = nb::none())
      .def(
          "add_legal_op",
          [](PyConversionTarget &self, const nb::args &ops) {
            for (auto op : ops) {
              self.addLegalOp(operationNameFromObject(op));
            }
          },
          "ops"_a, "Mark the given operations as legal.")
      .def(
          "add_illegal_op",
          [](PyConversionTarget &self, const nb::args &ops) {
            for (auto op : ops) {
              self.addIllegalOp(operationNameFromObject(op));
            }
          },
          "ops"_a, "Mark the given operations as illegal.")
      .def(
          "add_legal_dialect",
          [](PyConversionTarget &self, const nb::args &dialects) {
            for (auto dialect : dialects) {
              self.addLegalDialect(dialectNameFromObject(dialect));
            }
          },
          "dialects"_a, "Mark the given dialects as legal.")
      .def(
          "add_illegal_dialect",
          [](PyConversionTarget &self, const nb::args &dialects) {
            for (auto dialect : dialects) {
              self.addIllegalDialect(dialectNameFromObject(dialect));
            }
          },
          "dialects"_a, "Mark the given dialect as illegal.");

  nb::class_<PyTypeConverter>(m, "TypeConverter")
      .def(nb::init<>(), "Create a new TypeConverter.")
      .def("add_conversion", &PyTypeConverter::addConversion, "convert"_a,
           nb::keep_alive<0, 1>(), "Register a type conversion function.")
      .def("convert_type", &PyTypeConverter::convertType, "type"_a,
           "Convert the given type. Returns None if conversion fails.");

  //----------------------------------------------------------------------------
  // Mapping of the PDLResultList and PDLModule
  //----------------------------------------------------------------------------
#if AIIR_ENABLE_PDL_IN_PATTERNMATCH
  nb::class_<PyAiirPDLResultList>(m, "PDLResultList")
      .def("append",
           [](PyAiirPDLResultList results, const PyValue &value) {
             aiirPDLResultListPushBackValue(results, value);
           })
      .def("append",
           [](PyAiirPDLResultList results, const PyOperation &op) {
             aiirPDLResultListPushBackOperation(results, op);
           })
      .def("append",
           [](PyAiirPDLResultList results, const PyType &type) {
             aiirPDLResultListPushBackType(results, type);
           })
      .def("append", [](PyAiirPDLResultList results, const PyAttribute &attr) {
        aiirPDLResultListPushBackAttribute(results, attr);
      });
  nb::class_<PyPDLPatternModule>(m, "PDLModule")
      .def(
          "__init__",
          [](PyPDLPatternModule &self, PyModule &module) {
            new (&self) PyPDLPatternModule(
                aiirPDLPatternModuleFromModule(module.get()));
          },
          "module"_a, "Create a PDL module from the given module.")
      .def(
          "__init__",
          [](PyPDLPatternModule &self, PyModule &module) {
            new (&self) PyPDLPatternModule(
                aiirPDLPatternModuleFromModule(module.get()));
          },
          "module"_a, "Create a PDL module from the given module.")
      .def(
          "freeze",
          [](PyPDLPatternModule &self) {
            return PyFrozenRewritePatternSet(aiirFreezeRewritePattern(
                aiirRewritePatternSetFromPDLPatternModule(self.get())));
          },
          nb::keep_alive<0, 1>())
      .def(
          "register_rewrite_function",
          [](PyPDLPatternModule &self, const std::string &name,
             const nb::callable &fn) {
            self.registerRewriteFunction(name, fn);
          },
          nb::keep_alive<1, 3>())
      .def(
          "register_constraint_function",
          [](PyPDLPatternModule &self, const std::string &name,
             const nb::callable &fn) {
            self.registerConstraintFunction(name, fn);
          },
          nb::keep_alive<1, 3>());
#endif // AIIR_ENABLE_PDL_IN_PATTERNMATCH

  nb::class_<PyGreedyRewriteConfig>(m, "GreedyRewriteConfig")
      .def(nb::init<>(), "Create a greedy rewrite driver config with defaults")
      .def_prop_rw("max_iterations", &PyGreedyRewriteConfig::getMaxIterations,
                   &PyGreedyRewriteConfig::setMaxIterations,
                   "Maximum number of iterations")
      .def_prop_rw("max_num_rewrites",
                   &PyGreedyRewriteConfig::getMaxNumRewrites,
                   &PyGreedyRewriteConfig::setMaxNumRewrites,
                   "Maximum number of rewrites per iteration")
      .def_prop_rw("use_top_down_traversal",
                   &PyGreedyRewriteConfig::getUseTopDownTraversal,
                   &PyGreedyRewriteConfig::setUseTopDownTraversal,
                   "Whether to use top-down traversal")
      .def_prop_rw("enable_folding", &PyGreedyRewriteConfig::isFoldingEnabled,
                   &PyGreedyRewriteConfig::enableFolding,
                   "Enable or disable folding")
      .def_prop_rw("strictness", &PyGreedyRewriteConfig::getStrictness,
                   &PyGreedyRewriteConfig::setStrictness,
                   "Rewrite strictness level")
      .def_prop_rw("region_simplification_level",
                   &PyGreedyRewriteConfig::getRegionSimplificationLevel,
                   &PyGreedyRewriteConfig::setRegionSimplificationLevel,
                   "Region simplification level")
      .def_prop_rw("enable_constant_cse",
                   &PyGreedyRewriteConfig::isConstantCSEEnabled,
                   &PyGreedyRewriteConfig::enableConstantCSE,
                   "Enable or disable constant CSE");

  nb::class_<PyConversionConfig>(m, "ConversionConfig")
      .def(nb::init<>(), "Create a conversion config with defaults")
      .def_prop_rw("folding_mode", &PyConversionConfig::getFoldingMode,
                   &PyConversionConfig::setFoldingMode,
                   "folding behavior during dialect conversion")
      .def_prop_rw("build_materializations",
                   &PyConversionConfig::isBuildMaterializationsEnabled,
                   &PyConversionConfig::enableBuildMaterializations,
                   "Whether the dialect conversion attempts to build "
                   "source/target materializations");

  nb::class_<PyFrozenRewritePatternSet>(m, "FrozenRewritePatternSet")
      .def_prop_ro(AIIR_PYTHON_CAPI_PTR_ATTR,
                   &PyFrozenRewritePatternSet::getCapsule)
      .def(AIIR_PYTHON_CAPI_FACTORY_ATTR,
           &PyFrozenRewritePatternSet::createFromCapsule);
  m.def(
       "apply_patterns_and_fold_greedily",
       [](PyModule &module, PyFrozenRewritePatternSet &set,
          std::optional<PyGreedyRewriteConfig> config) {
         AiirLogicalResult status = aiirApplyPatternsAndFoldGreedily(
             module.get(), set.get(),
             config.has_value() ? config->get()
                                : aiirGreedyRewriteDriverConfigCreate());
         if (aiirLogicalResultIsFailure(status))
           throw std::runtime_error("pattern application failed to converge");
       },
       "module"_a, "set"_a, "config"_a = nb::none(),
       "Applys the given patterns to the given module greedily while folding "
       "results.")
      .def(
          "apply_patterns_and_fold_greedily",
          [](PyOperationBase &op, PyFrozenRewritePatternSet &set,
             std::optional<PyGreedyRewriteConfig> config) {
            AiirLogicalResult status = aiirApplyPatternsAndFoldGreedilyWithOp(
                op.getOperation(), set.get(),
                config.has_value() ? config->get()
                                   : aiirGreedyRewriteDriverConfigCreate());
            if (aiirLogicalResultIsFailure(status))
              throw std::runtime_error(
                  "pattern application failed to converge");
          },
          "op"_a, "set"_a, "config"_a = nb::none(),
          "Applys the given patterns to the given op greedily while folding "
          "results.")
      .def(
          "walk_and_apply_patterns",
          [](PyOperationBase &op, PyFrozenRewritePatternSet &set) {
            aiirWalkAndApplyPatterns(op.getOperation(), set.get());
          },
          "op"_a, "set"_a,
          "Applies the given patterns to the given op by a fast walk-based "
          "driver.")
      .def(
          "apply_partial_conversion",
          [](PyOperationBase &op, PyConversionTarget &target,
             PyFrozenRewritePatternSet &set,
             std::optional<PyConversionConfig> config) {
            if (!config)
              config.emplace(PyConversionConfig());
            PyAiirContext::ErrorCapture errors(op.getOperation().getContext());
            AiirLogicalResult status = aiirApplyPartialConversion(
                op.getOperation(), target.get(), set.get(), config->get());
            if (aiirLogicalResultIsFailure(status))
              throw AIIRError("partial conversion failed", errors.take());
          },
          "op"_a, "target"_a, "set"_a, "config"_a = nb::none(),
          "Applies a partial conversion on the given operation.")
      .def(
          "apply_full_conversion",
          [](PyOperationBase &op, PyConversionTarget &target,
             PyFrozenRewritePatternSet &set,
             std::optional<PyConversionConfig> config) {
            if (!config)
              config.emplace(PyConversionConfig());
            PyAiirContext::ErrorCapture errors(op.getOperation().getContext());
            AiirLogicalResult status = aiirApplyFullConversion(
                op.getOperation(), target.get(), set.get(), config->get());
            if (aiirLogicalResultIsFailure(status))
              throw AIIRError("full conversion failed", errors.take());
          },
          "op"_a, "target"_a, "set"_a, "config"_a = nb::none(),
          "Applies a full conversion on the given operation.");
}
} // namespace AIIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace aiir
