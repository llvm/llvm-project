//===- NanobindAdaptors.h - Interop with AIIR APIs via nanobind -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file contains adaptors for clients of the core AIIR Python APIs to
// interop via AIIR CAPI types, using nanobind. The facilities here do not
// depend on implementation details of the AIIR Python API and do not introduce
// C++-level dependencies with it (requiring only Python and CAPI-level
// dependencies).
//
// It is encouraged to be used both in-tree and out-of-tree. For in-tree use
// cases, it should be used for dialect implementations (versus relying on
// Pybind-based internals of the core libraries).
//===----------------------------------------------------------------------===//

#ifndef AIIR_BINDINGS_PYTHON_NANOBINDADAPTORS_H
#define AIIR_BINDINGS_PYTHON_NANOBINDADAPTORS_H

#include <cstdint>
#include <memory>
#include <optional>

#include "aiir-c/Diagnostics.h"
#include "aiir-c/IR.h"
// clang-format off
#include "aiir/Bindings/Python/Nanobind.h"
#include "aiir-c/Bindings/Python/Interop.h" // This is expected after nanobind.
// clang-format on
#include "aiir/Bindings/Python/NanobindUtils.h"

namespace aiir {
namespace python {
namespace {

nanobind::module_ &irModule() {
  static SafeInit<nanobind::module_> init([]() {
    return std::make_unique<nanobind::module_>(
        nanobind::module_::import_(MAKE_AIIR_PYTHON_QUALNAME("ir")));
  });
  return init.get();
}

} // namespace
} // namespace python
} // namespace aiir

// Raw CAPI type casters need to be declared before use, so always include them
// first.
namespace nanobind {
namespace detail {

/// Helper to convert a presumed AIIR API object to a capsule, accepting either
/// an explicit Capsule (which can happen when two C APIs are communicating
/// directly via Python) or indirectly by querying the AIIR_PYTHON_CAPI_PTR_ATTR
/// attribute (through which supported AIIR Python API objects export their
/// contained API pointer as a capsule). Throws a type error if the object is
/// neither. This is intended to be used from type casters, which are invoked
/// with a raw handle (unowned). The returned object's lifetime may not extend
/// beyond the apiObject handle without explicitly having its refcount increased
/// (i.e. on return).
static std::optional<nanobind::object>
aiirApiObjectToCapsule(nanobind::handle apiObject) {
  if (PyCapsule_CheckExact(apiObject.ptr()))
    return nanobind::borrow<nanobind::object>(apiObject);
  nanobind::object api =
      nanobind::getattr(apiObject, AIIR_PYTHON_CAPI_PTR_ATTR, nanobind::none());
  if (api.is_none())
    return {};
  return api;
}

// Note: Currently all of the following support cast from nanobind::object to
// the Aiir* C-API type, but only a few light-weight, context-bound ones
// implicitly cast the other way because the use case has not yet emerged and
// ownership is unclear.

/// Casts object <-> AiirAffineMap.
template <>
struct type_caster<AiirAffineMap> {
  NB_TYPE_CASTER(AiirAffineMap,
                 const_name(MAKE_AIIR_PYTHON_QUALNAME("ir.AffineMap")))
  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    if (auto capsule = aiirApiObjectToCapsule(src)) {
      value = aiirPythonCapsuleToAffineMap(capsule->ptr());
      return !aiirAffineMapIsNull(value);
    }
    return false;
  }
  static handle from_cpp(AiirAffineMap v, rv_policy,
                         cleanup_list *cleanup) noexcept {
    nanobind::object capsule =
        nanobind::steal<nanobind::object>(aiirPythonAffineMapToCapsule(v));
    return aiir::python::irModule()
        .attr("AffineMap")
        .attr(AIIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
        .release();
  }
};

/// Casts object <-> AiirAttribute.
template <>
struct type_caster<AiirAttribute> {
  NB_TYPE_CASTER(AiirAttribute,
                 const_name(MAKE_AIIR_PYTHON_QUALNAME("ir.Attribute")))
  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    if (auto capsule = aiirApiObjectToCapsule(src)) {
      value = aiirPythonCapsuleToAttribute(capsule->ptr());
      return !aiirAttributeIsNull(value);
    }
    return false;
  }
  static handle from_cpp(AiirAttribute v, rv_policy,
                         cleanup_list *cleanup) noexcept {
    nanobind::object capsule =
        nanobind::steal<nanobind::object>(aiirPythonAttributeToCapsule(v));
    return aiir::python::irModule()
        .attr("Attribute")
        .attr(AIIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
        .attr(AIIR_PYTHON_MAYBE_DOWNCAST_ATTR)()
        .release();
  }
};

/// Casts object -> AiirBlock.
template <>
struct type_caster<AiirBlock> {
  NB_TYPE_CASTER(AiirBlock, const_name(MAKE_AIIR_PYTHON_QUALNAME("ir.Block")))
  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    if (auto capsule = aiirApiObjectToCapsule(src)) {
      value = aiirPythonCapsuleToBlock(capsule->ptr());
      return !aiirBlockIsNull(value);
    }
    return false;
  }
};

/// Casts object -> AiirContext.
template <>
struct type_caster<AiirContext> {
  NB_TYPE_CASTER(AiirContext,
                 const_name(MAKE_AIIR_PYTHON_QUALNAME("ir.Context")))
  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    if (src.is_none()) {
      // Gets the current thread-bound context.
      src = aiir::python::irModule().attr("Context").attr("current");
    }
    // If there is no context, including thread-bound, emit a warning (since
    // this function is not allowed to throw) and fail to cast.
    if (src.is_none()) {
      PyErr_WarnEx(
          PyExc_RuntimeWarning,
          "Passing None as AIIR Context is only allowed inside "
          "the " MAKE_AIIR_PYTHON_QUALNAME("ir.Context") " context manager.",
          /*stacklevel=*/1);
      return false;
    }
    if (std::optional<nanobind::object> capsule = aiirApiObjectToCapsule(src)) {
      value = aiirPythonCapsuleToContext(capsule->ptr());
      return !aiirContextIsNull(value);
    }
    return false;
  }
};

/// Casts object <-> AiirDialectRegistry.
template <>
struct type_caster<AiirDialectRegistry> {
  NB_TYPE_CASTER(AiirDialectRegistry,
                 const_name(MAKE_AIIR_PYTHON_QUALNAME("ir.DialectRegistry")))
  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    if (auto capsule = aiirApiObjectToCapsule(src)) {
      value = aiirPythonCapsuleToDialectRegistry(capsule->ptr());
      return !aiirDialectRegistryIsNull(value);
    }
    return false;
  }
  static handle from_cpp(AiirDialectRegistry v, rv_policy,
                         cleanup_list *cleanup) noexcept {
    nanobind::object capsule = nanobind::steal<nanobind::object>(
        aiirPythonDialectRegistryToCapsule(v));
    return aiir::python::irModule()
        .attr("DialectRegistry")
        .attr(AIIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
        .release();
  }
};

/// Casts object <-> AiirLocation.
template <>
struct type_caster<AiirLocation> {
  NB_TYPE_CASTER(AiirLocation,
                 const_name(MAKE_AIIR_PYTHON_QUALNAME("ir.Location")))
  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    if (src.is_none()) {
      // Gets the current thread-bound context.
      src = aiir::python::irModule().attr("Location").attr("current");
    }
    if (auto capsule = aiirApiObjectToCapsule(src)) {
      value = aiirPythonCapsuleToLocation(capsule->ptr());
      return !aiirLocationIsNull(value);
    }
    return false;
  }
  static handle from_cpp(AiirLocation v, rv_policy,
                         cleanup_list *cleanup) noexcept {
    nanobind::object capsule =
        nanobind::steal<nanobind::object>(aiirPythonLocationToCapsule(v));
    return aiir::python::irModule()
        .attr("Location")
        .attr(AIIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
        .release();
  }
};

/// Casts object <-> AiirModule.
template <>
struct type_caster<AiirModule> {
  NB_TYPE_CASTER(AiirModule, const_name(MAKE_AIIR_PYTHON_QUALNAME("ir.Module")))
  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    if (auto capsule = aiirApiObjectToCapsule(src)) {
      value = aiirPythonCapsuleToModule(capsule->ptr());
      return !aiirModuleIsNull(value);
    }
    return false;
  }
  static handle from_cpp(AiirModule v, rv_policy,
                         cleanup_list *cleanup) noexcept {
    nanobind::object capsule =
        nanobind::steal<nanobind::object>(aiirPythonModuleToCapsule(v));
    return aiir::python::irModule()
        .attr("Module")
        .attr(AIIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
        .release();
  };
};

/// Casts object <-> AiirFrozenRewritePatternSet.
template <>
struct type_caster<AiirFrozenRewritePatternSet> {
  NB_TYPE_CASTER(
      AiirFrozenRewritePatternSet,
      const_name(MAKE_AIIR_PYTHON_QUALNAME("rewrite.FrozenRewritePatternSet")))
  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    if (auto capsule = aiirApiObjectToCapsule(src)) {
      value = aiirPythonCapsuleToFrozenRewritePatternSet(capsule->ptr());
      return value.ptr != nullptr;
    }
    return false;
  }
  static handle from_cpp(AiirFrozenRewritePatternSet v, rv_policy,
                         handle) noexcept {
    nanobind::object capsule = nanobind::steal<nanobind::object>(
        aiirPythonFrozenRewritePatternSetToCapsule(v));
    return nanobind::module_::import_(MAKE_AIIR_PYTHON_QUALNAME("rewrite"))
        .attr("FrozenRewritePatternSet")
        .attr(AIIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
        .release();
  };
};

/// Casts object <-> AiirOperation.
template <>
struct type_caster<AiirOperation> {
  NB_TYPE_CASTER(AiirOperation,
                 const_name(MAKE_AIIR_PYTHON_QUALNAME("ir.Operation")))
  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    if (auto capsule = aiirApiObjectToCapsule(src)) {
      value = aiirPythonCapsuleToOperation(capsule->ptr());
      return !aiirOperationIsNull(value);
    }
    return false;
  }
  static handle from_cpp(AiirOperation v, rv_policy,
                         cleanup_list *cleanup) noexcept {
    if (v.ptr == nullptr)
      return nanobind::none();
    nanobind::object capsule =
        nanobind::steal<nanobind::object>(aiirPythonOperationToCapsule(v));
    return aiir::python::irModule()
        .attr("Operation")
        .attr(AIIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
        .release();
  };
};

/// Casts object <-> AiirValue.
template <>
struct type_caster<AiirValue> {
  NB_TYPE_CASTER(AiirValue, const_name(MAKE_AIIR_PYTHON_QUALNAME("ir.Value")))
  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    if (auto capsule = aiirApiObjectToCapsule(src)) {
      value = aiirPythonCapsuleToValue(capsule->ptr());
      return !aiirValueIsNull(value);
    }
    return false;
  }
  static handle from_cpp(AiirValue v, rv_policy,
                         cleanup_list *cleanup) noexcept {
    if (v.ptr == nullptr)
      return nanobind::none();
    nanobind::object capsule =
        nanobind::steal<nanobind::object>(aiirPythonValueToCapsule(v));
    return aiir::python::irModule()
        .attr("Value")
        .attr(AIIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
        .attr(AIIR_PYTHON_MAYBE_DOWNCAST_ATTR)()
        .release();
  };
};

/// Casts object -> AiirPassManager.
template <>
struct type_caster<AiirPassManager> {
  NB_TYPE_CASTER(AiirPassManager, const_name(MAKE_AIIR_PYTHON_QUALNAME(
                                      "passmanager.PassManager")))
  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    if (auto capsule = aiirApiObjectToCapsule(src)) {
      value = aiirPythonCapsuleToPassManager(capsule->ptr());
      return !aiirPassManagerIsNull(value);
    }
    return false;
  }
};

/// Casts object <-> AiirTypeID.
template <>
struct type_caster<AiirTypeID> {
  NB_TYPE_CASTER(AiirTypeID, const_name(MAKE_AIIR_PYTHON_QUALNAME("ir.TypeID")))
  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    if (auto capsule = aiirApiObjectToCapsule(src)) {
      value = aiirPythonCapsuleToTypeID(capsule->ptr());
      return !aiirTypeIDIsNull(value);
    }
    return false;
  }
  static handle from_cpp(AiirTypeID v, rv_policy,
                         cleanup_list *cleanup) noexcept {
    if (v.ptr == nullptr)
      return nanobind::none();
    nanobind::object capsule =
        nanobind::steal<nanobind::object>(aiirPythonTypeIDToCapsule(v));
    return aiir::python::irModule()
        .attr("TypeID")
        .attr(AIIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
        .release();
  };
};

/// Casts object <-> AiirType.
template <>
struct type_caster<AiirType> {
  NB_TYPE_CASTER(AiirType, const_name(MAKE_AIIR_PYTHON_QUALNAME("ir.Type")))
  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    if (auto capsule = aiirApiObjectToCapsule(src)) {
      value = aiirPythonCapsuleToType(capsule->ptr());
      return !aiirTypeIsNull(value);
    }
    return false;
  }
  static handle from_cpp(AiirType t, rv_policy,
                         cleanup_list *cleanup) noexcept {
    nanobind::object capsule =
        nanobind::steal<nanobind::object>(aiirPythonTypeToCapsule(t));
    return aiir::python::irModule()
        .attr("Type")
        .attr(AIIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
        .attr(AIIR_PYTHON_MAYBE_DOWNCAST_ATTR)()
        .release();
  }
};

/// Casts AiirStringRef -> object.
template <>
struct type_caster<AiirStringRef> {
  NB_TYPE_CASTER(AiirStringRef, const_name("str"))
  static handle from_cpp(AiirStringRef s, rv_policy,
                         cleanup_list *cleanup) noexcept {
    return nanobind::str(s.data, s.length).release();
  }
};

} // namespace detail
} // namespace nanobind

namespace aiir {
namespace python {
namespace nanobind_adaptors {

/// Provides a facility like nanobind::class_ for defining a new class in a
/// scope, but this allows extension of an arbitrary Python class, defining
/// methods on it is a similar way. Classes defined in this way are very similar
/// to if defined in Python in the usual way but use nanobind machinery to
/// do it. These are not "real" nanobind classes but pure Python classes
/// with no relation to a concrete C++ class.
///
/// Derived from a discussion upstream:
///   https://github.com/pybind/pybind11/issues/1193
///   (plus a fair amount of extra curricular poking)
///   TODO: If this proves useful, see about including it in nanobind.
class pure_subclass {
public:
  pure_subclass(nanobind::handle scope, const char *derivedClassName,
                const nanobind::object &superClass) {
    nanobind::object pyType =
        nanobind::borrow<nanobind::object>((PyObject *)&PyType_Type);
    nanobind::object metaclass = pyType(superClass);
    nanobind::dict attributes;

    thisClass = metaclass(derivedClassName, nanobind::make_tuple(superClass),
                          attributes);
    scope.attr(derivedClassName) = thisClass;
    thisClass.attr("__module__") = scope.attr("__name__");
  }

  template <typename Func, typename... Extra>
  pure_subclass &def(const char *name, Func &&f, const Extra &...extra) {
    nanobind::object cf = nanobind::cpp_function(
        std::forward<Func>(f), nanobind::name(name), nanobind::is_method(),
        nanobind::scope(thisClass), extra...);
    thisClass.attr(name) = cf;
    return *this;
  }

  template <typename Func, typename... Extra>
  pure_subclass &def_property_readonly(const char *name, Func &&f,
                                       const Extra &...extra) {
    nanobind::object cf = nanobind::cpp_function(
        std::forward<Func>(f), nanobind::name(name), nanobind::is_method(),
        nanobind::scope(thisClass), extra...);
    auto builtinProperty =
        nanobind::borrow<nanobind::object>((PyObject *)&PyProperty_Type);
    thisClass.attr(name) = builtinProperty(cf);
    return *this;
  }

  template <typename Func, typename... Extra>
  pure_subclass &def_staticmethod(const char *name, Func &&f,
                                  const Extra &...extra) {
    static_assert(!std::is_member_function_pointer<Func>::value,
                  "def_staticmethod(...) called with a non-static member "
                  "function pointer");
    nanobind::object cf = nanobind::cpp_function(
        std::forward<Func>(f),
        nanobind::name(name), // nanobind::scope(thisClass),
        extra...);
    thisClass.attr(name) = cf;
    return *this;
  }

  template <typename Func, typename... Extra>
  pure_subclass &def_classmethod(const char *name, Func &&f,
                                 const Extra &...extra) {
    static_assert(!std::is_member_function_pointer<Func>::value,
                  "def_classmethod(...) called with a non-static member "
                  "function pointer");
    nanobind::object cf = nanobind::cpp_function(
        std::forward<Func>(f),
        nanobind::name(name), // nanobind::scope(thisClass),
        extra...);
    static SafeInit<nanobind::object> classmethodFn([]() {
      return std::make_unique<nanobind::object>(
          nanobind::module_::import_("builtins").attr("classmethod"));
    });
    thisClass.attr(name) = classmethodFn.get()(cf);
    return *this;
  }

  nanobind::object get_class() const { return thisClass; }

protected:
  nanobind::object superClass;
  nanobind::object thisClass;
};

/// Creates a custom subclass of aiir.ir.Attribute, implementing a casting
/// constructor and type checking methods.
class aiir_attribute_subclass : public pure_subclass {
public:
  using IsAFunctionTy = bool (*)(AiirAttribute);
  using GetTypeIDFunctionTy = AiirTypeID (*)();

  /// Subclasses by looking up the super-class dynamically.
  aiir_attribute_subclass(nanobind::handle scope, const char *attrClassName,
                          IsAFunctionTy isaFunction,
                          GetTypeIDFunctionTy getTypeIDFunction = nullptr)
      : aiir_attribute_subclass(scope, attrClassName, isaFunction,
                                irModule().attr("Attribute"),
                                getTypeIDFunction) {}

  /// Subclasses with a provided aiir.ir.Attribute super-class. This must
  /// be used if the subclass is being defined in the same extension module
  /// as the aiir.ir class (otherwise, it will trigger a recursive
  /// initialization).
  aiir_attribute_subclass(nanobind::handle scope, const char *typeClassName,
                          IsAFunctionTy isaFunction,
                          const nanobind::object &superCls,
                          GetTypeIDFunctionTy getTypeIDFunction = nullptr)
      : pure_subclass(scope, typeClassName, superCls) {
    // Casting constructor. Note that it is hard, if not impossible, to properly
    // call chain to parent `__init__` in nanobind due to its special handling
    // for init functions that don't have a fully constructed self-reference,
    // which makes it impossible to forward it to `__init__` of a superclass.
    // Instead, provide a custom `__new__` and call that of a superclass, which
    // eventually calls `__init__` of the superclass. Since attribute subclasses
    // have no additional members, we can just return the instance thus created
    // without amending it.
    std::string captureTypeName(
        typeClassName); // As string in case if typeClassName is not static.
    nanobind::object newCf = nanobind::cpp_function(
        [superCls, isaFunction, captureTypeName](
            nanobind::object cls, nanobind::object otherAttribute) {
          AiirAttribute rawAttribute;
          if (!nanobind::try_cast<AiirAttribute>(otherAttribute,
                                                 rawAttribute) ||
              !isaFunction(rawAttribute)) {
            auto origRepr =
                nanobind::cast<std::string>(nanobind::repr(otherAttribute));
            throw std::invalid_argument(nanobind::detail::join(
                "Cannot cast attribute to ", captureTypeName, " (from ",
                origRepr, ")"));
          }
          nanobind::object self = superCls.attr("__new__")(cls, otherAttribute);
          return self;
        },
        nanobind::name("__new__"), nanobind::arg("cls"),
        nanobind::arg("cast_from_attr"));
    thisClass.attr("__new__") = newCf;

    // 'isinstance' method.
    static const char kIsinstanceSig[] =
        "def isinstance(other_attribute: " MAKE_AIIR_PYTHON_QUALNAME(
            "ir") ".Attribute) -> bool";
    def_staticmethod(
        "isinstance",
        [isaFunction](AiirAttribute other) { return isaFunction(other); },
        nanobind::arg("other_attribute"), nanobind::sig(kIsinstanceSig));
    def("__repr__", [superCls, captureTypeName](nanobind::object self) {
      return nanobind::cast<std::string>(
          nanobind::repr(superCls(self))
              .attr("replace")(superCls.attr("__name__"), captureTypeName));
    });
    if (getTypeIDFunction) {
      def_staticmethod(
          "get_static_typeid",
          [getTypeIDFunction]() { return getTypeIDFunction(); },
          // clang-format off
          nanobind::sig("def get_static_typeid() -> " MAKE_AIIR_PYTHON_QUALNAME("ir.TypeID"))
          // clang-format on
      );
      nanobind::module_::import_(MAKE_AIIR_PYTHON_QUALNAME("ir"))
          .attr(AIIR_PYTHON_CAPI_TYPE_CASTER_REGISTER_ATTR)(
              getTypeIDFunction())(nanobind::cpp_function(
              [thisClass = thisClass](const nanobind::object &aiirAttribute) {
                return thisClass(aiirAttribute);
              }));
    }
  }
};

/// Creates a custom subclass of aiir.ir.Type, implementing a casting
/// constructor and type checking methods.
class aiir_type_subclass : public pure_subclass {
public:
  using IsAFunctionTy = bool (*)(AiirType);
  using GetTypeIDFunctionTy = AiirTypeID (*)();

  /// Subclasses by looking up the super-class dynamically.
  aiir_type_subclass(nanobind::handle scope, const char *typeClassName,
                     IsAFunctionTy isaFunction,
                     GetTypeIDFunctionTy getTypeIDFunction = nullptr)
      : aiir_type_subclass(scope, typeClassName, isaFunction,
                           irModule().attr("Type"), getTypeIDFunction) {}

  /// Subclasses with a provided aiir.ir.Type super-class. This must
  /// be used if the subclass is being defined in the same extension module
  /// as the aiir.ir class (otherwise, it will trigger a recursive
  /// initialization).
  aiir_type_subclass(nanobind::handle scope, const char *typeClassName,
                     IsAFunctionTy isaFunction,
                     const nanobind::object &superCls,
                     GetTypeIDFunctionTy getTypeIDFunction = nullptr)
      : pure_subclass(scope, typeClassName, superCls) {
    // Casting constructor. Note that it is hard, if not impossible, to properly
    // call chain to parent `__init__` in nanobind due to its special handling
    // for init functions that don't have a fully constructed self-reference,
    // which makes it impossible to forward it to `__init__` of a superclass.
    // Instead, provide a custom `__new__` and call that of a superclass, which
    // eventually calls `__init__` of the superclass. Since attribute subclasses
    // have no additional members, we can just return the instance thus created
    // without amending it.
    std::string captureTypeName(
        typeClassName); // As string in case if typeClassName is not static.
    nanobind::object newCf = nanobind::cpp_function(
        [superCls, isaFunction, captureTypeName](nanobind::object cls,
                                                 nanobind::object otherType) {
          AiirType rawType;
          if (!nanobind::try_cast<AiirType>(otherType, rawType) ||
              !isaFunction(rawType)) {
            auto origRepr =
                nanobind::cast<std::string>(nanobind::repr(otherType));
            throw std::invalid_argument(
                nanobind::detail::join("Cannot cast type to ", captureTypeName,
                                       " (from ", origRepr, ")"));
          }
          nanobind::object self = superCls.attr("__new__")(cls, otherType);
          return self;
        },
        nanobind::name("__new__"), nanobind::arg("cls"),
        nanobind::arg("cast_from_type"));
    thisClass.attr("__new__") = newCf;

    // 'isinstance' method.
    static const char kIsinstanceSig[] =
        // clang-format off
        "def isinstance(other_type: " MAKE_AIIR_PYTHON_QUALNAME("ir.Type") ") -> bool";
    // clang-format on
    def_staticmethod(
        "isinstance",
        [isaFunction](AiirType other) { return isaFunction(other); },
        nanobind::arg("other_type"), nanobind::sig(kIsinstanceSig));
    def("__repr__", [superCls, captureTypeName](nanobind::object self) {
      return nanobind::cast<std::string>(
          nanobind::repr(superCls(self))
              .attr("replace")(superCls.attr("__name__"), captureTypeName));
    });
    if (getTypeIDFunction) {
      // 'get_static_typeid' method.
      // This is modeled as a static method instead of a static property because
      // `def_property_readonly_static` is not available in `pure_subclass` and
      // we do not want to introduce the complexity that pybind uses to
      // implement it.
      def_staticmethod(
          "get_static_typeid",
          [getTypeIDFunction]() { return getTypeIDFunction(); },
          // clang-format off
          nanobind::sig("def get_static_typeid() -> " MAKE_AIIR_PYTHON_QUALNAME("ir.TypeID"))
          // clang-format on
      );
      nanobind::module_::import_(MAKE_AIIR_PYTHON_QUALNAME("ir"))
          .attr(AIIR_PYTHON_CAPI_TYPE_CASTER_REGISTER_ATTR)(
              getTypeIDFunction())(nanobind::cpp_function(
              [thisClass = thisClass](const nanobind::object &aiirType) {
                return thisClass(aiirType);
              }));
    }
  }
};

/// Creates a custom subclass of aiir.ir.Value, implementing a casting
/// constructor and type checking methods.
class aiir_value_subclass : public pure_subclass {
public:
  using IsAFunctionTy = bool (*)(AiirValue);

  /// Subclasses by looking up the super-class dynamically.
  aiir_value_subclass(nanobind::handle scope, const char *valueClassName,
                      IsAFunctionTy isaFunction)
      : aiir_value_subclass(scope, valueClassName, isaFunction,
                            irModule().attr("Value")) {}

  /// Subclasses with a provided aiir.ir.Value super-class. This must
  /// be used if the subclass is being defined in the same extension module
  /// as the aiir.ir class (otherwise, it will trigger a recursive
  /// initialization).
  aiir_value_subclass(nanobind::handle scope, const char *valueClassName,
                      IsAFunctionTy isaFunction,
                      const nanobind::object &superCls)
      : pure_subclass(scope, valueClassName, superCls) {
    // Casting constructor. Note that it is hard, if not impossible, to properly
    // call chain to parent `__init__` in nanobind due to its special handling
    // for init functions that don't have a fully constructed self-reference,
    // which makes it impossible to forward it to `__init__` of a superclass.
    // Instead, provide a custom `__new__` and call that of a superclass, which
    // eventually calls `__init__` of the superclass. Since attribute subclasses
    // have no additional members, we can just return the instance thus created
    // without amending it.
    std::string captureValueName(
        valueClassName); // As string in case if valueClassName is not static.
    nanobind::object newCf = nanobind::cpp_function(
        [superCls, isaFunction, captureValueName](nanobind::object cls,
                                                  nanobind::object otherValue) {
          AiirValue rawValue;
          if (!nanobind::try_cast<AiirValue>(otherValue, rawValue) ||
              !isaFunction(rawValue)) {
            auto origRepr =
                nanobind::cast<std::string>(nanobind::repr(otherValue));
            throw std::invalid_argument(nanobind::detail::join(
                "Cannot cast value to ", captureValueName, " (from ", origRepr,
                ")"));
          }
          nanobind::object self = superCls.attr("__new__")(cls, otherValue);
          return self;
        },
        nanobind::name("__new__"), nanobind::arg("cls"),
        nanobind::arg("cast_from_value"));
    thisClass.attr("__new__") = newCf;

    // 'isinstance' method.
    static const char kIsinstanceSig[] =
        // clang-format off
        "def isinstance(other_value: " MAKE_AIIR_PYTHON_QUALNAME("ir.Value") ") -> bool";
    // clang-format on
    def_staticmethod(
        "isinstance",
        [isaFunction](AiirValue other) { return isaFunction(other); },
        nanobind::arg("other_value"), nanobind::sig(kIsinstanceSig));
  }
};

} // namespace nanobind_adaptors

} // namespace python
} // namespace aiir

#endif // AIIR_BINDINGS_PYTHON_NANOBINDADAPTORS_H
