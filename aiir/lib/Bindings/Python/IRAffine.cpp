//===- IRAffine.cpp - Exports 'ir' module affine related bindings ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "aiir-c/AffineExpr.h"
#include "aiir-c/AffineMap.h"
#include "aiir/Bindings/Python/IRCore.h"
// clang-format off
#include "aiir/Bindings/Python/NanobindUtils.h"
#include "aiir-c/Bindings/Python/Interop.h" // This is expected after nanobind.
// clang-format on
#include "aiir-c/IntegerSet.h"
#include "aiir/Bindings/Python/Nanobind.h"

namespace nb = nanobind;
using namespace aiir;
using namespace aiir::python::AIIR_BINDINGS_PYTHON_DOMAIN;

static const char kDumpDocstring[] =
    R"(Dumps a debug representation of the object to stderr.)";

/// Attempts to populate `result` with the content of `list` casted to the
/// appropriate type (Python and C types are provided as template arguments).
/// Throws errors in case of failure, using "action" to describe what the caller
/// was attempting to do.
template <typename PyType, typename CType>
static void pyListToVector(const nb::sequence &list, std::vector<CType> &result,
                           std::string_view action) {
  result.reserve(nb::len(list));
  for (nb::handle item : list) {
    try {
      result.push_back(nb::cast<PyType>(item));
    } catch (nb::cast_error &err) {
      std::string msg = nanobind::detail::join("Invalid expression when ",
                                               action, " (", err.what(), ")");
      throw std::runtime_error(msg.c_str());
    } catch (std::runtime_error &err) {
      std::string msg = nanobind::detail::join(
          "Invalid expression (None?) when ", action, " (", err.what(), ")");
      throw std::runtime_error(msg.c_str());
    }
  }
}

template <typename PermutationTy>
static bool isPermutation(const std::vector<PermutationTy> &permutation) {
  std::vector<bool> seen(permutation.size(), false);
  for (auto val : permutation) {
    if (val < permutation.size()) {
      if (seen[val])
        return false;
      seen[val] = true;
      continue;
    }
    return false;
  }
  return true;
}

namespace aiir {
namespace python {
namespace AIIR_BINDINGS_PYTHON_DOMAIN {

/// CRTP base class for Python AIIR affine expressions that subclass AffineExpr
/// and should be castable from it. Intermediate hierarchy classes can be
/// modeled by specifying BaseTy.
template <typename DerivedTy, typename BaseTy = PyAffineExpr>
class PyConcreteAffineExpr : public BaseTy {
public:
  // Derived classes must define statics for:
  //   IsAFunctionTy isaFunction
  //   const char *pyClassName
  // and redefine bindDerived.
  using ClassTy = nb::class_<DerivedTy, BaseTy>;
  using IsAFunctionTy = bool (*)(AiirAffineExpr);

  PyConcreteAffineExpr() = default;
  PyConcreteAffineExpr(PyAiirContextRef contextRef, AiirAffineExpr affineExpr)
      : BaseTy(std::move(contextRef), affineExpr) {}
  PyConcreteAffineExpr(PyAffineExpr &orig)
      : PyConcreteAffineExpr(orig.getContext(), castFrom(orig)) {}

  static AiirAffineExpr castFrom(PyAffineExpr &orig) {
    if (!DerivedTy::isaFunction(orig)) {
      auto origRepr = nb::cast<std::string>(nb::repr(nb::cast(orig)));
      throw nb::value_error(
          nanobind::detail::join("Cannot cast affine expression to ",
                                 DerivedTy::pyClassName, " (from ", origRepr,
                                 ")")
              .c_str());
    }
    return orig;
  }

  static void bind(nb::module_ &m) {
    auto cls = ClassTy(m, DerivedTy::pyClassName);
    cls.def(nb::init<PyAffineExpr &>(), nb::arg("expr"));
    DerivedTy::bindDerived(cls);
  }

  /// Implemented by derived classes to add methods to the Python subclass.
  static void bindDerived(ClassTy &m) {}
};

class PyAffineConstantExpr : public PyConcreteAffineExpr<PyAffineConstantExpr> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirAffineExprIsAConstant;
  static constexpr const char *pyClassName = "AffineConstantExpr";
  using PyConcreteAffineExpr::PyConcreteAffineExpr;

  static PyAffineConstantExpr get(intptr_t value,
                                  DefaultingPyAiirContext context) {
    AiirAffineExpr affineExpr =
        aiirAffineConstantExprGet(context->get(), static_cast<int64_t>(value));
    return PyAffineConstantExpr(context->getRef(), affineExpr);
  }

  static void bindDerived(ClassTy &c) {
    c.def_static("get", &PyAffineConstantExpr::get, nb::arg("value"),
                 nb::arg("context") = nb::none());
    c.def_prop_ro("value", [](PyAffineConstantExpr &self) {
      return aiirAffineConstantExprGetValue(self);
    });
  }
};

class PyAffineDimExpr : public PyConcreteAffineExpr<PyAffineDimExpr> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirAffineExprIsADim;
  static constexpr const char *pyClassName = "AffineDimExpr";
  using PyConcreteAffineExpr::PyConcreteAffineExpr;

  static PyAffineDimExpr get(intptr_t pos, DefaultingPyAiirContext context) {
    AiirAffineExpr affineExpr = aiirAffineDimExprGet(context->get(), pos);
    return PyAffineDimExpr(context->getRef(), affineExpr);
  }

  static void bindDerived(ClassTy &c) {
    c.def_static("get", &PyAffineDimExpr::get, nb::arg("position"),
                 nb::arg("context") = nb::none());
    c.def_prop_ro("position", [](PyAffineDimExpr &self) {
      return aiirAffineDimExprGetPosition(self);
    });
  }
};

class PyAffineSymbolExpr : public PyConcreteAffineExpr<PyAffineSymbolExpr> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirAffineExprIsASymbol;
  static constexpr const char *pyClassName = "AffineSymbolExpr";
  using PyConcreteAffineExpr::PyConcreteAffineExpr;

  static PyAffineSymbolExpr get(intptr_t pos, DefaultingPyAiirContext context) {
    AiirAffineExpr affineExpr = aiirAffineSymbolExprGet(context->get(), pos);
    return PyAffineSymbolExpr(context->getRef(), affineExpr);
  }

  static void bindDerived(ClassTy &c) {
    c.def_static("get", &PyAffineSymbolExpr::get, nb::arg("position"),
                 nb::arg("context") = nb::none());
    c.def_prop_ro("position", [](PyAffineSymbolExpr &self) {
      return aiirAffineSymbolExprGetPosition(self);
    });
  }
};

class PyAffineBinaryExpr : public PyConcreteAffineExpr<PyAffineBinaryExpr> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirAffineExprIsABinary;
  static constexpr const char *pyClassName = "AffineBinaryExpr";
  using PyConcreteAffineExpr::PyConcreteAffineExpr;

  nb::typed<nb::object, PyAffineExpr> lhs() {
    AiirAffineExpr lhsExpr = aiirAffineBinaryOpExprGetLHS(get());
    return PyAffineExpr(getContext(), lhsExpr).maybeDownCast();
  }

  nb::typed<nb::object, PyAffineExpr> rhs() {
    AiirAffineExpr rhsExpr = aiirAffineBinaryOpExprGetRHS(get());
    return PyAffineExpr(getContext(), rhsExpr).maybeDownCast();
  }

  static void bindDerived(ClassTy &c) {
    c.def_prop_ro("lhs", &PyAffineBinaryExpr::lhs);
    c.def_prop_ro("rhs", &PyAffineBinaryExpr::rhs);
  }
};

class PyAffineAddExpr
    : public PyConcreteAffineExpr<PyAffineAddExpr, PyAffineBinaryExpr> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirAffineExprIsAAdd;
  static constexpr const char *pyClassName = "AffineAddExpr";
  using PyConcreteAffineExpr::PyConcreteAffineExpr;

  static PyAffineAddExpr get(PyAffineExpr lhs, const PyAffineExpr &rhs) {
    AiirAffineExpr expr = aiirAffineAddExprGet(lhs, rhs);
    return PyAffineAddExpr(lhs.getContext(), expr);
  }

  static PyAffineAddExpr getRHSConstant(PyAffineExpr lhs, intptr_t rhs) {
    AiirAffineExpr expr = aiirAffineAddExprGet(
        lhs, aiirAffineConstantExprGet(aiirAffineExprGetContext(lhs), rhs));
    return PyAffineAddExpr(lhs.getContext(), expr);
  }

  static PyAffineAddExpr getLHSConstant(intptr_t lhs, PyAffineExpr rhs) {
    AiirAffineExpr expr = aiirAffineAddExprGet(
        aiirAffineConstantExprGet(aiirAffineExprGetContext(rhs), lhs), rhs);
    return PyAffineAddExpr(rhs.getContext(), expr);
  }

  static void bindDerived(ClassTy &c) {
    c.def_static("get", &PyAffineAddExpr::get);
  }
};

class PyAffineMulExpr
    : public PyConcreteAffineExpr<PyAffineMulExpr, PyAffineBinaryExpr> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirAffineExprIsAMul;
  static constexpr const char *pyClassName = "AffineMulExpr";
  using PyConcreteAffineExpr::PyConcreteAffineExpr;

  static PyAffineMulExpr get(PyAffineExpr lhs, const PyAffineExpr &rhs) {
    AiirAffineExpr expr = aiirAffineMulExprGet(lhs, rhs);
    return PyAffineMulExpr(lhs.getContext(), expr);
  }

  static PyAffineMulExpr getRHSConstant(PyAffineExpr lhs, intptr_t rhs) {
    AiirAffineExpr expr = aiirAffineMulExprGet(
        lhs, aiirAffineConstantExprGet(aiirAffineExprGetContext(lhs), rhs));
    return PyAffineMulExpr(lhs.getContext(), expr);
  }

  static PyAffineMulExpr getLHSConstant(intptr_t lhs, PyAffineExpr rhs) {
    AiirAffineExpr expr = aiirAffineMulExprGet(
        aiirAffineConstantExprGet(aiirAffineExprGetContext(rhs), lhs), rhs);
    return PyAffineMulExpr(rhs.getContext(), expr);
  }

  static void bindDerived(ClassTy &c) {
    c.def_static("get", &PyAffineMulExpr::get);
  }
};

class PyAffineModExpr
    : public PyConcreteAffineExpr<PyAffineModExpr, PyAffineBinaryExpr> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirAffineExprIsAMod;
  static constexpr const char *pyClassName = "AffineModExpr";
  using PyConcreteAffineExpr::PyConcreteAffineExpr;

  static PyAffineModExpr get(PyAffineExpr lhs, const PyAffineExpr &rhs) {
    AiirAffineExpr expr = aiirAffineModExprGet(lhs, rhs);
    return PyAffineModExpr(lhs.getContext(), expr);
  }

  static PyAffineModExpr getRHSConstant(PyAffineExpr lhs, intptr_t rhs) {
    AiirAffineExpr expr = aiirAffineModExprGet(
        lhs, aiirAffineConstantExprGet(aiirAffineExprGetContext(lhs), rhs));
    return PyAffineModExpr(lhs.getContext(), expr);
  }

  static PyAffineModExpr getLHSConstant(intptr_t lhs, PyAffineExpr rhs) {
    AiirAffineExpr expr = aiirAffineModExprGet(
        aiirAffineConstantExprGet(aiirAffineExprGetContext(rhs), lhs), rhs);
    return PyAffineModExpr(rhs.getContext(), expr);
  }

  static void bindDerived(ClassTy &c) {
    c.def_static("get", &PyAffineModExpr::get);
  }
};

class PyAffineFloorDivExpr
    : public PyConcreteAffineExpr<PyAffineFloorDivExpr, PyAffineBinaryExpr> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirAffineExprIsAFloorDiv;
  static constexpr const char *pyClassName = "AffineFloorDivExpr";
  using PyConcreteAffineExpr::PyConcreteAffineExpr;

  static PyAffineFloorDivExpr get(PyAffineExpr lhs, const PyAffineExpr &rhs) {
    AiirAffineExpr expr = aiirAffineFloorDivExprGet(lhs, rhs);
    return PyAffineFloorDivExpr(lhs.getContext(), expr);
  }

  static PyAffineFloorDivExpr getRHSConstant(PyAffineExpr lhs, intptr_t rhs) {
    AiirAffineExpr expr = aiirAffineFloorDivExprGet(
        lhs, aiirAffineConstantExprGet(aiirAffineExprGetContext(lhs), rhs));
    return PyAffineFloorDivExpr(lhs.getContext(), expr);
  }

  static PyAffineFloorDivExpr getLHSConstant(intptr_t lhs, PyAffineExpr rhs) {
    AiirAffineExpr expr = aiirAffineFloorDivExprGet(
        aiirAffineConstantExprGet(aiirAffineExprGetContext(rhs), lhs), rhs);
    return PyAffineFloorDivExpr(rhs.getContext(), expr);
  }

  static void bindDerived(ClassTy &c) {
    c.def_static("get", &PyAffineFloorDivExpr::get);
  }
};

class PyAffineCeilDivExpr
    : public PyConcreteAffineExpr<PyAffineCeilDivExpr, PyAffineBinaryExpr> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirAffineExprIsACeilDiv;
  static constexpr const char *pyClassName = "AffineCeilDivExpr";
  using PyConcreteAffineExpr::PyConcreteAffineExpr;

  static PyAffineCeilDivExpr get(PyAffineExpr lhs, const PyAffineExpr &rhs) {
    AiirAffineExpr expr = aiirAffineCeilDivExprGet(lhs, rhs);
    return PyAffineCeilDivExpr(lhs.getContext(), expr);
  }

  static PyAffineCeilDivExpr getRHSConstant(PyAffineExpr lhs, intptr_t rhs) {
    AiirAffineExpr expr = aiirAffineCeilDivExprGet(
        lhs, aiirAffineConstantExprGet(aiirAffineExprGetContext(lhs), rhs));
    return PyAffineCeilDivExpr(lhs.getContext(), expr);
  }

  static PyAffineCeilDivExpr getLHSConstant(intptr_t lhs, PyAffineExpr rhs) {
    AiirAffineExpr expr = aiirAffineCeilDivExprGet(
        aiirAffineConstantExprGet(aiirAffineExprGetContext(rhs), lhs), rhs);
    return PyAffineCeilDivExpr(rhs.getContext(), expr);
  }

  static void bindDerived(ClassTy &c) {
    c.def_static("get", &PyAffineCeilDivExpr::get);
  }
};

} // namespace AIIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace aiir

bool PyAffineExpr::operator==(const PyAffineExpr &other) const {
  return aiirAffineExprEqual(affineExpr, other.affineExpr);
}

nb::object PyAffineExpr::getCapsule() {
  return nb::steal<nb::object>(aiirPythonAffineExprToCapsule(*this));
}

PyAffineExpr PyAffineExpr::createFromCapsule(const nb::object &capsule) {
  AiirAffineExpr rawAffineExpr = aiirPythonCapsuleToAffineExpr(capsule.ptr());
  if (aiirAffineExprIsNull(rawAffineExpr))
    throw nb::python_error();
  return PyAffineExpr(
      PyAiirContext::forContext(aiirAffineExprGetContext(rawAffineExpr)),
      rawAffineExpr);
}

nb::typed<nb::object, PyAffineExpr> PyAffineExpr::maybeDownCast() {
  AiirAffineExpr expr = get();
  if (aiirAffineExprIsAConstant(expr))
    return nb::cast(PyAffineConstantExpr(getContext(), expr));
  if (aiirAffineExprIsADim(expr))
    return nb::cast(PyAffineDimExpr(getContext(), expr));
  if (aiirAffineExprIsASymbol(expr))
    return nb::cast(PyAffineSymbolExpr(getContext(), expr));
  if (aiirAffineExprIsAAdd(expr))
    return nb::cast(PyAffineAddExpr(getContext(), expr));
  if (aiirAffineExprIsAMul(expr))
    return nb::cast(PyAffineMulExpr(getContext(), expr));
  if (aiirAffineExprIsAMod(expr))
    return nb::cast(PyAffineModExpr(getContext(), expr));
  if (aiirAffineExprIsAFloorDiv(expr))
    return nb::cast(PyAffineFloorDivExpr(getContext(), expr));
  if (aiirAffineExprIsACeilDiv(expr))
    return nb::cast(PyAffineCeilDivExpr(getContext(), expr));
  return nb::cast(*this);
}

//------------------------------------------------------------------------------
// PyAffineMap and utilities.
//------------------------------------------------------------------------------
namespace aiir {
namespace python {
namespace AIIR_BINDINGS_PYTHON_DOMAIN {

/// A list of expressions contained in an affine map. Internally these are
/// stored as a consecutive array leading to inexpensive random access. Both
/// the map and the expression are owned by the context so we need not bother
/// with lifetime extension.
class PyAffineMapExprList
    : public Sliceable<PyAffineMapExprList, PyAffineExpr> {
public:
  static constexpr const char *pyClassName = "AffineExprList";

  PyAffineMapExprList(const PyAffineMap &map, intptr_t startIndex = 0,
                      intptr_t length = -1, intptr_t step = 1)
      : Sliceable(startIndex,
                  length == -1 ? aiirAffineMapGetNumResults(map) : length,
                  step),
        affineMap(map) {}

private:
  /// Give the parent CRTP class access to hook implementations below.
  friend class Sliceable<PyAffineMapExprList, PyAffineExpr>;

  intptr_t getRawNumElements() { return aiirAffineMapGetNumResults(affineMap); }

  PyAffineExpr getRawElement(intptr_t pos) {
    return PyAffineExpr(affineMap.getContext(),
                        aiirAffineMapGetResult(affineMap, pos));
  }

  PyAffineMapExprList slice(intptr_t startIndex, intptr_t length,
                            intptr_t step) {
    return PyAffineMapExprList(affineMap, startIndex, length, step);
  }

  PyAffineMap affineMap;
};
} // namespace AIIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace aiir

bool PyAffineMap::operator==(const PyAffineMap &other) const {
  return aiirAffineMapEqual(affineMap, other.affineMap);
}

nb::object PyAffineMap::getCapsule() {
  return nb::steal<nb::object>(aiirPythonAffineMapToCapsule(*this));
}

PyAffineMap PyAffineMap::createFromCapsule(const nb::object &capsule) {
  AiirAffineMap rawAffineMap = aiirPythonCapsuleToAffineMap(capsule.ptr());
  if (aiirAffineMapIsNull(rawAffineMap))
    throw nb::python_error();
  return PyAffineMap(
      PyAiirContext::forContext(aiirAffineMapGetContext(rawAffineMap)),
      rawAffineMap);
}

//------------------------------------------------------------------------------
// PyIntegerSet and utilities.
//------------------------------------------------------------------------------
namespace aiir {
namespace python {
namespace AIIR_BINDINGS_PYTHON_DOMAIN {

class PyIntegerSetConstraint {
public:
  PyIntegerSetConstraint(PyIntegerSet set, intptr_t pos)
      : set(std::move(set)), pos(pos) {}

  PyAffineExpr getExpr() {
    return PyAffineExpr(set.getContext(),
                        aiirIntegerSetGetConstraint(set, pos));
  }

  bool isEq() { return aiirIntegerSetIsConstraintEq(set, pos); }

  static void bind(nb::module_ &m) {
    nb::class_<PyIntegerSetConstraint>(m, "IntegerSetConstraint")
        .def_prop_ro("expr", &PyIntegerSetConstraint::getExpr)
        .def_prop_ro("is_eq", &PyIntegerSetConstraint::isEq);
  }

private:
  PyIntegerSet set;
  intptr_t pos;
};

class PyIntegerSetConstraintList
    : public Sliceable<PyIntegerSetConstraintList, PyIntegerSetConstraint> {
public:
  static constexpr const char *pyClassName = "IntegerSetConstraintList";

  PyIntegerSetConstraintList(const PyIntegerSet &set, intptr_t startIndex = 0,
                             intptr_t length = -1, intptr_t step = 1)
      : Sliceable(startIndex,
                  length == -1 ? aiirIntegerSetGetNumConstraints(set) : length,
                  step),
        set(set) {}

private:
  /// Give the parent CRTP class access to hook implementations below.
  friend class Sliceable<PyIntegerSetConstraintList, PyIntegerSetConstraint>;

  intptr_t getRawNumElements() { return aiirIntegerSetGetNumConstraints(set); }

  PyIntegerSetConstraint getRawElement(intptr_t pos) {
    return PyIntegerSetConstraint(set, pos);
  }

  PyIntegerSetConstraintList slice(intptr_t startIndex, intptr_t length,
                                   intptr_t step) {
    return PyIntegerSetConstraintList(set, startIndex, length, step);
  }

  PyIntegerSet set;
};
} // namespace AIIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace aiir

bool PyIntegerSet::operator==(const PyIntegerSet &other) const {
  return aiirIntegerSetEqual(integerSet, other.integerSet);
}

nb::object PyIntegerSet::getCapsule() {
  return nb::steal<nb::object>(aiirPythonIntegerSetToCapsule(*this));
}

PyIntegerSet PyIntegerSet::createFromCapsule(const nb::object &capsule) {
  AiirIntegerSet rawIntegerSet = aiirPythonCapsuleToIntegerSet(capsule.ptr());
  if (aiirIntegerSetIsNull(rawIntegerSet))
    throw nb::python_error();
  return PyIntegerSet(
      PyAiirContext::forContext(aiirIntegerSetGetContext(rawIntegerSet)),
      rawIntegerSet);
}

namespace aiir {
namespace python {
namespace AIIR_BINDINGS_PYTHON_DOMAIN {
void populateIRAffine(nb::module_ &m) {
  //----------------------------------------------------------------------------
  // Mapping of PyAffineExpr and derived classes.
  //----------------------------------------------------------------------------
  nb::class_<PyAffineExpr>(m, "AffineExpr")
      .def_prop_ro(AIIR_PYTHON_CAPI_PTR_ATTR, &PyAffineExpr::getCapsule)
      .def(AIIR_PYTHON_CAPI_FACTORY_ATTR, &PyAffineExpr::createFromCapsule)
      .def("__add__", &PyAffineAddExpr::get)
      .def("__add__", &PyAffineAddExpr::getRHSConstant)
      .def("__radd__", &PyAffineAddExpr::getRHSConstant)
      .def("__mul__", &PyAffineMulExpr::get)
      .def("__mul__", &PyAffineMulExpr::getRHSConstant)
      .def("__rmul__", &PyAffineMulExpr::getRHSConstant)
      .def("__mod__", &PyAffineModExpr::get)
      .def("__mod__", &PyAffineModExpr::getRHSConstant)
      .def("__rmod__",
           [](PyAffineExpr &self, intptr_t other) {
             return PyAffineModExpr::get(
                 PyAffineConstantExpr::get(other, *self.getContext().get()),
                 self);
           })
      .def("__sub__",
           [](PyAffineExpr &self, PyAffineExpr &other) {
             auto negOne =
                 PyAffineConstantExpr::get(-1, *self.getContext().get());
             return PyAffineAddExpr::get(self,
                                         PyAffineMulExpr::get(negOne, other));
           })
      .def("__sub__",
           [](PyAffineExpr &self, intptr_t other) {
             return PyAffineAddExpr::get(
                 self,
                 PyAffineConstantExpr::get(-other, *self.getContext().get()));
           })
      .def("__rsub__",
           [](PyAffineExpr &self, intptr_t other) {
             return PyAffineAddExpr::getLHSConstant(
                 other, PyAffineMulExpr::getLHSConstant(-1, self));
           })
      .def("__eq__", [](PyAffineExpr &self,
                        PyAffineExpr &other) { return self == other; })
      .def("__eq__",
           [](PyAffineExpr &self, nb::object &other) { return false; })
      .def("__str__",
           [](PyAffineExpr &self) {
             PyPrintAccumulator printAccum;
             aiirAffineExprPrint(self, printAccum.getCallback(),
                                 printAccum.getUserData());
             return printAccum.join();
           })
      .def("__repr__",
           [](PyAffineExpr &self) {
             PyPrintAccumulator printAccum;
             printAccum.parts.append("AffineExpr(");
             aiirAffineExprPrint(self, printAccum.getCallback(),
                                 printAccum.getUserData());
             printAccum.parts.append(")");
             return printAccum.join();
           })
      .def("__hash__",
           [](PyAffineExpr &self) {
             return std::hash<const void *>{}(self.get().ptr);
           })
      .def_prop_ro(
          "context",
          [](PyAffineExpr &self) -> nb::typed<nb::object, PyAiirContext> {
            return self.getContext().getObject();
          })
      .def("compose",
           [](PyAffineExpr &self, PyAffineMap &other) {
             return PyAffineExpr(self.getContext(),
                                 aiirAffineExprCompose(self, other));
           })
      .def(AIIR_PYTHON_MAYBE_DOWNCAST_ATTR, &PyAffineExpr::maybeDownCast)
      .def(
          "shift_dims",
          [](PyAffineExpr &self, uint32_t numDims, uint32_t shift,
             uint32_t offset) {
            return PyAffineExpr(
                self.getContext(),
                aiirAffineExprShiftDims(self, numDims, shift, offset));
          },
          nb::arg("num_dims"), nb::arg("shift"), nb::arg("offset") = 0)
      .def(
          "shift_symbols",
          [](PyAffineExpr &self, uint32_t numSymbols, uint32_t shift,
             uint32_t offset) {
            return PyAffineExpr(
                self.getContext(),
                aiirAffineExprShiftSymbols(self, numSymbols, shift, offset));
          },
          nb::arg("num_symbols"), nb::arg("shift"), nb::arg("offset") = 0)
      .def_static(
          "simplify_affine_expr",
          [](PyAffineExpr &self, uint32_t numDims, uint32_t numSymbols) {
            return PyAffineExpr(
                self.getContext(),
                aiirSimplifyAffineExpr(self, numDims, numSymbols));
          },
          nb::arg("expr"), nb::arg("num_dims"), nb::arg("num_symbols"),
          "Simplify an affine expression by flattening and some amount of "
          "simple analysis.")
      .def_static(
          "get_add", &PyAffineAddExpr::get,
          "Gets an affine expression containing a sum of two expressions.")
      .def_static("get_add", &PyAffineAddExpr::getLHSConstant,
                  "Gets an affine expression containing a sum of a constant "
                  "and another expression.")
      .def_static("get_add", &PyAffineAddExpr::getRHSConstant,
                  "Gets an affine expression containing a sum of an expression "
                  "and a constant.")
      .def_static(
          "get_mul", &PyAffineMulExpr::get,
          "Gets an affine expression containing a product of two expressions.")
      .def_static("get_mul", &PyAffineMulExpr::getLHSConstant,
                  "Gets an affine expression containing a product of a "
                  "constant and another expression.")
      .def_static("get_mul", &PyAffineMulExpr::getRHSConstant,
                  "Gets an affine expression containing a product of an "
                  "expression and a constant.")
      .def_static("get_mod", &PyAffineModExpr::get,
                  "Gets an affine expression containing the modulo of dividing "
                  "one expression by another.")
      .def_static("get_mod", &PyAffineModExpr::getLHSConstant,
                  "Gets a semi-affine expression containing the modulo of "
                  "dividing a constant by an expression.")
      .def_static("get_mod", &PyAffineModExpr::getRHSConstant,
                  "Gets an affine expression containing the module of dividing"
                  "an expression by a constant.")
      .def_static("get_floor_div", &PyAffineFloorDivExpr::get,
                  "Gets an affine expression containing the rounded-down "
                  "result of dividing one expression by another.")
      .def_static("get_floor_div", &PyAffineFloorDivExpr::getLHSConstant,
                  "Gets a semi-affine expression containing the rounded-down "
                  "result of dividing a constant by an expression.")
      .def_static("get_floor_div", &PyAffineFloorDivExpr::getRHSConstant,
                  "Gets an affine expression containing the rounded-down "
                  "result of dividing an expression by a constant.")
      .def_static("get_ceil_div", &PyAffineCeilDivExpr::get,
                  "Gets an affine expression containing the rounded-up result "
                  "of dividing one expression by another.")
      .def_static("get_ceil_div", &PyAffineCeilDivExpr::getLHSConstant,
                  "Gets a semi-affine expression containing the rounded-up "
                  "result of dividing a constant by an expression.")
      .def_static("get_ceil_div", &PyAffineCeilDivExpr::getRHSConstant,
                  "Gets an affine expression containing the rounded-up result "
                  "of dividing an expression by a constant.")
      .def_static("get_constant", &PyAffineConstantExpr::get, nb::arg("value"),
                  nb::arg("context") = nb::none(),
                  "Gets a constant affine expression with the given value.")
      .def_static(
          "get_dim", &PyAffineDimExpr::get, nb::arg("position"),
          nb::arg("context") = nb::none(),
          "Gets an affine expression of a dimension at the given position.")
      .def_static(
          "get_symbol", &PyAffineSymbolExpr::get, nb::arg("position"),
          nb::arg("context") = nb::none(),
          "Gets an affine expression of a symbol at the given position.")
      .def(
          "dump", [](PyAffineExpr &self) { aiirAffineExprDump(self); },
          kDumpDocstring);
  PyAffineConstantExpr::bind(m);
  PyAffineDimExpr::bind(m);
  PyAffineSymbolExpr::bind(m);
  PyAffineBinaryExpr::bind(m);
  PyAffineAddExpr::bind(m);
  PyAffineMulExpr::bind(m);
  PyAffineModExpr::bind(m);
  PyAffineFloorDivExpr::bind(m);
  PyAffineCeilDivExpr::bind(m);

  //----------------------------------------------------------------------------
  // Mapping of PyAffineMap.
  //----------------------------------------------------------------------------
  nb::class_<PyAffineMap>(m, "AffineMap")
      .def_prop_ro(AIIR_PYTHON_CAPI_PTR_ATTR, &PyAffineMap::getCapsule)
      .def(AIIR_PYTHON_CAPI_FACTORY_ATTR, &PyAffineMap::createFromCapsule)
      .def("__eq__",
           [](PyAffineMap &self, PyAffineMap &other) { return self == other; })
      .def("__eq__", [](PyAffineMap &self, nb::object &other) { return false; })
      .def("__str__",
           [](PyAffineMap &self) {
             PyPrintAccumulator printAccum;
             aiirAffineMapPrint(self, printAccum.getCallback(),
                                printAccum.getUserData());
             return printAccum.join();
           })
      .def("__repr__",
           [](PyAffineMap &self) {
             PyPrintAccumulator printAccum;
             printAccum.parts.append("AffineMap(");
             aiirAffineMapPrint(self, printAccum.getCallback(),
                                printAccum.getUserData());
             printAccum.parts.append(")");
             return printAccum.join();
           })
      .def("__hash__",
           [](PyAffineMap &self) {
             return std::hash<const void *>{}(self.get().ptr);
           })
      .def_static(
          "compress_unused_symbols",
          [](nb::typed<nb::sequence, PyAffineMap> affineMaps,
             DefaultingPyAiirContext context) {
            std::vector<AiirAffineMap> maps;
            pyListToVector<PyAffineMap, AiirAffineMap>(
                affineMaps, maps, "attempting to create an AffineMap");
            std::vector<AiirAffineMap> compressed(nb::len(affineMaps));
            auto populate = [](void *result, intptr_t idx, AiirAffineMap m) {
              static_cast<AiirAffineMap *>(result)[idx] = (m);
            };
            aiirAffineMapCompressUnusedSymbols(maps.data(), maps.size(),
                                               compressed.data(), populate);
            std::vector<PyAffineMap> res;
            res.reserve(compressed.size());
            for (auto m : compressed)
              res.emplace_back(context->getRef(), m);
            return res;
          })
      .def_prop_ro(
          "context",
          [](PyAffineMap &self) -> nb::typed<nb::object, PyAiirContext> {
            return self.getContext().getObject();
          },
          "Context that owns the Affine Map")
      .def(
          "dump", [](PyAffineMap &self) { aiirAffineMapDump(self); },
          kDumpDocstring)
      .def_static(
          "get",
          [](intptr_t dimCount, intptr_t symbolCount,
             nb::typed<nb::sequence, PyAffineExpr> exprs,
             DefaultingPyAiirContext context) {
            std::vector<AiirAffineExpr> affineExprs;
            pyListToVector<PyAffineExpr, AiirAffineExpr>(
                exprs, affineExprs, "attempting to create an AffineMap");
            AiirAffineMap map =
                aiirAffineMapGet(context->get(), dimCount, symbolCount,
                                 affineExprs.size(), affineExprs.data());
            return PyAffineMap(context->getRef(), map);
          },
          nb::arg("dim_count"), nb::arg("symbol_count"), nb::arg("exprs"),
          nb::arg("context") = nb::none(),
          "Gets a map with the given expressions as results.")
      .def_static(
          "get_constant",
          [](intptr_t value, DefaultingPyAiirContext context) {
            AiirAffineMap affineMap =
                aiirAffineMapConstantGet(context->get(), value);
            return PyAffineMap(context->getRef(), affineMap);
          },
          nb::arg("value"), nb::arg("context") = nb::none(),
          "Gets an affine map with a single constant result")
      .def_static(
          "get_empty",
          [](DefaultingPyAiirContext context) {
            AiirAffineMap affineMap = aiirAffineMapEmptyGet(context->get());
            return PyAffineMap(context->getRef(), affineMap);
          },
          nb::arg("context") = nb::none(), "Gets an empty affine map.")
      .def_static(
          "get_identity",
          [](intptr_t nDims, DefaultingPyAiirContext context) {
            AiirAffineMap affineMap =
                aiirAffineMapMultiDimIdentityGet(context->get(), nDims);
            return PyAffineMap(context->getRef(), affineMap);
          },
          nb::arg("n_dims"), nb::arg("context") = nb::none(),
          "Gets an identity map with the given number of dimensions.")
      .def_static(
          "get_minor_identity",
          [](intptr_t nDims, intptr_t nResults,
             DefaultingPyAiirContext context) {
            AiirAffineMap affineMap =
                aiirAffineMapMinorIdentityGet(context->get(), nDims, nResults);
            return PyAffineMap(context->getRef(), affineMap);
          },
          nb::arg("n_dims"), nb::arg("n_results"),
          nb::arg("context") = nb::none(),
          "Gets a minor identity map with the given number of dimensions and "
          "results.")
      .def_static(
          "get_permutation",
          [](std::vector<unsigned> permutation,
             DefaultingPyAiirContext context) {
            if (!isPermutation(permutation))
              throw std::runtime_error("Invalid permutation when attempting to "
                                       "create an AffineMap");
            AiirAffineMap affineMap = aiirAffineMapPermutationGet(
                context->get(), permutation.size(), permutation.data());
            return PyAffineMap(context->getRef(), affineMap);
          },
          nb::arg("permutation"), nb::arg("context") = nb::none(),
          "Gets an affine map that permutes its inputs.")
      .def(
          "get_submap",
          [](PyAffineMap &self, std::vector<intptr_t> &resultPos) {
            intptr_t numResults = aiirAffineMapGetNumResults(self);
            for (intptr_t pos : resultPos) {
              if (pos < 0 || pos >= numResults)
                throw nb::value_error("result position out of bounds");
            }
            AiirAffineMap affineMap = aiirAffineMapGetSubMap(
                self, resultPos.size(), resultPos.data());
            return PyAffineMap(self.getContext(), affineMap);
          },
          nb::arg("result_positions"))
      .def(
          "get_major_submap",
          [](PyAffineMap &self, intptr_t nResults) {
            if (nResults >= aiirAffineMapGetNumResults(self))
              throw nb::value_error("number of results out of bounds");
            AiirAffineMap affineMap =
                aiirAffineMapGetMajorSubMap(self, nResults);
            return PyAffineMap(self.getContext(), affineMap);
          },
          nb::arg("n_results"))
      .def(
          "get_minor_submap",
          [](PyAffineMap &self, intptr_t nResults) {
            if (nResults >= aiirAffineMapGetNumResults(self))
              throw nb::value_error("number of results out of bounds");
            AiirAffineMap affineMap =
                aiirAffineMapGetMinorSubMap(self, nResults);
            return PyAffineMap(self.getContext(), affineMap);
          },
          nb::arg("n_results"))
      .def(
          "replace",
          [](PyAffineMap &self, PyAffineExpr &expression,
             PyAffineExpr &replacement, intptr_t numResultDims,
             intptr_t numResultSyms) {
            AiirAffineMap affineMap = aiirAffineMapReplace(
                self, expression, replacement, numResultDims, numResultSyms);
            return PyAffineMap(self.getContext(), affineMap);
          },
          nb::arg("expr"), nb::arg("replacement"), nb::arg("n_result_dims"),
          nb::arg("n_result_syms"))
      .def_prop_ro(
          "is_permutation",
          [](PyAffineMap &self) { return aiirAffineMapIsPermutation(self); })
      .def_prop_ro("is_projected_permutation",
                   [](PyAffineMap &self) {
                     return aiirAffineMapIsProjectedPermutation(self);
                   })
      .def_prop_ro(
          "n_dims",
          [](PyAffineMap &self) { return aiirAffineMapGetNumDims(self); })
      .def_prop_ro(
          "n_inputs",
          [](PyAffineMap &self) { return aiirAffineMapGetNumInputs(self); })
      .def_prop_ro(
          "n_symbols",
          [](PyAffineMap &self) { return aiirAffineMapGetNumSymbols(self); })
      .def_prop_ro("results",
                   [](PyAffineMap &self) { return PyAffineMapExprList(self); });
  PyAffineMapExprList::bind(m);

  //----------------------------------------------------------------------------
  // Mapping of PyIntegerSet.
  //----------------------------------------------------------------------------
  nb::class_<PyIntegerSet>(m, "IntegerSet")
      .def_prop_ro(AIIR_PYTHON_CAPI_PTR_ATTR, &PyIntegerSet::getCapsule)
      .def(AIIR_PYTHON_CAPI_FACTORY_ATTR, &PyIntegerSet::createFromCapsule)
      .def("__eq__", [](PyIntegerSet &self,
                        PyIntegerSet &other) { return self == other; })
      .def("__eq__",
           [](PyIntegerSet &self, const nb::object &other) { return false; })
      .def("__str__",
           [](PyIntegerSet &self) {
             PyPrintAccumulator printAccum;
             aiirIntegerSetPrint(self, printAccum.getCallback(),
                                 printAccum.getUserData());
             return printAccum.join();
           })
      .def("__repr__",
           [](PyIntegerSet &self) {
             PyPrintAccumulator printAccum;
             printAccum.parts.append("IntegerSet(");
             aiirIntegerSetPrint(self, printAccum.getCallback(),
                                 printAccum.getUserData());
             printAccum.parts.append(")");
             return printAccum.join();
           })
      .def("__hash__",
           [](PyIntegerSet &self) {
             return std::hash<const void *>{}(self.get().ptr);
           })
      .def_prop_ro(
          "context",
          [](PyIntegerSet &self) -> nb::typed<nb::object, PyAiirContext> {
            return self.getContext().getObject();
          })
      .def(
          "dump", [](PyIntegerSet &self) { aiirIntegerSetDump(self); },
          kDumpDocstring)
      .def_static(
          "get",
          [](intptr_t numDims, intptr_t numSymbols,
             nb::typed<nb::sequence, PyAffineExpr> exprs,
             std::vector<bool> eqFlags, DefaultingPyAiirContext context) {
            if (nb::len(exprs) != eqFlags.size())
              throw nb::value_error(
                  "Expected the number of constraints to match "
                  "that of equality flags");
            if (nb::len(exprs) == 0)
              throw nb::value_error("Expected non-empty list of constraints");

            // std::vector<bool> does not expose a bool* data pointer.
            std::vector<char> flags(eqFlags.begin(), eqFlags.end());
            std::vector<AiirAffineExpr> affineExprs;
            pyListToVector<PyAffineExpr>(exprs, affineExprs,
                                         "attempting to create an IntegerSet");
            AiirIntegerSet set = aiirIntegerSetGet(
                context->get(), numDims, numSymbols, nb::len(exprs),
                affineExprs.data(), reinterpret_cast<bool *>(flags.data()));
            return PyIntegerSet(context->getRef(), set);
          },
          nb::arg("num_dims"), nb::arg("num_symbols"), nb::arg("exprs"),
          nb::arg("eq_flags"), nb::arg("context") = nb::none())
      .def_static(
          "get_empty",
          [](intptr_t numDims, intptr_t numSymbols,
             DefaultingPyAiirContext context) {
            AiirIntegerSet set =
                aiirIntegerSetEmptyGet(context->get(), numDims, numSymbols);
            return PyIntegerSet(context->getRef(), set);
          },
          nb::arg("num_dims"), nb::arg("num_symbols"),
          nb::arg("context") = nb::none())
      .def(
          "get_replaced",
          [](PyIntegerSet &self, nb::typed<nb::sequence, PyAffineExpr> dimExprs,
             nb::typed<nb::sequence, PyAffineExpr> symbolExprs,
             intptr_t numResultDims, intptr_t numResultSymbols) {
            if (static_cast<intptr_t>(nb::len(dimExprs)) !=
                aiirIntegerSetGetNumDims(self))
              throw nb::value_error(
                  "Expected the number of dimension replacement expressions "
                  "to match that of dimensions");
            if (static_cast<intptr_t>(nb::len(symbolExprs)) !=
                aiirIntegerSetGetNumSymbols(self))
              throw nb::value_error(
                  "Expected the number of symbol replacement expressions "
                  "to match that of symbols");

            std::vector<AiirAffineExpr> dimAffineExprs;
            std::vector<AiirAffineExpr> symbolAffineExprs;
            pyListToVector<PyAffineExpr>(
                dimExprs, dimAffineExprs,
                "attempting to create an IntegerSet by replacing dimensions");
            pyListToVector<PyAffineExpr>(
                symbolExprs, symbolAffineExprs,
                "attempting to create an IntegerSet by replacing symbols");
            AiirIntegerSet set = aiirIntegerSetReplaceGet(
                self, dimAffineExprs.data(), symbolAffineExprs.data(),
                numResultDims, numResultSymbols);
            return PyIntegerSet(self.getContext(), set);
          },
          nb::arg("dim_exprs"), nb::arg("symbol_exprs"),
          nb::arg("num_result_dims"), nb::arg("num_result_symbols"))
      .def_prop_ro("is_canonical_empty",
                   [](PyIntegerSet &self) {
                     return aiirIntegerSetIsCanonicalEmpty(self);
                   })
      .def_prop_ro(
          "n_dims",
          [](PyIntegerSet &self) { return aiirIntegerSetGetNumDims(self); })
      .def_prop_ro(
          "n_symbols",
          [](PyIntegerSet &self) { return aiirIntegerSetGetNumSymbols(self); })
      .def_prop_ro(
          "n_inputs",
          [](PyIntegerSet &self) { return aiirIntegerSetGetNumInputs(self); })
      .def_prop_ro("n_equalities",
                   [](PyIntegerSet &self) {
                     return aiirIntegerSetGetNumEqualities(self);
                   })
      .def_prop_ro("n_inequalities",
                   [](PyIntegerSet &self) {
                     return aiirIntegerSetGetNumInequalities(self);
                   })
      .def_prop_ro("constraints", [](PyIntegerSet &self) {
        return PyIntegerSetConstraintList(self);
      });
  PyIntegerSetConstraint::bind(m);
  PyIntegerSetConstraintList::bind(m);
}
} // namespace AIIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace aiir
