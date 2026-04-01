//===- IRAttributes.h - Exports builtin and standard attributes -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_BINDINGS_PYTHON_IRATTRIBUTES_H
#define AIIR_BINDINGS_PYTHON_IRATTRIBUTES_H

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "aiir-c/BuiltinAttributes.h"
#include "aiir-c/BuiltinTypes.h"
#include "aiir/Bindings/Python/IRCore.h"
#include "aiir/Bindings/Python/Nanobind.h"
#include "aiir/Bindings/Python/NanobindAdaptors.h"
#include "aiir/Bindings/Python/NanobindUtils.h"

namespace aiir {
namespace python {
namespace AIIR_BINDINGS_PYTHON_DOMAIN {

struct nb_buffer_info {
  void *ptr = nullptr;
  Py_ssize_t itemsize = 0;
  Py_ssize_t size = 0;
  const char *format = nullptr;
  Py_ssize_t ndim = 0;
  std::vector<Py_ssize_t> shape;
  std::vector<Py_ssize_t> strides;
  bool readonly = false;

  nb_buffer_info(
      void *ptr, Py_ssize_t itemsize, const char *format, Py_ssize_t ndim,
      std::vector<Py_ssize_t> shape_in, std::vector<Py_ssize_t> strides_in,
      bool readonly = false,
      std::unique_ptr<Py_buffer, void (*)(Py_buffer *)> owned_view_in =
          std::unique_ptr<Py_buffer, void (*)(Py_buffer *)>(nullptr, nullptr));

  explicit nb_buffer_info(Py_buffer *view)
      : nb_buffer_info(view->buf, view->itemsize, view->format, view->ndim,
                       {view->shape, view->shape + view->ndim},
                       // TODO(phawkins): check for null strides
                       {view->strides, view->strides + view->ndim},
                       view->readonly != 0,
                       std::unique_ptr<Py_buffer, void (*)(Py_buffer *)>(
                           view, PyBuffer_Release)) {}

  nb_buffer_info(const nb_buffer_info &) = delete;
  nb_buffer_info(nb_buffer_info &&) = default;
  nb_buffer_info &operator=(const nb_buffer_info &) = delete;
  nb_buffer_info &operator=(nb_buffer_info &&) = default;

private:
  std::unique_ptr<Py_buffer, void (*)(Py_buffer *)> owned_view;
};

class AIIR_PYTHON_API_EXPORTED nb_buffer : public nanobind::object {
  NB_OBJECT_DEFAULT(nb_buffer, object, "Buffer", PyObject_CheckBuffer);

  nb_buffer_info request() const;
};

template <typename T>
struct nb_format_descriptor {};

class AIIR_PYTHON_API_EXPORTED PyAffineMapAttribute
    : public PyConcreteAttribute<PyAffineMapAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirAttributeIsAAffineMap;
  static constexpr const char *pyClassName = "AffineMapAttr";
  using PyConcreteAttribute::PyConcreteAttribute;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirAffineMapAttrGetTypeID;
  static inline const AiirStringRef name = aiirAffineMapAttrGetName();

  static void bindDerived(ClassTy &c);
};

class AIIR_PYTHON_API_EXPORTED PyIntegerSetAttribute
    : public PyConcreteAttribute<PyIntegerSetAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirAttributeIsAIntegerSet;
  static constexpr const char *pyClassName = "IntegerSetAttr";
  using PyConcreteAttribute::PyConcreteAttribute;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirIntegerSetAttrGetTypeID;
  static inline const AiirStringRef name = aiirIntegerSetAttrGetName();

  static void bindDerived(ClassTy &c);
};

template <typename T>
static T pyTryCast(nanobind::handle object) {
  try {
    return nanobind::cast<T>(object);
  } catch (nanobind::cast_error &err) {
    std::string msg = std::string("Invalid attribute when attempting to "
                                  "create an ArrayAttribute (") +
                      err.what() + ")";
    throw std::runtime_error(msg.c_str());
  } catch (std::runtime_error &err) {
    std::string msg = std::string("Invalid attribute (None?) when attempting "
                                  "to create an ArrayAttribute (") +
                      err.what() + ")";
    throw std::runtime_error(msg.c_str());
  }
}

/// A python-wrapped dense array attribute with an element type and a derived
/// implementation class.
template <typename EltTy, typename DerivedT>
class AIIR_PYTHON_API_EXPORTED PyDenseArrayAttribute
    : public PyConcreteAttribute<DerivedT> {
public:
  using PyConcreteAttribute<DerivedT>::PyConcreteAttribute;

  /// Iterator over the integer elements of a dense array.
  class PyDenseArrayIterator {
  public:
    PyDenseArrayIterator(PyAttribute attr) : attr(std::move(attr)) {}

    /// Return a copy of the iterator.
    PyDenseArrayIterator dunderIter() { return *this; }

    /// Return the next element.
    EltTy dunderNext() {
      // Throw if the index has reached the end.
      if (nextIndex >= aiirDenseArrayGetNumElements(attr.get()))
        throw nanobind::stop_iteration();
      return DerivedT::getElement(attr.get(), nextIndex++);
    }

    /// Bind the iterator class.
    static void bind(nanobind::module_ &m) {
      nanobind::class_<PyDenseArrayIterator>(m, DerivedT::pyIteratorName)
          .def("__iter__", &PyDenseArrayIterator::dunderIter)
          .def("__next__", &PyDenseArrayIterator::dunderNext);
    }

  private:
    /// The referenced dense array attribute.
    PyAttribute attr;
    /// The next index to read.
    int nextIndex = 0;
  };

  /// Get the element at the given index.
  EltTy getItem(intptr_t i) { return DerivedT::getElement(*this, i); }

  /// Bind the attribute class.
  static void bindDerived(typename PyConcreteAttribute<DerivedT>::ClassTy &c) {
    // Bind the constructor.
    if constexpr (std::is_same_v<EltTy, bool>) {
      c.def_static(
          "get",
          [](const nanobind::sequence &py_values, DefaultingPyAiirContext ctx) {
            std::vector<bool> values;
            for (nanobind::handle py_value : py_values) {
              int is_true = PyObject_IsTrue(py_value.ptr());
              if (is_true < 0) {
                throw nanobind::python_error();
              }
              values.push_back(is_true);
            }
            return getAttribute(values, ctx->getRef());
          },
          nanobind::arg("values"), nanobind::arg("context") = nanobind::none(),
          "Gets a uniqued dense array attribute");
    } else {
      c.def_static(
          "get",
          [](const std::vector<EltTy> &values, DefaultingPyAiirContext ctx) {
            return getAttribute(values, ctx->getRef());
          },
          nanobind::arg("values"), nanobind::arg("context") = nanobind::none(),
          "Gets a uniqued dense array attribute");
    }
    // Bind the array methods.
    c.def("__getitem__", [](DerivedT &arr, intptr_t i) {
      if (i >= aiirDenseArrayGetNumElements(arr))
        throw nanobind::index_error("DenseArray index out of range");
      return arr.getItem(i);
    });
    c.def("__len__", [](const DerivedT &arr) {
      return aiirDenseArrayGetNumElements(arr);
    });
    c.def("__iter__",
          [](const DerivedT &arr) { return PyDenseArrayIterator(arr); });
    c.def("__add__", [](DerivedT &arr, const nanobind::sequence &extras) {
      std::vector<EltTy> values;
      intptr_t numOldElements = aiirDenseArrayGetNumElements(arr);
      values.reserve(numOldElements + nanobind::len(extras));
      for (intptr_t i = 0; i < numOldElements; ++i)
        values.push_back(arr.getItem(i));
      for (nanobind::handle attr : extras)
        values.push_back(pyTryCast<EltTy>(attr));
      return getAttribute(values, arr.getContext());
    });
  }

private:
  static DerivedT getAttribute(const std::vector<EltTy> &values,
                               PyAiirContextRef ctx) {
    if constexpr (std::is_same_v<EltTy, bool>) {
      std::vector<int> intValues(values.begin(), values.end());
      AiirAttribute attr = DerivedT::getAttribute(ctx->get(), intValues.size(),
                                                  intValues.data());
      return DerivedT(ctx, attr);
    } else {
      AiirAttribute attr =
          DerivedT::getAttribute(ctx->get(), values.size(), values.data());
      return DerivedT(ctx, attr);
    }
  }
};

/// Instantiate the python dense array classes.
struct PyDenseBoolArrayAttribute
    : public PyDenseArrayAttribute<bool, PyDenseBoolArrayAttribute> {
  static constexpr IsAFunctionTy isaFunction = aiirAttributeIsADenseBoolArray;
  static constexpr auto getAttribute = aiirDenseBoolArrayGet;
  static constexpr auto getElement = aiirDenseBoolArrayGetElement;
  static constexpr const char *pyClassName = "DenseBoolArrayAttr";
  static constexpr const char *pyIteratorName = "DenseBoolArrayIterator";
  using PyDenseArrayAttribute::PyDenseArrayAttribute;
};
struct PyDenseI8ArrayAttribute
    : public PyDenseArrayAttribute<int8_t, PyDenseI8ArrayAttribute> {
  static constexpr IsAFunctionTy isaFunction = aiirAttributeIsADenseI8Array;
  static constexpr auto getAttribute = aiirDenseI8ArrayGet;
  static constexpr auto getElement = aiirDenseI8ArrayGetElement;
  static constexpr const char *pyClassName = "DenseI8ArrayAttr";
  static constexpr const char *pyIteratorName = "DenseI8ArrayIterator";
  using PyDenseArrayAttribute::PyDenseArrayAttribute;
};
struct PyDenseI16ArrayAttribute
    : public PyDenseArrayAttribute<int16_t, PyDenseI16ArrayAttribute> {
  static constexpr IsAFunctionTy isaFunction = aiirAttributeIsADenseI16Array;
  static constexpr auto getAttribute = aiirDenseI16ArrayGet;
  static constexpr auto getElement = aiirDenseI16ArrayGetElement;
  static constexpr const char *pyClassName = "DenseI16ArrayAttr";
  static constexpr const char *pyIteratorName = "DenseI16ArrayIterator";
  using PyDenseArrayAttribute::PyDenseArrayAttribute;
};
struct PyDenseI32ArrayAttribute
    : public PyDenseArrayAttribute<int32_t, PyDenseI32ArrayAttribute> {
  static constexpr IsAFunctionTy isaFunction = aiirAttributeIsADenseI32Array;
  static constexpr auto getAttribute = aiirDenseI32ArrayGet;
  static constexpr auto getElement = aiirDenseI32ArrayGetElement;
  static constexpr const char *pyClassName = "DenseI32ArrayAttr";
  static constexpr const char *pyIteratorName = "DenseI32ArrayIterator";
  using PyDenseArrayAttribute::PyDenseArrayAttribute;
};
struct PyDenseI64ArrayAttribute
    : public PyDenseArrayAttribute<int64_t, PyDenseI64ArrayAttribute> {
  static constexpr IsAFunctionTy isaFunction = aiirAttributeIsADenseI64Array;
  static constexpr auto getAttribute = aiirDenseI64ArrayGet;
  static constexpr auto getElement = aiirDenseI64ArrayGetElement;
  static constexpr const char *pyClassName = "DenseI64ArrayAttr";
  static constexpr const char *pyIteratorName = "DenseI64ArrayIterator";
  using PyDenseArrayAttribute::PyDenseArrayAttribute;
};
struct PyDenseF32ArrayAttribute
    : public PyDenseArrayAttribute<float, PyDenseF32ArrayAttribute> {
  static constexpr IsAFunctionTy isaFunction = aiirAttributeIsADenseF32Array;
  static constexpr auto getAttribute = aiirDenseF32ArrayGet;
  static constexpr auto getElement = aiirDenseF32ArrayGetElement;
  static constexpr const char *pyClassName = "DenseF32ArrayAttr";
  static constexpr const char *pyIteratorName = "DenseF32ArrayIterator";
  using PyDenseArrayAttribute::PyDenseArrayAttribute;
};
struct PyDenseF64ArrayAttribute
    : public PyDenseArrayAttribute<double, PyDenseF64ArrayAttribute> {
  static constexpr IsAFunctionTy isaFunction = aiirAttributeIsADenseF64Array;
  static constexpr auto getAttribute = aiirDenseF64ArrayGet;
  static constexpr auto getElement = aiirDenseF64ArrayGetElement;
  static constexpr const char *pyClassName = "DenseF64ArrayAttr";
  static constexpr const char *pyIteratorName = "DenseF64ArrayIterator";
  using PyDenseArrayAttribute::PyDenseArrayAttribute;
};

class AIIR_PYTHON_API_EXPORTED PyArrayAttribute
    : public PyConcreteAttribute<PyArrayAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirAttributeIsAArray;
  static constexpr const char *pyClassName = "ArrayAttr";
  using PyConcreteAttribute::PyConcreteAttribute;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirArrayAttrGetTypeID;
  static inline const AiirStringRef name = aiirArrayAttrGetName();

  class PyArrayAttributeIterator {
  public:
    PyArrayAttributeIterator(PyAttribute attr) : attr(std::move(attr)) {}

    PyArrayAttributeIterator &dunderIter() { return *this; }

    nanobind::typed<nanobind::object, PyAttribute> dunderNext();

    static void bind(nanobind::module_ &m);

  private:
    PyAttribute attr;
    int nextIndex = 0;
  };

  AiirAttribute getItem(intptr_t i) const;

  static void bindDerived(ClassTy &c);
};

/// Float Point Attribute subclass - FloatAttr.
class AIIR_PYTHON_API_EXPORTED PyFloatAttribute
    : public PyConcreteAttribute<PyFloatAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirAttributeIsAFloat;
  static constexpr const char *pyClassName = "FloatAttr";
  using PyConcreteAttribute::PyConcreteAttribute;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirFloatAttrGetTypeID;
  static inline const AiirStringRef name = aiirFloatAttrGetName();

  static void bindDerived(ClassTy &c);
};

/// Integer Attribute subclass - IntegerAttr.
class AIIR_PYTHON_API_EXPORTED PyIntegerAttribute
    : public PyConcreteAttribute<PyIntegerAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirAttributeIsAInteger;
  static constexpr const char *pyClassName = "IntegerAttr";
  using PyConcreteAttribute::PyConcreteAttribute;
  static inline const AiirStringRef name = aiirIntegerAttrGetName();

  static void bindDerived(ClassTy &c);

private:
  static nanobind::int_ toPyInt(PyIntegerAttribute &self);
};

/// Bool Attribute subclass - BoolAttr.
class AIIR_PYTHON_API_EXPORTED PyBoolAttribute
    : public PyConcreteAttribute<PyBoolAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirAttributeIsABool;
  static constexpr const char *pyClassName = "BoolAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c);
};

class AIIR_PYTHON_API_EXPORTED PySymbolRefAttribute
    : public PyConcreteAttribute<PySymbolRefAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirAttributeIsASymbolRef;
  static constexpr const char *pyClassName = "SymbolRefAttr";
  using PyConcreteAttribute::PyConcreteAttribute;
  static inline const AiirStringRef name = aiirSymbolRefAttrGetName();

  static PySymbolRefAttribute fromList(const std::vector<std::string> &symbols,
                                       PyAiirContext &context);

  static void bindDerived(ClassTy &c);
};

class AIIR_PYTHON_API_EXPORTED PyFlatSymbolRefAttribute
    : public PyConcreteAttribute<PyFlatSymbolRefAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirAttributeIsAFlatSymbolRef;
  static constexpr const char *pyClassName = "FlatSymbolRefAttr";
  using PyConcreteAttribute::PyConcreteAttribute;
  static inline const AiirStringRef name = aiirFlatSymbolRefAttrGetName();

  static void bindDerived(ClassTy &c);
};

class AIIR_PYTHON_API_EXPORTED PyOpaqueAttribute
    : public PyConcreteAttribute<PyOpaqueAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirAttributeIsAOpaque;
  static constexpr const char *pyClassName = "OpaqueAttr";
  using PyConcreteAttribute::PyConcreteAttribute;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirOpaqueAttrGetTypeID;
  static inline const AiirStringRef name = aiirOpaqueAttrGetName();

  static void bindDerived(ClassTy &c);
};

// TODO: Support construction of string elements.
class AIIR_PYTHON_API_EXPORTED PyDenseElementsAttribute
    : public PyConcreteAttribute<PyDenseElementsAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirAttributeIsADenseElements;
  static constexpr const char *pyClassName = "DenseElementsAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static PyDenseElementsAttribute getFromList(
      const nanobind::typed<nanobind::sequence, PyAttribute> &attributes,
      std::optional<PyType> explicitType,
      DefaultingPyAiirContext contextWrapper);

  static PyDenseElementsAttribute
  getFromBuffer(const nb_buffer &array, bool signless,
                const std::optional<PyType> &explicitType,
                std::optional<std::vector<int64_t>> explicitShape,
                DefaultingPyAiirContext contextWrapper);

  static PyDenseElementsAttribute getSplat(const PyType &shapedType,
                                           PyAttribute &elementAttr);

  intptr_t dunderLen() const;

  std::unique_ptr<nb_buffer_info> accessBuffer();

  static void bindDerived(ClassTy &c);

  static PyType_Slot slots[];

protected:
  /// Registers get/get_splat factory methods with the concrete return
  /// type in the nb::sig. Subclasses call this from their bindDerived
  /// to override the return type in generated stubs.
  template <typename ClassT>
  static void bindFactoryMethods(ClassT &c, const char *pyClassName);

private:
  static int bf_getbuffer(PyObject *exporter, Py_buffer *view, int flags);
  static void bf_releasebuffer(PyObject *, Py_buffer *buffer);

  static bool isUnsignedIntegerFormat(std::string_view format);

  static bool isSignedIntegerFormat(std::string_view format);

  static AiirType
  getShapedType(std::optional<AiirType> bulkLoadElementType,
                std::optional<std::vector<int64_t>> explicitShape,
                Py_buffer &view);

  static AiirAttribute getAttributeFromBuffer(
      Py_buffer &view, bool signless, std::optional<PyType> explicitType,
      const std::optional<std::vector<int64_t>> &explicitShape,
      AiirContext &context);

  template <typename Type>
  std::unique_ptr<nb_buffer_info>
  bufferInfo(AiirType shapedType, const char *explicitFormat = nullptr) {
    intptr_t rank = aiirShapedTypeGetRank(shapedType);
    // Prepare the data for the buffer_info.
    // Buffer is configured for read-only access below.
    Type *data = static_cast<Type *>(
        const_cast<void *>(aiirDenseElementsAttrGetRawData(*this)));
    // Prepare the shape for the buffer_info.
    std::vector<Py_ssize_t> shape;
    for (intptr_t i = 0; i < rank; ++i)
      shape.push_back(aiirShapedTypeGetDimSize(shapedType, i));
    // Prepare the strides for the buffer_info.
    std::vector<Py_ssize_t> strides;
    if (aiirDenseElementsAttrIsSplat(*this)) {
      // Splats are special, only the single value is stored.
      strides.assign(rank, 0);
    } else {
      for (intptr_t i = 1; i < rank; ++i) {
        intptr_t strideFactor = 1;
        for (intptr_t j = i; j < rank; ++j)
          strideFactor *= aiirShapedTypeGetDimSize(shapedType, j);
        strides.push_back(sizeof(Type) * strideFactor);
      }
      strides.push_back(sizeof(Type));
    }
    const char *format;
    if (explicitFormat) {
      format = explicitFormat;
    } else {
      format = nb_format_descriptor<Type>::format();
    }
    return std::make_unique<nb_buffer_info>(
        data, sizeof(Type), format, rank, std::move(shape), std::move(strides),
        /*readonly=*/true);
  }
};

/// Refinement of the PyDenseElementsAttribute for attributes containing
/// integer (and boolean) values. Supports element access.
class AIIR_PYTHON_API_EXPORTED PyDenseIntElementsAttribute
    : public PyConcreteAttribute<PyDenseIntElementsAttribute,
                                 PyDenseElementsAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirAttributeIsADenseIntElements;
  static constexpr const char *pyClassName = "DenseIntElementsAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  /// Returns the element at the given linear position. Asserts if the index
  /// is out of range.
  nanobind::int_ dunderGetItem(intptr_t pos) const;

  static void bindDerived(ClassTy &c);
};

class AIIR_PYTHON_API_EXPORTED PyDenseResourceElementsAttribute
    : public PyConcreteAttribute<PyDenseResourceElementsAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction =
      aiirAttributeIsADenseResourceElements;
  static constexpr const char *pyClassName = "DenseResourceElementsAttr";
  using PyConcreteAttribute::PyConcreteAttribute;
  static inline const AiirStringRef name =
      aiirDenseResourceElementsAttrGetName();

  static PyDenseResourceElementsAttribute
  getFromBuffer(const nb_buffer &buffer, const std::string &name,
                const PyType &type, std::optional<size_t> alignment,
                bool isMutable, DefaultingPyAiirContext contextWrapper);

  static void bindDerived(ClassTy &c);
};

class AIIR_PYTHON_API_EXPORTED PyDictAttribute
    : public PyConcreteAttribute<PyDictAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirAttributeIsADictionary;
  static constexpr const char *pyClassName = "DictAttr";
  using PyConcreteAttribute::PyConcreteAttribute;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirDictionaryAttrGetTypeID;
  static inline const AiirStringRef name = aiirDictionaryAttrGetName();

  intptr_t dunderLen() const;

  bool dunderContains(const std::string &name) const;

  static void bindDerived(ClassTy &c);
};

/// Refinement of PyDenseElementsAttribute for attributes containing
/// floating-point values. Supports element access.
class AIIR_PYTHON_API_EXPORTED PyDenseFPElementsAttribute
    : public PyConcreteAttribute<PyDenseFPElementsAttribute,
                                 PyDenseElementsAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirAttributeIsADenseFPElements;
  static constexpr const char *pyClassName = "DenseFPElementsAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  nanobind::float_ dunderGetItem(intptr_t pos) const;

  static void bindDerived(ClassTy &c);
};

class AIIR_PYTHON_API_EXPORTED PyTypeAttribute
    : public PyConcreteAttribute<PyTypeAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirAttributeIsAType;
  static constexpr const char *pyClassName = "TypeAttr";
  using PyConcreteAttribute::PyConcreteAttribute;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirTypeAttrGetTypeID;
  static inline const AiirStringRef name = aiirTypeAttrGetName();

  static void bindDerived(ClassTy &c);
};

/// Unit Attribute subclass. Unit attributes don't have values.
class AIIR_PYTHON_API_EXPORTED PyUnitAttribute
    : public PyConcreteAttribute<PyUnitAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirAttributeIsAUnit;
  static constexpr const char *pyClassName = "UnitAttr";
  using PyConcreteAttribute::PyConcreteAttribute;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirUnitAttrGetTypeID;
  static inline const AiirStringRef name = aiirUnitAttrGetName();

  static void bindDerived(ClassTy &c);
};

/// Strided layout attribute subclass.
class AIIR_PYTHON_API_EXPORTED PyStridedLayoutAttribute
    : public PyConcreteAttribute<PyStridedLayoutAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirAttributeIsAStridedLayout;
  static constexpr const char *pyClassName = "StridedLayoutAttr";
  using PyConcreteAttribute::PyConcreteAttribute;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirStridedLayoutAttrGetTypeID;
  static inline const AiirStringRef name = aiirStridedLayoutAttrGetName();

  static void bindDerived(ClassTy &c);
};

class AIIR_PYTHON_API_EXPORTED PyDynamicAttribute
    : public PyConcreteAttribute<PyDynamicAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirAttributeIsADynamicAttr;
  static constexpr const char *pyClassName = "DynamicAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c);
};

AIIR_PYTHON_API_EXPORTED void populateIRAttributes(nanobind::module_ &m);
} // namespace AIIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace aiir

#endif
