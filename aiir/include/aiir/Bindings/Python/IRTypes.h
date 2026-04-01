//===- IRTypes.h - Type Interfaces ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_BINDINGS_PYTHON_IRTYPES_H
#define AIIR_BINDINGS_PYTHON_IRTYPES_H

#include "aiir-c/BuiltinTypes.h"
#include "aiir/Bindings/Python/IRCore.h"

namespace aiir {
namespace python {
namespace AIIR_BINDINGS_PYTHON_DOMAIN {

AIIR_PYTHON_API_EXPORTED int aiirTypeIsAIntegerOrFloat(AiirType type);

class AIIR_PYTHON_API_EXPORTED PyIntegerType
    : public PyConcreteType<PyIntegerType> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirTypeIsAInteger;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirIntegerTypeGetTypeID;
  static constexpr const char *pyClassName = "IntegerType";
  static inline const AiirStringRef name = aiirIntegerTypeGetName();
  using PyConcreteType::PyConcreteType;

  enum Signedness { Signless, Signed, Unsigned };

  static void bindDerived(ClassTy &c);
};

/// Index Type subclass - IndexType.
class AIIR_PYTHON_API_EXPORTED PyIndexType
    : public PyConcreteType<PyIndexType> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirTypeIsAIndex;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirIndexTypeGetTypeID;
  static constexpr const char *pyClassName = "IndexType";
  static inline const AiirStringRef name = aiirIndexTypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

class AIIR_PYTHON_API_EXPORTED PyFloatType
    : public PyConcreteType<PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirTypeIsAFloat;
  static constexpr const char *pyClassName = "FloatType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Floating Point Type subclass - Float4E2M1FNType.
class AIIR_PYTHON_API_EXPORTED PyFloat4E2M1FNType
    : public PyConcreteType<PyFloat4E2M1FNType, PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirTypeIsAFloat4E2M1FN;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirFloat4E2M1FNTypeGetTypeID;
  static constexpr const char *pyClassName = "Float4E2M1FNType";
  static inline const AiirStringRef name = aiirFloat4E2M1FNTypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Floating Point Type subclass - Float6E2M3FNType.
class AIIR_PYTHON_API_EXPORTED PyFloat6E2M3FNType
    : public PyConcreteType<PyFloat6E2M3FNType, PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirTypeIsAFloat6E2M3FN;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirFloat6E2M3FNTypeGetTypeID;
  static constexpr const char *pyClassName = "Float6E2M3FNType";
  static inline const AiirStringRef name = aiirFloat6E2M3FNTypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Floating Point Type subclass - Float6E3M2FNType.
class AIIR_PYTHON_API_EXPORTED PyFloat6E3M2FNType
    : public PyConcreteType<PyFloat6E3M2FNType, PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirTypeIsAFloat6E3M2FN;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirFloat6E3M2FNTypeGetTypeID;
  static constexpr const char *pyClassName = "Float6E3M2FNType";
  static inline const AiirStringRef name = aiirFloat6E3M2FNTypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Floating Point Type subclass - Float8E4M3FNType.
class AIIR_PYTHON_API_EXPORTED PyFloat8E4M3FNType
    : public PyConcreteType<PyFloat8E4M3FNType, PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirTypeIsAFloat8E4M3FN;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirFloat8E4M3FNTypeGetTypeID;
  static constexpr const char *pyClassName = "Float8E4M3FNType";
  static inline const AiirStringRef name = aiirFloat8E4M3FNTypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Floating Point Type subclass - Float8E5M2Type.
class AIIR_PYTHON_API_EXPORTED PyFloat8E5M2Type
    : public PyConcreteType<PyFloat8E5M2Type, PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirTypeIsAFloat8E5M2;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirFloat8E5M2TypeGetTypeID;
  static constexpr const char *pyClassName = "Float8E5M2Type";
  static inline const AiirStringRef name = aiirFloat8E5M2TypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Floating Point Type subclass - Float8E4M3Type.
class AIIR_PYTHON_API_EXPORTED PyFloat8E4M3Type
    : public PyConcreteType<PyFloat8E4M3Type, PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirTypeIsAFloat8E4M3;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirFloat8E4M3TypeGetTypeID;
  static constexpr const char *pyClassName = "Float8E4M3Type";
  static inline const AiirStringRef name = aiirFloat8E4M3TypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Floating Point Type subclass - Float8E4M3FNUZ.
class AIIR_PYTHON_API_EXPORTED PyFloat8E4M3FNUZType
    : public PyConcreteType<PyFloat8E4M3FNUZType, PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirTypeIsAFloat8E4M3FNUZ;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirFloat8E4M3FNUZTypeGetTypeID;
  static constexpr const char *pyClassName = "Float8E4M3FNUZType";
  static inline const AiirStringRef name = aiirFloat8E4M3FNUZTypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Floating Point Type subclass - Float8E4M3B11FNUZ.
class AIIR_PYTHON_API_EXPORTED PyFloat8E4M3B11FNUZType
    : public PyConcreteType<PyFloat8E4M3B11FNUZType, PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirTypeIsAFloat8E4M3B11FNUZ;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirFloat8E4M3B11FNUZTypeGetTypeID;
  static constexpr const char *pyClassName = "Float8E4M3B11FNUZType";
  static inline const AiirStringRef name = aiirFloat8E4M3B11FNUZTypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Floating Point Type subclass - Float8E5M2FNUZ.
class AIIR_PYTHON_API_EXPORTED PyFloat8E5M2FNUZType
    : public PyConcreteType<PyFloat8E5M2FNUZType, PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirTypeIsAFloat8E5M2FNUZ;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirFloat8E5M2FNUZTypeGetTypeID;
  static constexpr const char *pyClassName = "Float8E5M2FNUZType";
  static inline const AiirStringRef name = aiirFloat8E5M2FNUZTypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Floating Point Type subclass - Float8E3M4Type.
class AIIR_PYTHON_API_EXPORTED PyFloat8E3M4Type
    : public PyConcreteType<PyFloat8E3M4Type, PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirTypeIsAFloat8E3M4;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirFloat8E3M4TypeGetTypeID;
  static constexpr const char *pyClassName = "Float8E3M4Type";
  static inline const AiirStringRef name = aiirFloat8E3M4TypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Floating Point Type subclass - Float8E8M0FNUType.
class AIIR_PYTHON_API_EXPORTED PyFloat8E8M0FNUType
    : public PyConcreteType<PyFloat8E8M0FNUType, PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirTypeIsAFloat8E8M0FNU;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirFloat8E8M0FNUTypeGetTypeID;
  static constexpr const char *pyClassName = "Float8E8M0FNUType";
  static inline const AiirStringRef name = aiirFloat8E8M0FNUTypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Floating Point Type subclass - BF16Type.
class AIIR_PYTHON_API_EXPORTED PyBF16Type
    : public PyConcreteType<PyBF16Type, PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirTypeIsABF16;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirBFloat16TypeGetTypeID;
  static constexpr const char *pyClassName = "BF16Type";
  static inline const AiirStringRef name = aiirBF16TypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Floating Point Type subclass - F16Type.
class AIIR_PYTHON_API_EXPORTED PyF16Type
    : public PyConcreteType<PyF16Type, PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirTypeIsAF16;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirFloat16TypeGetTypeID;
  static constexpr const char *pyClassName = "F16Type";
  static inline const AiirStringRef name = aiirF16TypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Floating Point Type subclass - TF32Type.
class AIIR_PYTHON_API_EXPORTED PyTF32Type
    : public PyConcreteType<PyTF32Type, PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirTypeIsATF32;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirFloatTF32TypeGetTypeID;
  static constexpr const char *pyClassName = "FloatTF32Type";
  static inline const AiirStringRef name = aiirTF32TypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Floating Point Type subclass - F32Type.
class AIIR_PYTHON_API_EXPORTED PyF32Type
    : public PyConcreteType<PyF32Type, PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirTypeIsAF32;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirFloat32TypeGetTypeID;
  static constexpr const char *pyClassName = "F32Type";
  static inline const AiirStringRef name = aiirF32TypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Floating Point Type subclass - F64Type.
class AIIR_PYTHON_API_EXPORTED PyF64Type
    : public PyConcreteType<PyF64Type, PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirTypeIsAF64;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirFloat64TypeGetTypeID;
  static constexpr const char *pyClassName = "F64Type";
  static inline const AiirStringRef name = aiirF64TypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// None Type subclass - NoneType.
class AIIR_PYTHON_API_EXPORTED PyNoneType : public PyConcreteType<PyNoneType> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirTypeIsANone;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirNoneTypeGetTypeID;
  static constexpr const char *pyClassName = "NoneType";
  static inline const AiirStringRef name = aiirNoneTypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Complex Type subclass - ComplexType.
class AIIR_PYTHON_API_EXPORTED PyComplexType
    : public PyConcreteType<PyComplexType> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirTypeIsAComplex;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirComplexTypeGetTypeID;
  static constexpr const char *pyClassName = "ComplexType";
  static inline const AiirStringRef name = aiirComplexTypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Shaped Type Interface - ShapedType
class AIIR_PYTHON_API_EXPORTED AIIR_PYTHON_API_EXPORTED PyShapedType
    : public PyConcreteType<PyShapedType> {
public:
  static const IsAFunctionTy isaFunction;
  static constexpr const char *pyClassName = "ShapedType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);

private:
  void requireHasRank();
};

/// Vector Type subclass - VectorType.
class AIIR_PYTHON_API_EXPORTED PyVectorType
    : public PyConcreteType<PyVectorType, PyShapedType> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirTypeIsAVector;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirVectorTypeGetTypeID;
  static constexpr const char *pyClassName = "VectorType";
  static inline const AiirStringRef name = aiirVectorTypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);

private:
  static PyVectorType
  getChecked(std::vector<int64_t> shape, PyType &elementType,
             std::optional<nanobind::sequence> scalable,
             std::optional<std::vector<int64_t>> scalableDims,
             DefaultingPyLocation loc);

  static PyVectorType get(std::vector<int64_t> shape, PyType &elementType,
                          std::optional<nanobind::sequence> scalable,
                          std::optional<std::vector<int64_t>> scalableDims,
                          DefaultingPyAiirContext context);
};

/// Ranked Tensor Type subclass - RankedTensorType.
class AIIR_PYTHON_API_EXPORTED PyRankedTensorType
    : public PyConcreteType<PyRankedTensorType, PyShapedType> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirTypeIsARankedTensor;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirRankedTensorTypeGetTypeID;
  static constexpr const char *pyClassName = "RankedTensorType";
  static inline const AiirStringRef name = aiirRankedTensorTypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Unranked Tensor Type subclass - UnrankedTensorType.
class AIIR_PYTHON_API_EXPORTED PyUnrankedTensorType
    : public PyConcreteType<PyUnrankedTensorType, PyShapedType> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirTypeIsAUnrankedTensor;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirUnrankedTensorTypeGetTypeID;
  static constexpr const char *pyClassName = "UnrankedTensorType";
  static inline const AiirStringRef name = aiirUnrankedTensorTypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Ranked MemRef Type subclass - MemRefType.
class AIIR_PYTHON_API_EXPORTED PyMemRefType
    : public PyConcreteType<PyMemRefType, PyShapedType> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirTypeIsAMemRef;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirMemRefTypeGetTypeID;
  static constexpr const char *pyClassName = "MemRefType";
  static inline const AiirStringRef name = aiirMemRefTypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Unranked MemRef Type subclass - UnrankedMemRefType.
class AIIR_PYTHON_API_EXPORTED PyUnrankedMemRefType
    : public PyConcreteType<PyUnrankedMemRefType, PyShapedType> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirTypeIsAUnrankedMemRef;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirUnrankedMemRefTypeGetTypeID;
  static constexpr const char *pyClassName = "UnrankedMemRefType";
  static inline const AiirStringRef name = aiirUnrankedMemRefTypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Tuple Type subclass - TupleType.
class AIIR_PYTHON_API_EXPORTED PyTupleType
    : public PyConcreteType<PyTupleType> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirTypeIsATuple;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirTupleTypeGetTypeID;
  static constexpr const char *pyClassName = "TupleType";
  static inline const AiirStringRef name = aiirTupleTypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Function type.
class AIIR_PYTHON_API_EXPORTED PyFunctionType
    : public PyConcreteType<PyFunctionType> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirTypeIsAFunction;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirFunctionTypeGetTypeID;
  static constexpr const char *pyClassName = "FunctionType";
  static inline const AiirStringRef name = aiirFunctionTypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Opaque Type subclass - OpaqueType.
class AIIR_PYTHON_API_EXPORTED PyOpaqueType
    : public PyConcreteType<PyOpaqueType> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirTypeIsAOpaque;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirOpaqueTypeGetTypeID;
  static constexpr const char *pyClassName = "OpaqueType";
  static inline const AiirStringRef name = aiirOpaqueTypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

class AIIR_PYTHON_API_EXPORTED PyDynamicType
    : public PyConcreteType<PyDynamicType> {
public:
  static constexpr IsAFunctionTy isaFunction = aiirTypeIsADynamicType;
  static constexpr const char *pyClassName = "DynamicType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

AIIR_PYTHON_API_EXPORTED void populateIRTypes(nanobind::module_ &m);
} // namespace AIIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace aiir

#endif // AIIR_BINDINGS_PYTHON_IRTYPES_H
