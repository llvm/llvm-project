//===-- C++ code generation from NamedFunctionDescriptors -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This code is responsible for generating the "Implementation.cpp" file.
// The file is composed like this:
//
// 1. Includes
// 2. Using statements to help readability.
// 3. Source code for all the mem function implementations.
// 4. The function to retrieve all the function descriptors with their name.
//      llvm::ArrayRef<NamedFunctionDescriptor> getFunctionDescriptors();
// 5. The functions for the benchmarking infrastructure:
//      llvm::ArrayRef<MemcpyConfiguration> getMemcpyConfigurations();
//      llvm::ArrayRef<MemcmpOrBcmpConfiguration> getMemcmpConfigurations();
//      llvm::ArrayRef<MemcmpOrBcmpConfiguration> getBcmpConfigurations();
//      llvm::ArrayRef<MemsetConfiguration> getMemsetConfigurations();
//      llvm::ArrayRef<BzeroConfiguration> getBzeroConfigurations();
//
//
// Sections 3, 4 and 5 are handled by the following namespaces:
// - codegen::functions
// - codegen::descriptors
// - codegen::configurations
//
// The programming style is functionnal. In each of these namespace, the
// original `NamedFunctionDescriptor` object is turned into a different type. We
// make use of overloaded stream operators to format the resulting type into
// either a function, a descriptor or a configuration. The entry point of each
// namespace is the Serialize function.
//
// Note the code here is better understood by starting from the `Serialize`
// function at the end of the file.

#include "automemcpy/CodeGen.h"
#include <cassert>
#include <llvm/ADT/Optional.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>
#include <set>

namespace llvm {
namespace automemcpy {
namespace codegen {

// The indentation string.
static constexpr StringRef kIndent = "  ";

// The codegen namespace handles the serialization of a NamedFunctionDescriptor
// into source code for the function, the descriptor and the configuration.

namespace functions {

// This namespace turns a NamedFunctionDescriptor into an actual implementation.
// -----------------------------------------------------------------------------
// e.g.
// static void memcpy_0xB20D4702493C397E(char *__restrict dst,
//                                       const char *__restrict src,
//                                       size_t size) {
//   using namespace __llvm_libc::x86;
//   if(size == 0) return;
//   if(size == 1) return copy<_1>(dst, src);
//   if(size < 4) return copy<HeadTail<_2>>(dst, src, size);
//   if(size < 8) return copy<HeadTail<_4>>(dst, src, size);
//   if(size < 16) return copy<HeadTail<_8>>(dst, src, size);
//   if(size < 32) return copy<HeadTail<_16>>(dst, src, size);
//   return copy<Accelerator>(dst, src, size);
// }

// The `Serialize` method turns a `NamedFunctionDescriptor` into a
// `FunctionImplementation` which holds all the information needed to produce
// the C++ source code.

// An Element with its size (e.g. `_16` in the example above).
struct ElementType {
  size_t Size;
};
// The case `if(size == 0)` is encoded as a the Zero type.
struct Zero {
  StringRef DefaultReturnValue;
};
// An individual size `if(size == X)` is encoded as an Individual type.
struct Individual {
  size_t IfEq;
  ElementType Element;
};
// An overlap strategy is encoded as an Overlap type.
struct Overlap {
  size_t IfLt;
  ElementType Element;
};
// A loop strategy is encoded as a Loop type.
struct Loop {
  size_t IfLt;
  ElementType Element;
};
// An aligned loop strategy is encoded as an AlignedLoop type.
struct AlignedLoop {
  size_t IfLt;
  ElementType Element;
  ElementType Alignment;
  StringRef AlignTo;
};
// The accelerator strategy.
struct Accelerator {
  size_t IfLt;
};
// The Context stores data about the function type.
struct Context {
  StringRef FunctionReturnType; // e.g. void* or int
  StringRef FunctionArgs;
  StringRef ElementOp; // copy, three_way_compare, splat_set, ...
  StringRef FixedSizeArgs;
  StringRef RuntimeSizeArgs;
  StringRef DefaultReturnValue;
};
// A detailed representation of the function implementation mapped from the
// NamedFunctionDescriptor.
struct FunctionImplementation {
  Context Ctx;
  StringRef Name;
  std::vector<Individual> Individuals;
  std::vector<Overlap> Overlaps;
  Optional<Loop> Loop;
  Optional<AlignedLoop> AlignedLoop;
  Optional<Accelerator> Accelerator;
  ElementTypeClass ElementClass;
};

// Returns the Context for each FunctionType.
static Context getCtx(FunctionType FT) {
  switch (FT) {
  case FunctionType::MEMCPY:
    return {"void",
            "(char *__restrict dst, const char *__restrict src, size_t size)",
            "copy",
            "(dst, src)",
            "(dst, src, size)",
            ""};
  case FunctionType::MEMCMP:
    return {"int",
            "(const char * lhs, const char * rhs, size_t size)",
            "three_way_compare",
            "(lhs, rhs)",
            "(lhs, rhs, size)",
            "0"};
  case FunctionType::MEMSET:
    return {"void",
            "(char * dst, int value, size_t size)",
            "splat_set",
            "(dst, value)",
            "(dst, value, size)",
            ""};
  case FunctionType::BZERO:
    return {"void",           "(char * dst, size_t size)",
            "splat_set",      "(dst, 0)",
            "(dst, 0, size)", ""};
  default:
    report_fatal_error("Not yet implemented");
  }
}

static StringRef getAligntoString(const AlignArg &AlignTo) {
  switch (AlignTo) {
  case AlignArg::_1:
    return "Arg::P1";
  case AlignArg::_2:
    return "Arg::P2";
  case AlignArg::ARRAY_SIZE:
    report_fatal_error("logic error");
  }
}

static raw_ostream &operator<<(raw_ostream &Stream, const ElementType &E) {
  return Stream << '_' << E.Size;
}
static raw_ostream &operator<<(raw_ostream &Stream, const Individual &O) {
  return Stream << O.Element;
}
static raw_ostream &operator<<(raw_ostream &Stream, const Overlap &O) {
  return Stream << "HeadTail<" << O.Element << '>';
}
static raw_ostream &operator<<(raw_ostream &Stream, const Loop &O) {
  return Stream << "Loop<" << O.Element << '>';
}
static raw_ostream &operator<<(raw_ostream &Stream, const AlignedLoop &O) {
  return Stream << "Align<" << O.Alignment << ',' << O.AlignTo << ">::Then<"
                << Loop{O.IfLt, O.Element} << ">";
}
static raw_ostream &operator<<(raw_ostream &Stream, const Accelerator &O) {
  return Stream << "Accelerator";
}

template <typename T> struct IfEq {
  StringRef Op;
  StringRef Args;
  const T &Element;
};

template <typename T> struct IfLt {
  StringRef Op;
  StringRef Args;
  const T &Element;
};

static raw_ostream &operator<<(raw_ostream &Stream, const Zero &O) {
  Stream << kIndent << "if(size == 0) return";
  if (!O.DefaultReturnValue.empty())
    Stream << ' ' << O.DefaultReturnValue;
  return Stream << ";\n";
}

template <typename T>
static raw_ostream &operator<<(raw_ostream &Stream, const IfEq<T> &O) {
  return Stream << kIndent << "if(size == " << O.Element.IfEq << ") return "
                << O.Op << '<' << O.Element << '>' << O.Args << ";\n";
}

template <typename T>
static raw_ostream &operator<<(raw_ostream &Stream, const IfLt<T> &O) {
  Stream << kIndent;
  if (O.Element.IfLt != kMaxSize)
    Stream << "if(size < " << O.Element.IfLt << ") ";
  return Stream << "return " << O.Op << '<' << O.Element << '>' << O.Args
                << ";\n";
}

static raw_ostream &operator<<(raw_ostream &Stream,
                               const ElementTypeClass &Class) {
  switch (Class) {
  case ElementTypeClass::SCALAR:
    return Stream << "scalar";
  case ElementTypeClass::BUILTIN:
    return Stream << "builtin";
  case ElementTypeClass::NATIVE:
    // FIXME: the framework should provide a `native` namespace that redirect to
    // x86, arm or other architectures.
    return Stream << "x86";
  }
}

static raw_ostream &operator<<(raw_ostream &Stream,
                               const FunctionImplementation &FI) {
  const auto &Ctx = FI.Ctx;
  Stream << "static " << Ctx.FunctionReturnType << ' ' << FI.Name
         << Ctx.FunctionArgs << " {\n";
  Stream << kIndent << "using namespace __llvm_libc::" << FI.ElementClass
         << ";\n";
  for (const auto &I : FI.Individuals)
    if (I.Element.Size == 0)
      Stream << Zero{Ctx.DefaultReturnValue};
    else
      Stream << IfEq<Individual>{Ctx.ElementOp, Ctx.FixedSizeArgs, I};
  for (const auto &O : FI.Overlaps)
    Stream << IfLt<Overlap>{Ctx.ElementOp, Ctx.RuntimeSizeArgs, O};
  if (const auto &C = FI.Loop)
    Stream << IfLt<Loop>{Ctx.ElementOp, Ctx.RuntimeSizeArgs, *C};
  if (const auto &C = FI.AlignedLoop)
    Stream << IfLt<AlignedLoop>{Ctx.ElementOp, Ctx.RuntimeSizeArgs, *C};
  if (const auto &C = FI.Accelerator)
    Stream << IfLt<Accelerator>{Ctx.ElementOp, Ctx.RuntimeSizeArgs, *C};
  return Stream << "}\n";
}

// Turns a `NamedFunctionDescriptor` into a `FunctionImplementation` unfolding
// the contiguous and overlap region into several statements. The zero case is
// also mapped to its own type.
static FunctionImplementation
getImplementation(const NamedFunctionDescriptor &NamedFD) {
  const FunctionDescriptor &FD = NamedFD.Desc;
  FunctionImplementation Impl;
  Impl.Ctx = getCtx(FD.Type);
  Impl.Name = NamedFD.Name;
  Impl.ElementClass = FD.ElementClass;
  if (auto C = FD.Contiguous)
    for (size_t I = C->Span.Begin; I < C->Span.End; ++I)
      Impl.Individuals.push_back(Individual{I, ElementType{I}});
  if (auto C = FD.Overlap)
    for (size_t I = C->Span.Begin; I < C->Span.End; I *= 2)
      Impl.Overlaps.push_back(Overlap{2 * I, ElementType{I}});
  if (const auto &L = FD.Loop)
    Impl.Loop = Loop{L->Span.End, ElementType{L->BlockSize}};
  if (const auto &AL = FD.AlignedLoop)
    Impl.AlignedLoop =
        AlignedLoop{AL->Loop.Span.End, ElementType{AL->Loop.BlockSize},
                    ElementType{AL->Alignment}, getAligntoString(AL->AlignTo)};
  if (const auto &A = FD.Accelerator)
    Impl.Accelerator = Accelerator{A->Span.End};
  return Impl;
}

static void Serialize(raw_ostream &Stream,
                      ArrayRef<NamedFunctionDescriptor> Descriptors) {

  for (const auto &FD : Descriptors)
    Stream << getImplementation(FD);
}

} // namespace functions

namespace descriptors {

// This namespace generates the getFunctionDescriptors function:
// -------------------------------------------------------------
// e.g.
// ArrayRef<NamedFunctionDescriptor> getFunctionDescriptors() {
//   static constexpr NamedFunctionDescriptor kDescriptors[] = {
//     {"memcpy_0xE00E29EE73994E2B",{FunctionType::MEMCPY,std::nullopt,std::nullopt,std::nullopt,std::nullopt,Accelerator{{0,kMaxSize}},ElementTypeClass::NATIVE}},
//     {"memcpy_0x8661D80472487AB5",{FunctionType::MEMCPY,Contiguous{{0,1}},std::nullopt,std::nullopt,std::nullopt,Accelerator{{1,kMaxSize}},ElementTypeClass::NATIVE}},
//     ...
//   };
//   return ArrayRef(kDescriptors);
// }

static raw_ostream &operator<<(raw_ostream &Stream, const SizeSpan &SS) {
  Stream << "{" << SS.Begin << ',';
  if (SS.End == kMaxSize)
    Stream << "kMaxSize";
  else
    Stream << SS.End;
  return Stream << '}';
}
static raw_ostream &operator<<(raw_ostream &Stream, const Contiguous &O) {
  return Stream << "Contiguous{" << O.Span << '}';
}
static raw_ostream &operator<<(raw_ostream &Stream, const Overlap &O) {
  return Stream << "Overlap{" << O.Span << '}';
}
static raw_ostream &operator<<(raw_ostream &Stream, const Loop &O) {
  return Stream << "Loop{" << O.Span << ',' << O.BlockSize << '}';
}
static raw_ostream &operator<<(raw_ostream &Stream, const AlignArg &O) {
  switch (O) {
  case AlignArg::_1:
    return Stream << "AlignArg::_1";
  case AlignArg::_2:
    return Stream << "AlignArg::_2";
  case AlignArg::ARRAY_SIZE:
    report_fatal_error("logic error");
  }
}
static raw_ostream &operator<<(raw_ostream &Stream, const AlignedLoop &O) {
  return Stream << "AlignedLoop{" << O.Loop << ',' << O.Alignment << ','
                << O.AlignTo << '}';
}
static raw_ostream &operator<<(raw_ostream &Stream, const Accelerator &O) {
  return Stream << "Accelerator{" << O.Span << '}';
}
static raw_ostream &operator<<(raw_ostream &Stream, const ElementTypeClass &O) {
  switch (O) {
  case ElementTypeClass::SCALAR:
    return Stream << "ElementTypeClass::SCALAR";
  case ElementTypeClass::BUILTIN:
    return Stream << "ElementTypeClass::BUILTIN";
  case ElementTypeClass::NATIVE:
    return Stream << "ElementTypeClass::NATIVE";
  }
}
static raw_ostream &operator<<(raw_ostream &Stream, const FunctionType &T) {
  switch (T) {
  case FunctionType::MEMCPY:
    return Stream << "FunctionType::MEMCPY";
  case FunctionType::MEMCMP:
    return Stream << "FunctionType::MEMCMP";
  case FunctionType::BCMP:
    return Stream << "FunctionType::BCMP";
  case FunctionType::MEMSET:
    return Stream << "FunctionType::MEMSET";
  case FunctionType::BZERO:
    return Stream << "FunctionType::BZERO";
  }
}
template <typename T>
static raw_ostream &operator<<(raw_ostream &Stream,
                               const std::optional<T> &MaybeT) {
  if (MaybeT)
    return Stream << *MaybeT;
  return Stream << "std::nullopt";
}
static raw_ostream &operator<<(raw_ostream &Stream,
                               const FunctionDescriptor &FD) {
  return Stream << '{' << FD.Type << ',' << FD.Contiguous << ',' << FD.Overlap
                << ',' << FD.Loop << ',' << FD.AlignedLoop << ','
                << FD.Accelerator << ',' << FD.ElementClass << '}';
}
static raw_ostream &operator<<(raw_ostream &Stream,
                               const NamedFunctionDescriptor &NFD) {
  return Stream << '{' << '"' << NFD.Name << '"' << ',' << NFD.Desc << '}';
}
template <typename T>
static raw_ostream &operator<<(raw_ostream &Stream,
                               const std::vector<T> &VectorT) {
  Stream << '{';
  bool First = true;
  for (const auto &Obj : VectorT) {
    if (!First)
      Stream << ',';
    Stream << Obj;
    First = false;
  }
  return Stream << '}';
}

static void Serialize(raw_ostream &Stream,
                      ArrayRef<NamedFunctionDescriptor> Descriptors) {
  Stream << R"(ArrayRef<NamedFunctionDescriptor> getFunctionDescriptors() {
  static constexpr NamedFunctionDescriptor kDescriptors[] = {
)";
  for (size_t I = 0, E = Descriptors.size(); I < E; ++I) {
    Stream << kIndent << kIndent << Descriptors[I] << ",\n";
  }
  Stream << R"(  };
  return ArrayRef(kDescriptors);
}
)";
}

} // namespace descriptors

namespace configurations {

// This namespace generates the getXXXConfigurations functions:
// ------------------------------------------------------------
// e.g.
// llvm::ArrayRef<MemcpyConfiguration> getMemcpyConfigurations() {
//   using namespace __llvm_libc;
//   static constexpr MemcpyConfiguration kConfigurations[] = {
//     {Wrap<memcpy_0xE00E29EE73994E2B>, "memcpy_0xE00E29EE73994E2B"},
//     {Wrap<memcpy_0x8661D80472487AB5>, "memcpy_0x8661D80472487AB5"},
//     ...
//   };
//   return llvm::ArrayRef(kConfigurations);
// }

// The `Wrap` template function is provided in the `Main` function below.
// It is used to adapt the gnerated code to the prototype of the C function.
// For instance, the generated code for a `memcpy` takes `char*` pointers and
// returns nothing but the original C `memcpy` function take and returns `void*`
// pointers.

struct FunctionName {
  FunctionType ForType;
};

struct ReturnType {
  FunctionType ForType;
};

struct Configuration {
  FunctionName Name;
  ReturnType Type;
  std::vector<const NamedFunctionDescriptor *> Descriptors;
};

static raw_ostream &operator<<(raw_ostream &Stream, const FunctionName &FN) {
  switch (FN.ForType) {
  case FunctionType::MEMCPY:
    return Stream << "getMemcpyConfigurations";
  case FunctionType::MEMCMP:
    return Stream << "getMemcmpConfigurations";
  case FunctionType::BCMP:
    return Stream << "getBcmpConfigurations";
  case FunctionType::MEMSET:
    return Stream << "getMemsetConfigurations";
  case FunctionType::BZERO:
    return Stream << "getBzeroConfigurations";
  }
}

static raw_ostream &operator<<(raw_ostream &Stream, const ReturnType &RT) {
  switch (RT.ForType) {
  case FunctionType::MEMCPY:
    return Stream << "MemcpyConfiguration";
  case FunctionType::MEMCMP:
  case FunctionType::BCMP:
    return Stream << "MemcmpOrBcmpConfiguration";
  case FunctionType::MEMSET:
    return Stream << "MemsetConfiguration";
  case FunctionType::BZERO:
    return Stream << "BzeroConfiguration";
  }
}

static raw_ostream &operator<<(raw_ostream &Stream,
                               const NamedFunctionDescriptor *FD) {
  return Stream << formatv("{Wrap<{0}>, \"{0}\"}", FD->Name);
}

static raw_ostream &
operator<<(raw_ostream &Stream,
           const std::vector<const NamedFunctionDescriptor *> &Descriptors) {
  for (size_t I = 0, E = Descriptors.size(); I < E; ++I)
    Stream << kIndent << kIndent << Descriptors[I] << ",\n";
  return Stream;
}

static raw_ostream &operator<<(raw_ostream &Stream, const Configuration &C) {
  Stream << "llvm::ArrayRef<" << C.Type << "> " << C.Name << "() {\n";
  if (C.Descriptors.empty())
    Stream << kIndent << "return {};\n";
  else {
    Stream << kIndent << "using namespace __llvm_libc;\n";
    Stream << kIndent << "static constexpr " << C.Type
           << " kConfigurations[] = {\n";
    Stream << C.Descriptors;
    Stream << kIndent << "};\n";
    Stream << kIndent << "return llvm::ArrayRef(kConfigurations);\n";
  }
  Stream << "}\n";
  return Stream;
}

static void Serialize(raw_ostream &Stream, FunctionType FT,
                      ArrayRef<NamedFunctionDescriptor> Descriptors) {
  Configuration Conf;
  Conf.Name = {FT};
  Conf.Type = {FT};
  for (const auto &FD : Descriptors)
    if (FD.Desc.Type == FT)
      Conf.Descriptors.push_back(&FD);
  Stream << Conf;
}

} // namespace configurations
static void Serialize(raw_ostream &Stream,
                      ArrayRef<NamedFunctionDescriptor> Descriptors) {
  Stream << "// This file is auto-generated by libc/benchmarks/automemcpy.\n";
  Stream << "// Functions : " << Descriptors.size() << "\n";
  Stream << "\n";
  Stream << "#include \"LibcFunctionPrototypes.h\"\n";
  Stream << "#include \"automemcpy/FunctionDescriptor.h\"\n";
  Stream << "#include \"src/string/memory_utils/elements.h\"\n";
  Stream << "\n";
  Stream << "using llvm::libc_benchmarks::BzeroConfiguration;\n";
  Stream << "using llvm::libc_benchmarks::MemcmpOrBcmpConfiguration;\n";
  Stream << "using llvm::libc_benchmarks::MemcpyConfiguration;\n";
  Stream << "using llvm::libc_benchmarks::MemmoveConfiguration;\n";
  Stream << "using llvm::libc_benchmarks::MemsetConfiguration;\n";
  Stream << "\n";
  Stream << "namespace __llvm_libc {\n";
  Stream << "\n";
  codegen::functions::Serialize(Stream, Descriptors);
  Stream << "\n";
  Stream << "} // namespace __llvm_libc\n";
  Stream << "\n";
  Stream << "namespace llvm {\n";
  Stream << "namespace automemcpy {\n";
  Stream << "\n";
  codegen::descriptors::Serialize(Stream, Descriptors);
  Stream << "\n";
  Stream << "} // namespace automemcpy\n";
  Stream << "} // namespace llvm\n";
  Stream << "\n";
  Stream << R"(
using MemcpyStub = void (*)(char *__restrict, const char *__restrict, size_t);
template <MemcpyStub Foo>
void *Wrap(void *__restrict dst, const void *__restrict src, size_t size) {
  Foo(reinterpret_cast<char *__restrict>(dst),
      reinterpret_cast<const char *__restrict>(src), size);
  return dst;
}
)";
  codegen::configurations::Serialize(Stream, FunctionType::MEMCPY, Descriptors);
  Stream << R"(
using MemcmpStub = int (*)(const char *, const char *, size_t);
template <MemcmpStub Foo>
int Wrap(const void *lhs, const void *rhs, size_t size) {
  return Foo(reinterpret_cast<const char *>(lhs),
             reinterpret_cast<const char *>(rhs), size);
}
)";
  codegen::configurations::Serialize(Stream, FunctionType::MEMCMP, Descriptors);
  codegen::configurations::Serialize(Stream, FunctionType::BCMP, Descriptors);
  Stream << R"(
using MemsetStub = void (*)(char *, int, size_t);
template <MemsetStub Foo> void *Wrap(void *dst, int value, size_t size) {
  Foo(reinterpret_cast<char *>(dst), value, size);
  return dst;
}
)";
  codegen::configurations::Serialize(Stream, FunctionType::MEMSET, Descriptors);
  Stream << R"(
using BzeroStub = void (*)(char *, size_t);
template <BzeroStub Foo> void Wrap(void *dst, size_t size) {
  Foo(reinterpret_cast<char *>(dst), size);
}
)";
  codegen::configurations::Serialize(Stream, FunctionType::BZERO, Descriptors);
  Stream << R"(
llvm::ArrayRef<MemmoveConfiguration> getMemmoveConfigurations() {
  return {};
}
)";
  Stream << "// Functions : " << Descriptors.size() << "\n";
}

} // namespace codegen

// Stores `VolatileStr` into a cache and returns a StringRef of the cached
// version.
StringRef getInternalizedString(std::string VolatileStr) {
  static llvm::StringSet StringCache;
  return StringCache.insert(std::move(VolatileStr)).first->getKey();
}

static StringRef getString(FunctionType FT) {
  switch (FT) {
  case FunctionType::MEMCPY:
    return "memcpy";
  case FunctionType::MEMCMP:
    return "memcmp";
  case FunctionType::BCMP:
    return "bcmp";
  case FunctionType::MEMSET:
    return "memset";
  case FunctionType::BZERO:
    return "bzero";
  }
}

void Serialize(raw_ostream &Stream, ArrayRef<FunctionDescriptor> Descriptors) {
  std::vector<NamedFunctionDescriptor> FunctionDescriptors;
  FunctionDescriptors.reserve(Descriptors.size());
  for (auto &FD : Descriptors) {
    FunctionDescriptors.emplace_back();
    FunctionDescriptors.back().Name = getInternalizedString(
        formatv("{0}_{1:X16}", getString(FD.Type), FD.id()));
    FunctionDescriptors.back().Desc = std::move(FD);
  }
  // Sort functions so they are easier to spot in the generated C++ file.
  std::sort(FunctionDescriptors.begin(), FunctionDescriptors.end(),
            [](const NamedFunctionDescriptor &A,
               const NamedFunctionDescriptor &B) { return A.Desc < B.Desc; });
  codegen::Serialize(Stream, FunctionDescriptors);
}

} // namespace automemcpy
} // namespace llvm
