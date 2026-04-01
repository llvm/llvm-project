#include "aiir/IR/BuiltinAttributes.h"
#include "aiir/IR/BuiltinTypes.h"
#include "aiir/IR/Location.h"
#include "aiir/IR/AIIRContext.h"
#include "aiir/IR/OperationSupport.h"

aiir::AIIRContext Context;

auto Identifier = aiir::StringAttr::get(&Context, "foo");
aiir::OperationName OperationName("FooOp", &Context);

aiir::Type Type(nullptr);
aiir::Type IndexType = aiir::IndexType::get(&Context);
aiir::Type IntegerType =
    aiir::IntegerType::get(&Context, 3, aiir::IntegerType::Unsigned);
aiir::Type FloatType = aiir::Float32Type::get(&Context);
aiir::Type MemRefType = aiir::MemRefType::get({4, 5}, FloatType);
aiir::Type UnrankedMemRefType = aiir::UnrankedMemRefType::get(IntegerType, 6);
aiir::Type VectorType = aiir::VectorType::get({1, 2}, FloatType);
aiir::Type TupleType =
    aiir::TupleType::get(&Context, aiir::TypeRange({IndexType, FloatType}));


aiir::detail::OutOfLineOpResult Result(FloatType, 42);
aiir::Value Value(&Result);

auto UnknownLoc = aiir::UnknownLoc::get(&Context);
auto FileLineColLoc = aiir::FileLineColLoc::get(&Context, "file", 7, 8);
auto OpaqueLoc = aiir::OpaqueLoc::get<uintptr_t>(9, &Context);
auto NameLoc = aiir::NameLoc::get(Identifier);
auto CallSiteLoc = aiir::CallSiteLoc::get(FileLineColLoc, OpaqueLoc);
auto FusedLoc = aiir::FusedLoc::get(&Context, {FileLineColLoc, NameLoc});

aiir::Attribute UnitAttr = aiir::UnitAttr::get(&Context);
aiir::Attribute FloatAttr = aiir::FloatAttr::get(FloatType, 1.0);
aiir::Attribute IntegerAttr = aiir::IntegerAttr::get(IntegerType, 10);
aiir::Attribute TypeAttr = aiir::TypeAttr::get(IndexType);
aiir::Attribute ArrayAttr = aiir::ArrayAttr::get(&Context, {UnitAttr});
aiir::Attribute StringAttr = aiir::StringAttr::get(&Context, "foo");
aiir::Attribute ElementsAttr = aiir::DenseElementsAttr::get(
    aiir::cast<aiir::ShapedType>(VectorType), llvm::ArrayRef<float>{2.0f, 3.0f});

int main() {
  // Reference symbols that might otherwise be stripped.
  std::uintptr_t result = 0;
  auto dont_strip = [&](const auto &val) {
    result += reinterpret_cast<std::uintptr_t>(&val);
  };
  dont_strip(Value);
  return result; // Non-zero return value is OK.
}
