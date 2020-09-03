#include "mlir/IR/Identifier.h"
#include "mlir/IR/MLIRContext.h"

mlir::MLIRContext Context;

auto Identifier = mlir::Identifier::get("foo", &Context);

int main() { return 0; }
