#include "automemcpy/CodeGen.h"
#include "automemcpy/RandomFunctionGenerator.h"
#include <unordered_set>

namespace llvm {
namespace automemcpy {

std::vector<FunctionDescriptor> generateFunctionDescriptors() {
  std::unordered_set<FunctionDescriptor, FunctionDescriptor::Hasher> Seen;
  std::vector<FunctionDescriptor> FunctionDescriptors;
  RandomFunctionGenerator P;
  while (Optional<FunctionDescriptor> MaybeFD = P.next()) {
    FunctionDescriptor FD = *MaybeFD;
    if (Seen.count(FD)) // FIXME: Z3 sometimes returns twice the same object.
      continue;
    Seen.insert(FD);
    FunctionDescriptors.push_back(std::move(FD));
  }
  return FunctionDescriptors;
}

} // namespace automemcpy
} // namespace llvm

int main(int, char **) {
  llvm::automemcpy::Serialize(llvm::outs(),
                              llvm::automemcpy::generateFunctionDescriptors());
}
