#include "llvm/ADT/ArrayRef.h"

int Array[] = {1, 2, 3};

llvm::ArrayRef<int> ArrayRef(Array);
llvm::MutableArrayRef<int> MutableArrayRef(Array);

int main() { return 0; }
