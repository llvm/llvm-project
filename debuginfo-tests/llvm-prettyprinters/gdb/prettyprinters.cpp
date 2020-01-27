#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

int Array[] = {1, 2, 3};
auto IntPtr = reinterpret_cast<int *>(0xabc);

llvm::ArrayRef<int> ArrayRef(Array);
llvm::MutableArrayRef<int> MutableArrayRef(Array);
llvm::DenseMap<int, int> DenseMap = {{4, 5}, {6, 7}};
llvm::Expected<int> ExpectedValue(8);
llvm::Expected<int> ExpectedError(llvm::createStringError({}, ""));
llvm::Optional<int> OptionalValue(9);
llvm::Optional<int> OptionalNone(llvm::None);
llvm::SmallVector<int, 5> SmallVector = {10, 11, 12};
llvm::SmallString<5> SmallString("foo");
llvm::StringRef StringRef = "bar";
llvm::Twine Twine = llvm::Twine(SmallString) + StringRef;
llvm::PointerIntPair<int *, 1> PointerIntPair(IntPtr, 1);
llvm::PointerUnion<float *, int *> PointerUnion(IntPtr);

int main() { return 0; }
