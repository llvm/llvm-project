#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/ilist.h"
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

struct alignas(8) Z {};
llvm::PointerUnion<Z *, int *> PointerUnion(IntPtr);

// No members which instantiate PointerUnionUIntTraits<Z *> (e.g. get<T *>())
// are called, and this instance will therefore be raw-printed.
llvm::PointerUnion<Z *, float *> RawPrintingPointerUnion(nullptr);

using IlistTag = llvm::ilist_tag<struct A>;
using SimpleIlistTag = llvm::ilist_tag<struct B>;
struct IlistNode : llvm::ilist_node<IlistNode, IlistTag>,
                   llvm::ilist_node<IlistNode, SimpleIlistTag> {
  int Value;
};
auto Ilist = [] {
  llvm::ilist<IlistNode, IlistTag> Result;
  for (int I : {13, 14, 15}) {
    Result.push_back(new IlistNode);
    Result.back().Value = I;
  }
  return Result;
}();
auto SimpleIlist = []() {
  llvm::simple_ilist<IlistNode, SimpleIlistTag> Result;
  for (auto &Node : Ilist)
    Result.push_front(Node);
  return Result;
}();

int main() {
  // Reference symbols that might otherwise be stripped.
  ArrayRef[0];
  MutableArrayRef[0];
  !ExpectedValue;
  !ExpectedError;
  *OptionalValue;
  *OptionalNone;
  return 0;
}
