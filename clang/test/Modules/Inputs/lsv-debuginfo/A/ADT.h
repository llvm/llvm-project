#ifndef ADT
#define ADT

#ifdef WITH_NAMESPACE
namespace llvm {
#endif
template <unsigned Alignment, unsigned Size>
struct AlignedCharArray {
  alignas(Alignment) char buffer[Size];
};

template <typename T1>
class AlignerImpl {
  T1 t1;
};

template <typename T1>
union SizerImpl {
  char arr1[sizeof(T1)];
};

template <typename T1>
struct AlignedCharArrayUnion
    : AlignedCharArray<alignof(AlignerImpl<T1>), sizeof(SizerImpl<T1>)> {};

template <typename T, unsigned N>
struct SmallVectorStorage {
  AlignedCharArrayUnion<T> InlineElts[N];
};
template <typename T, unsigned N>
class SmallVector : SmallVectorStorage<T, N> {};

template <typename T>
struct OptionalStorage {
  AlignedCharArrayUnion<T> storage;
};
template <typename T>
class Optional {
  OptionalStorage<T> Storage;
};

#ifdef WITH_NAMESPACE
} // namespace llvm
#endif
#endif
