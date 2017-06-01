#ifndef T_H
#define T_H

template <typename ValueType> struct VarStreamArray;

template <typename ValueType> struct VarStreamArrayIterator {
  VarStreamArrayIterator(VarStreamArray<ValueType>) {}
  bool HasError{};
};

#endif // T_H
