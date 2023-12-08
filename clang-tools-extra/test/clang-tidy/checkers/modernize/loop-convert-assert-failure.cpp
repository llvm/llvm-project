// RUN: not clang-tidy %s -checks=-*,modernize-loop-convert --

// Note: this test expects no assert failure happened in clang-tidy.

class LinguisticItem {
  LinguisticItem *x0;
  class x1 {
    bool operator!= ( const x1 &;
    operator* ( ;
    LinguisticItem * &operator-> ( ;
    operator++ (
  } begin() const;
  x1 end() const {
    LinguisticStream x2;
    for (x1 x3 = x2.begin x3 != x2.end; ++x3)
      x3->x0
  }
};
