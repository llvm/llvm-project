template<class Element> struct TMatrixT;
typedef TMatrixT<double> TMatrixD;

void f(const TMatrixD &m);

template<class Element> struct TMatrixT {
  template <class Element2> TMatrixT(const TMatrixT<Element2> &);
  ~TMatrixT() {}
  void Determinant () { f(*this); }
};

template struct TMatrixT<float>;
template struct TMatrixT<double>;
