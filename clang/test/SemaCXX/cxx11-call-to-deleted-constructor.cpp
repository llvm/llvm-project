// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

namespace std {
template<class _Ep> class initializer_list { };
}

namespace cva {

template <class VT, unsigned int ROWS = 0, unsigned int COLS = 0>
class Matrix {
public:

    typedef VT value_type;
    inline __attribute__((always_inline)) value_type& at();
};

template <class VT, unsigned int SIZE> using Vector = Matrix<VT, SIZE, 1>;

template <class VT>
using RGBValue = Vector<VT, 3>;
using RGBFValue = RGBValue<float>;

template <class VT> class Matrix<VT, 0, 0> { // expected-note {{passing argument to parameter here}}
public:
    typedef VT value_type;
    Matrix(const unsigned int nRows, const unsigned int nColumns, const value_type* data = nullptr);

    Matrix(const std::initializer_list<value_type>& list) = delete; // expected-note {{'Matrix' has been explicitly marked deleted here}}

};

void getLaplacianClosedForm()
{
    Matrix<double> winI(0, 3);
    RGBFValue* inputPreL;
    winI = { inputPreL->at() }; // expected-error {{call to deleted constructor of 'Matrix<double> &&'}}
}

}
