#ifndef FORTRAN_LOWER_FIRMIXIN_H
#define FORTRAN_LOWER_FIRMIXIN_H

namespace Fortran::lower {

template <typename FirConverterT> class FirMixinBase {
public:
  FirConverterT *This() { return static_cast<FirConverterT *>(this); }
  const FirConverterT *This() const {
    return static_cast<const FirConverterT *>(this);
  }
};

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_FIRMIXIN_H
