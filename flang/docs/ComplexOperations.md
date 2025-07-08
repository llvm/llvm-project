# Complex Operations

```{contents}
---
local:
---
```

Fortran includes support for complex number types and a set of operators and
intrinsics that work on these types. Some of those operations are complicated
and require runtime function calls to implement.

This document outlines a design for generating these operations using the MLIR
complex dialect while avoiding cross-platform ABI issues.

## FIR Representation

MLIR contains a complex dialect, similar to the Math dialect also used for
lowering some integer and floating point operations in Flang. Conversion between
fir.complex types and MLIR complex types is supported.

As a result at the FIR level, complex operations can be represented as
conversions from the fir.complex type to the equivalent MLIR complex type, use
of the MLIR operation and a conversion back.

This is similar to the way the math intrinsics are lowered, as proposed [here][1]

**Fortran**
```fortran
function pow_self(c)
  complex, intent(in) :: c
  complex :: pow_self
  pow_self = c ** c
end function pow_self
```

**FIR**
```
func.func @_QPpow_self(%arg0: !fir.ref<!fir.complex<4>>) -> !fir.complex<4> {
    %0 = fir.alloca !fir.complex<4>
    %1 = fir.load %arg0 : !fir.ref<!fir.complex<4>>
    %2 = fir.load %arg0 : !fir.ref<!fir.complex<4>>
    %3 = fir.convert %1 : (!fir.complex<4>) -> complex<f32>
    %4 = fir.convert %2 : (!fir.complex<4>) -> complex<f32>
    %5 = complex.pow %3, %4 : complex<f32>
    %6 = fir.convert %5 : (complex<f32>) -> !fir.complex<4>
    fir.store %6 to %0 : !fir.ref<!fir.complex<4>>
    %7 = fir.load %0 : !fir.ref<!fir.complex<4>>
    return %7 : !fir.complex<4>
  }
```

Some operations are currently missing in the MLIR complex dialect that we would
want to use here, such as powi and the hyperbolic trigonometry functions.
For the missing operations we call directly to libm where possible, for powi
we provide an implementation in the flang runtime.

## Lowering

The MLIR complex dialect supports lowering either by emitting calls to the
complex functions in libm (ComplexToLibm), or through lowering to the standard
dialect (ComplexToStandard). However, as MLIR has no target awareness, the
lowering to libm functions suffers from ABI incompatibilities on some platforms.
As such the custom lowering to the standard dialect is used. This may be
something to revisit in future if performance could be improved by using the
libm functions.

Similarly to the numerical lowering through the math dialect, certain MLIR
optimisations could violate the precise floating point model, so when that is
requested lowering manually emits calls to libm, rather than going through the
MLIR complex dialect.

The ComplexToStandard dialect does still call into libm for some floating
point math operations, however these don't have the same ABI issues as the
complex libm functions.

The flang driver option `-fcomplex-arithmetic=` allows you to select whether
complex number division is lowered to function calls or to the `complex.div`
operation in the MLIR complex dialect. To avoid the ABI issues mentioned above,
the choice of function calls or the `complex.div` operation is made during the
lowering phase. The behavior of this option is as follows:

- `basic`: Lowers to `complex.div` and is converted to algebraic formulas. No
special handling to avoid overflow. NaN and infinite values are not handled.
- `improved`: Lowers to `complex.div` and is converted to Smith's algorithm. See
SMITH, R. L. Algorithm 116: Complex division. Commun. ACM 5, 8 (1962). This
value offers improved handling for overflow in intermediate calculations, but
overflow may occur. NaN and infinite values are handled.
- `full`: Lowers to a call to runtime library functions. Overflow and non-finite
values are handled by the library implementation. This is the default value.

While [the same option in clang][2] allows specifying `promoted`, this is not
implemented in Flang. Also, in the case of `improved`, clang does not handle NaN
and infinite values, but Flang does. These behavioral differences arise because
the transformation of complex division calculations depends on the implementation
of ComplexToStandard, which may change in the future.

[1]: https://discourse.llvm.org/t/rfc-change-lowering-of-fortran-math-intrinsics/63971
[2]: https://clang.llvm.org/docs/UsersManual.html#cmdoption-fcomplex-arithmetic
