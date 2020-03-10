Linker Script implementation notes and policy
=============================================

LLD implements a large subset of the GNU ld linker script notation. The LLD
implementation policy is to implement linker script features as they are
documented in the ld `manual <https://sourceware.org/binutils/docs/ld/Scripts.html>`_
We consider it a bug if the lld implementation does not agree with the manual
and it is not mentioned in the exceptions below.

The ld manual is not a complete specification, and is not sufficient to build
an implementation. In particular some features are only defined by the
implementation and have changed over time.

The lld implementation policy for properties of linker scripts that are not
defined by the documentation is to follow the GNU ld implementation wherever
possible. We reserve the right to make different implementation choices where
it is appropriate for LLD. Intentional deviations will be documented in this
file.
