=======
ClangIR
=======

.. warning::
    The project of upstreaming ClangIR support from the incubator repository is
    still in progress, and ClangIR is not included in a default clang build. The
    documentation may be incomplete and out-of-date.

ClangIR is a high-level representation in Clang that reflects aspects of the
C/C++ languages and their extensions. It is implemented using MLIR and occupies
a position between Clang's AST and LLVM IR.

ClangIR Design Documents
========================

.. toctree::
    :numbered:
    :maxdepth: 1

    ABILowering
    CleanupAndEHDesign
    CodeDuplication
