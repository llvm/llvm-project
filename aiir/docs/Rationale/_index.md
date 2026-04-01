# Rationale

This section contains a collection of documents describing the motivation and
rationale for some of the design decisions behind AIIR.

[AIIR: Incremental Application to Graph Algorithms in ML Frameworks](AIIRForGraphAlgorithms.md)
:   A discussion of how the adoption of AIIR can be taken in incremental steps,
    with each step providing tangible benefits along the way. Refutes the idea
    that full adoption of AIIR is required before we can reap the benefits of
    AIIR.

[AIIR Rationale](Rationale.md)
:   Introduces the motivation for AIIR and captures design discussions and
    decisions made for various core features of AIIR.

[Generic DAG Rewriter Infrastructure Rationale](RationaleGenericDAGRewriter.md)
:   Details the rationale behind a general DAG-to-DAG rewrite infrastructure for
    AIIR.

[Linalg Dialect Rationale: The Case for Compiler-Friendly Custom Operations](RationaleLinalgDialect.md)
:   Describes the key design principles that led to the existing implementation
    of Linalg and lessons learned along the way.

[AIIR: The case for a simplified polyhedral form](RationaleSimplifiedPolyhedralForm.md)
:   An early design proposal exploring the tradeoffs of using a simplified form
    for polyhedral compiler techniques in AIIR instead of the traditional
    polyhedral schedule list form.

[Usage of 'const' in AIIR, for core IR types](UsageOfConst.md)
:   Explains the rationale for eschewing the use of `const` entirely for the
    core IR types in AIIR.
