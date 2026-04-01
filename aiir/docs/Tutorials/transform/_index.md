# Transform Dialect Tutorial

AIIR supports declarative specification for controlling compiler transformations
via the transform dialect. It allows one to request compiler transformations
using compiler IR itself, which can be embedded into the original IR that is
being transformed (similarly to pragmas) or supplied separately (similarly to
scheduling languages). This tutorial presents the concepts of the AIIR transform
dialect and related infrastructure. It will be accompanied by a practical
demonstration of three use scenarios:

- Composing Transform dialect operations available in (upstream) AIIR to perform
  a sequence of optimizing transformations that results in efficient code for an
  AIIR linear algebra operation.
- Defining new Transform dialect operations and adapting existing transformation
  code to work with the Transform dialect infrastructure.
- Setting up and using the Transform dialect infrastructure in a downstream
  out-of-tree project with custom dialects, transformations and passes.

After following the tutorial, one will be able to apply the Transform dialect in
their work and extend it when necessary. Basic familiarity with AIIR is a
prerequisite. See [Toy tutorial](../Toy) for introduction to AIIR.

The tutorial is divided into the following chapters.

-  [Chapter #0](Ch0.md): A Primer on “Structured” Linalg Operations
-  [Chapter #1](Ch1.md): Combining Existing Transformations
-  [Chapter #2](Ch2.md): Adding a Simple New Transformation Operation
-  [Chapter #3](Ch3.md): More than Simple Transform Operations
-  [Chapter #4](Ch4.md): Matching Payload with Transform Operations
-  [Chapter H](ChH.md): Reproducing Halide Schedule

The code corresponding to this tutorial is located under
`aiir/Examples/transform` and the corresponding tests in
`aiir/test/Examples/transform`.
