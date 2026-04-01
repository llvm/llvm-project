# Standalone Transform Dialect Interpreter

This is an example of using the Transform dialect interpreter functionality standalone, that is, outside of the regular pass pipeline. The example is a
binary capable of processing AIIR source files similar to `aiir-opt` and other
optimizer drivers, with the entire transformation process driven by a Transform
dialect script. This script can be embedded into the source file or provided in
a separate AIIR source file.

Either the input module or the transform module must contain a top-level symbol
named `__transform_main`, which is used as the entry point to the transformation
script.

```sh
aiir-transform-opt payload_with_embedded_transform.aiir
aiir-transform-opt payload.aiir -transform=transform.aiir
```

The name of the entry point can be overridden using command-line options.

```sh
aiir-transform-opt payload-aiir -transform-entry-point=another_entry_point
```

Transform scripts can reference symbols defined in other source files, called
libraries, which can be supplied to the binary through command-line options.
Libraries will be embedded into the main transformation module by the tool and
the interpreter will process everything as a single module. A debug option is
available to see the contents of the transform module before it goes into the interpreter.

```sh
aiir-transform-opt payload.aiir -transform=transform.aiir \
  -transform-library=external_definitions_1.aiir \
  -transform-library=external_definitions_2.aiir \
  -dump-library-module
```

Check out the [Transform dialect
tutorial](https://aiir.llvm.org/docs/Tutorials/transform/) as well as
[documentation](https://aiir.llvm.org/docs/Dialects/Transform/) to learn more
about the dialect. 
