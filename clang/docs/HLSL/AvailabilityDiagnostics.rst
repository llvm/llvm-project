=============================
HLSL Availability Diagnostics
=============================

.. contents::
   :local:

Introduction
============

HLSL availability diagnostics emits errors or warning when unavailable shader APIs are used. Unavailable shader APIs are APIs that are exposed in HLSL code but are not available in the target shader stage or shader model version.

There are three modes of HLSL availability diagnostic:
1. **Default mode** - compiler emits an error when an unavailable shader API is found in a code that is reachable from the shader entry point function or from an exported library function (when compiling a shader library)
2. **Relaxed mode** - same as default mode except the compiler emits a warning. This mode is enabled by ``-Wno-error=hlsl-availability``.
3. **Strict mode** - compiler emits an error when when an unavailable API is found in parsed code regardless of whether it can be reached from the shader entry point or exported functions, or not. This mode is enabled by ``-fhlsl-strict-diagnostics``.

Implementation Details
======================

Environment Parameter
---------------------

In order to encode API availability based on the shader model version and shader model stage a new ``environment`` parameter was added to the existing Clang ``availability`` attribute. 

The values allowed for this parameter are a subset of values allowed as the ``llvm::Triple`` environment component. If the environment parameters is present, the declared availability attribute applies only to targets with the same platform and environment.

Default and Relaxed Diagnostic Modes
------------------------------------

This mode is implemeted in ``DiagnoseHLSLAvailability`` class in ``SemaHLSL.cpp`` and it is invoked after the whole translation unit is parsed (from ``Sema::ActOnEndOfTranslationUnit``). The implementation iterates over all shader entry points and exported library functions in the translation unit and performs an AST traversal of each function body.

When a reference to another function is found and it has a body, the AST of the referenced function is also scanned. This chain of AST traversals will reach all of the code that is reachable from the initial shader entry point or exported library function.

All shader APIs have an availability attribute that specifies the shader model version (and environment, if applicable) when this API was first introduced.When a reference to a function without a definition is found and it has an availability attribute, the version of the attribute is checked against the target shader model version and shader stage (if shader stage context is known), and an appropriate diagnostic is generated as needed.

All shader entry functions have ``HLSLShaderAttr`` attribute that specifies what type of shader this function represents. However, for exported library functions the target shader stage is unknown, so in this case the HLSL API availability will be only checked against the shader model version.

A list of functions that were already scanned is kept in order to avoid duplicate scans and diagnostics (see ``DiagnoseHLSLAvailability::ScannedDecls``). It might happen that a shader library has multiple shader entry points for different shader stages that all call into the same shared function. It is therefore important to record not just that a function has been scanned, but also in which shader stage context. This is done by using ``llvm::DenseMap`` that maps ``FunctionDecl *`` to a ``unsigned`` bitmap that represents a set of shader stages (or environments) the function has been scanned for. The ``N``'th bit in the set is set if the function has been scanned in shader environment whose ``HLSLShaderAttr::ShaderType`` integer value equals ``N``.

The emitted diagnostic messages belong to ``hlsl-availability`` diagnostic group and are reported as errors by default. With ``-Wno-error=hlsl-availability`` flang they become warning, making it relaxed HLSL diagnostics mode.

Strict Diagnostic Mode
----------------------

When strict HLSL availability diagnostic mode is enabled the compiler must report all HLSL API availability issues regardless of code reachability. The implementation of this mode takes advantage of an existing diagnostic scan in ``DiagnoseUnguardedAvailability`` class which is already traversing AST of each function as soon as the function body has been parsed.

If the compilation target is a shader library, only availability based on shader model version can be diagnosed during this scan. To diagnose availability based on shader stage, teh compiler will also run the AST traversals implementated in ``DiagnoseHLSLAvailability`` at the end of the translation unit as described in previous chapter.

As a result, availability based on specific shader stage will only be diagnosed in code that is reachable from a shader entry point or library export function. It also means that function bodies might be scanned multiple time. When that happens, care should be taken not to produce duplicated diagnostics.
