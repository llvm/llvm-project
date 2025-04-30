.. _flang-glossary:

**************
Flang glossary
**************

+------+--------------------------------------------------------------------------------------------+
| Name | Explanation                                                                                |
+======+============================================================================================+
| ADT  | Abstract Data Types.                                                                       |
+------+--------------------------------------------------------------------------------------------+
| ASD  | Array Subscript Descriptor - holds the number of dimensions for an array, see ``ast.h.``   |
+------+--------------------------------------------------------------------------------------------+
| AST  | Abstract Syntax Tree.                                                                      |
+------+--------------------------------------------------------------------------------------------+
| BE   | short for "backend" (used heavily in code and comments)                                    |
+------+--------------------------------------------------------------------------------------------+
| FE   | short for "frontend" (used heavily in code and comments)                                   |
+------+--------------------------------------------------------------------------------------------+
| ILM  | Intermediate Language Macros.                                                              |
|      | Representation of executable statements.                                                   |
|      | This is an output of flang1 and input for flang2.                                          |
+------+--------------------------------------------------------------------------------------------+
| ILI  | Intermediate Language Instructions.                                                        |
|      | The IR used by flang2 for optimisations.                                                   |
+------+--------------------------------------------------------------------------------------------+
| ILT  | Terminal node of an ILI statement which corresponds to a source language statement.        |
|      | A sequence of ILTs represent a block. ILTs have links to next and previous.                |
+------+--------------------------------------------------------------------------------------------+
| IPA  | InterProcedural Analysis https://en.wikipedia.org/wiki/Interprocedural_optimization        |
+------+--------------------------------------------------------------------------------------------+
| IR   | Intermediate Representation. A general term which may refer to many representations.       |
+------+--------------------------------------------------------------------------------------------+
| PGI  | Older version of Fortran compiler, which Flang front-end bases on.                         |
|      | PGI Compilers & Tools have evolved into the NVIDIA HPC SDK.                                |
+------+--------------------------------------------------------------------------------------------+
| SD   | Static Descriptor (used with pointers e.g. ``ptr$sd``)                                     |
+------+--------------------------------------------------------------------------------------------+
| STB  | Symbol TaBle, created symbol after symbol as identified by the parser                      |
+------+--------------------------------------------------------------------------------------------+
| SHD  | SHape Descriptor - holds the lower and upper bound of an array and the stride.             |
+------+--------------------------------------------------------------------------------------------+
| STD  | STatement Descriptor - a larger, more generic structure describing the parsed code elements|
|      | See ``ast.h`` for full members list.                                                       |
+------+--------------------------------------------------------------------------------------------+
| TBAA | Type Based Alias Analysis                                                                  |
+------+--------------------------------------------------------------------------------------------+
| TPV  | Target Processor Value (used in ``semant.c``)                                              |
+------+--------------------------------------------------------------------------------------------+
