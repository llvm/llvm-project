# ClangIR

ClangIR (CIR) is a high-level representation in Clang that reflects aspects
of the C/C++ languages and their extensions. It is implemented using MLIR and
occupies a position between Clang's AST and LLVM IR.

Follow and contribute to upstream development through the
[`ClangIR`-labeled issues and pull
requests](https://github.com/llvm/llvm-project/labels/ClangIR) on
llvm-project. See the [ClangIR Design
Documents](#clangir-design-documents) for in-depth coverage of CIR's design.

## Using ClangIR

ClangIR is an MLIR dialect tightly integrated with Clang: it lives in the
Clang project rather than as a standalone in-tree dialect.

Pass `-fclangir` to the Clang driver to compile through CIR (AST to CIR to
CIR passes to LLVM IR and the backend). In order to inspect CIR output, use
`-emit-cir`, the equivalent of `-emit-llvm` for CIR.

:::{note}
ClangIR can be disabled from the build of Clang via the `-DCLANG_ENABLE_CIR=OFF`,
see [CMake](https://llvm.org/docs/CMake.html) flag. Using this flag can reduce the compile
time and binary size of Clang, however it will make the `-fclangir` flag unavaiable.
:::

See this [Compiler Explorer
example](https://godbolt.org/z/dTGqMGx54), which shows the flow of compilation
through CIR, from higher-level CIR constructs to lower-level ones, the
pipeline viewer is convenient for following along.

## ClangIR Design Documents

```{toctree}
:maxdepth: 1
:numbered:

ABILowering
CleanupAndEHDesign
CodeDuplication
```

## RFCs

- September 2026: [RFC: Enable ClangIR Build By
  Default](https://discourse.llvm.org/t/rfc-enable-clangir-build-by-default/91730)
- June 2026: [\[RFC\]\[ClangIR\] Making CIR Pipeline Boundaries First-Class
  Driver
  Artifacts](https://discourse.llvm.org/t/rfc-clangir-making-cir-pipeline-boundaries-first-class-driver-artifacts/90998)
- January 2024: [RFC: Upstreaming
  ClangIR](https://discourse.llvm.org/t/rfc-upstreaming-clangir/76587)
- June 2022: [RFC: An MLIR based Clang IR
  (CIR)](https://discourse.llvm.org/t/rfc-an-mlir-based-clang-ir-cir/63319)

## Talks and posts

- Dec 2025: [GSoC 2025: ClangIR
  upstreaming](https://blog.llvm.org/posts/2025-gsoc-clangir-upstreaming/)
- Oct 2025: US LLVM Developers Meeting (keynote). *ClangIR: Upstreaming an
  Incubator Project*.
  [video](https://www.youtube.com/watch?v=mQXE_y1T6H0),
  [pdf](https://llvm.github.io/clangir/Files/2025_ClangIR_UpstreamingIncubator.pdf)
- Oct 2025: [A High-Level Overview of Address Spaces: Their Place in ClangIR
  and
  LLVM](https://dev.to/david_rivera_8d845b35931e/a-high-level-overview-of-address-spaces-their-place-in-clangir-5e0k)
- Jun 2025: Asia LLVM. *ClangIR's Footprint: A quick compile-time impact
  report*. [video](https://www.youtube.com/watch?v=Dh_RObp5SUE),
  [pdf](https://llvm.github.io/clangir/Files/asiallvm-brunolopes-25-cir-compile-time.pdf)
- Mar 2025: Cambridge Computer Science [Group
  Projects](https://group-projects.cst.cam.ac.uk/#__tabbed_1_2). Award on
  technical achievement. *[CUDA Support for
  ClangIR](https://wiki.cam.ac.uk/cl-design-projects/CUDA_Support_for_ClangIR)*.
  [video](https://www.youtube.com/watch?v=CO1w_L3yIbQ)
- Nov 2024: Supercomputing Workshop. *[Introducing ClangIR, High-Level IR for
  the C/C++ Family of
  Languages](https://sc24.supercomputing.org/proceedings/workshops/workshop_pages/misc312.html)*.
  [pdf](https://llvm.github.io/clangir/Files/sc24-clangir.pdf)
- Sep 2024: [GSoC 2024: Compile GPU kernels using
  ClangIR](https://blog.llvm.org/posts/2024-08-29-gsoc-opencl-c-support-for-clangir/)
- Sep 2024: [GSoC 2024: ABI Lowering in
  ClangIR](https://blog.llvm.org/posts/2024-09-07-abi-lowering-in-clangir/)
- Oct 2023: US LLVM Developers Meeting. *Evolution of ClangIR: A Year of
  Progress, Challenges, and Future Plans*.
  [video](https://www.youtube.com/watch?v=XNOPO3ogdfQ),
  [pdf](http://brunocardoso.cc/resources/2023-LLVMDevMtgClangIR.pdf)

## History

ClangIR [started](https://github.com/facebookarchive/clangir) in October 2021.
In June 2022, an
[RFC](https://discourse.llvm.org/t/rfc-an-mlir-based-clang-ir-cir/63319)
proposing to incorporate the project — then in very early stages — into
llvm-project led to ClangIR becoming part of the LLVM incubator, where
development happened for the following years. In January 2024, the
[upstreaming
RFC](https://discourse.llvm.org/t/rfc-upstreaming-clangir/76587) was accepted
as the graduation step from the [incubator](https://github.com/llvm/clangir)
(now archived), and ClangIR has been developed directly upstream ever since,
superseding the incubator in most functionality.
