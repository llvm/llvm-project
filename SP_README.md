# Socialpoint LLVM Fork

This repository contains a llvm fork. It contains some additional clang-tidy checks inside the socialpoint group of checks that may be useful to us. It also contains a modified ``run-clang-tidy.py`` that allows to exclude file paths from the checks done using a ``compile_commands.json``

Currently we are interested only in 2 binaries generated from this repo: ``clang-tidy`` and ``clang-apply-replacements``

## How can I run the clang-tidy checks?

You need to build ``clang-tidy`` and ``clang-apply-replacements``. Don't mix the use of those binaries with others present in other distributions (like Xcode, Android NDK, LLVM standalone installation) because they may not work properly together

To apply the checks and automatically apply fixes you need a compile commands database (``compile_comands.json``). This file contains all the compiler invocations to build a project. We are currently generating it for Android projects. We are not generating it for iOS, but it should not be hard to do it

Use ``clang-tools-extra/clang-tidy/socialpoint/tool/run-clang-tidy.py`` to run checks on all the project

Example to run ``socialpoint-definitions-in-headers`` test:

``run-clang-tidy.py -clang-tidy-binary path_to_clang-tidy_binary -clang-apply-replacements-binary path_to_clang-apply-replacements_binary -p path_to_compile_commands_json -header-filter=.* -checks=-*,socialpoint-definitions-in-headers -fix .``

This runs the test ``socialpoint-definitions-in-headers`` in all files found in compile commands database. It tries to apply automatically all the found fixes

It is important to pass the paths to the ``clang-tidy`` and ``clang-apply-replacements`` binaries because otherwise the system may use others found in PATH

You can exclude source files from the checks by adding ``-exclude`` flag. This is useful to avoid modifying external library files included on hydra for instance adding ``-exclude="['*/hydra/lib/*']"`` to previous ``run-clang-tidy.py`` invocation would apply checks on all files but the ones containing the pattern ``*/hydra/lib/*``

## What tests do we currently have in socialpoint clang-tidy group?

| Setting | Use | Options | Value |
|---|:---:|:---:|:---:|
| socialpoint-definitions-in-headers | Extends the original https://clang.llvm.org/extra/clang-tidy/checks/misc-definitions-in-headers.html by allowing to fix also elements with internal linkage that may be problematic | IncludeInternalLinkage | 1: Allows to fix variables with internal linkage that original check was ignoring |
|  |  | HeaderFileExtensions | Allows to define header file extensions. Default is "h,hh,hpp,hxx" |
|  |  | UseHeaderFileExtension | If set uses HeaderFileExtensions. Default is 1 |
| socialpoint-sort-constructor-initializers | Looks for constructor initialization lists improperly sorted. This allows to fix warnings shown by -Wreorder flag automatically|  |  |

To pass options add ``-config`` argument to ``run-clang-tidy.py`` invocation

For example:

``-config="{'CheckOptions': [{key: socialpoint-definitions-in-headers.IncludeInternalLinkage, value: 1},{key: modernize-use-override.IgnoreDestructors, value: 0}]}"``

Sets ``IncludeInternalLinkage`` for ``socialpoint-definitions-in-headers`` check to 1 and ``IgnoreDestructors`` for ``modernize-use-override`` to 0

## How I can add more tests?

LLVM has a ``clang-tools-extra/clang-tidy/add_new_check.py`` script to create a placeholder test and update needed files. You can invoke from ``clang-tools-extra folder`` as

``add_new_check.py socialpoint my-fancy-check-name``

This would add needed changes to register a new test called ``my-fancy-check-name`` in the ``socialpoint`` group

I've found issues sometimes on windows when it tries to update some documentation files. You can revert the problematic file and update it manually

Here is additional info on how to proceed when implementing it and which tools are available https://clang.llvm.org/extra/clang-tidy/Contributing.html . Checking source code for other checks is very very helpful

Your main files for the test will be named ``clang-tools-extra/clang-tidy/socialpoint/MyFancyCheckName.h`` and ``clang-tools-extra/clang-tidy/socialpoint/MyFancyCheckName.cpp``

## How to compile all this stuff?

Following the steps from main README.md would be enough, but we are just interested in a couple of binaries so generating the project excluding some targets reduces quite a lot compile times. 
When reaching the part of generating the project with ``cmake -G <generator> [options] ../llvm`` you'll need to set theese variables to reduce projects generated and time it takes to compile. Depending on the generator it is possible you need to pass some additional cmake options to run properly

| Setting | Meaning |
|---|:---:|
| LLVM_ENABLE_PROJECTS="clang;clang-tools-extra;" | Generate only needed projects to build clang-tidy |
| LLVM_TARGETS_TO_BUILD="" | Do not create any clang architecture target |
| CMAKE_BUILD_TYPE=Release | Use release build type | 
| LLVM_INCLUDE_BENCHMARKS=FALSE | Do not generate benchmark targets |
| LLVM_INCLUDE_DOCS=FALSE | Do not generate documentation targets |
| LLVM_INCLUDE_EXAMPLES=FALSE | Do not include example targets | 
| LLVM_INCLUDE_GO_TESTS=FALSE | Do not include go tests targets |
| LLVM_INCLUDE_TESTS=FALSE | Do include tests targets |
| CLANG_INCLUDE_DOCS=FALSE | Do include clang documentation targets |
| CLANG_INCLUDE_TESTS=FALSE | Do include clang tests targets |
| CLANG_TOOLS_EXTRA_INCLUDE_DOCS=FALSE | Do include clang-tools-extra documentation targets |

Examples:

Visual Studio 2019 x64 compiler targetting x64 architecture:

``cmake -G "Visual Studio 16 2019" -A x64 -Thost=x64 -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;" -DLLVM_TARGETS_TO_BUILD="" -DCMAKE_BUILD_TYPE=Release -DLLVM_INCLUDE_BENCHMARKS=FALSE -DLLVM_INCLUDE_DOCS=FALSE -DLLVM_INCLUDE_EXAMPLES=FALSE -DLLVM_INCLUDE_GO_TESTS=FALSE -DLLVM_INCLUDE_TESTS=FALSE -DCLANG_INCLUDE_DOCS=FALSE -DCLANG_INCLUDE_TESTS=FALSE -DCLANG_TOOLS_EXTRA_INCLUDE_DOCS=FALSE ..\llvm``

Xcode:

``cmake -G "Xcode" -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;" -DLLVM_TARGETS_TO_BUILD="" -DCMAKE_BUILD_TYPE=Release -DLLVM_INCLUDE_BENCHMARKS=FALSE -DLLVM_INCLUDE_DOCS=FALSE -DLLVM_INCLUDE_EXAMPLES=FALSE -DLLVM_INCLUDE_GO_TESTS=FALSE -DLLVM_INCLUDE_TESTS=FALSE -DCLANG_INCLUDE_DOCS=FALSE -DCLANG_INCLUDE_TESTS=FALSE -DCLANG_TOOLS_EXTRA_INCLUDE_DOCS=FALSE ..\llvm``

After generating the project we are only interested in 2 targets: clang-tidy and clang-apply-replacements``

```
cmake --build . -t clang-tidy
cmake --build . -t clang-apply-replacements
```

Remeber you can set parallel build by adding ``-j number_of_workers`` to the ``cmake --build`` command
