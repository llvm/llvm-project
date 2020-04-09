
# FIR

This file should not be upstreamed to llvm-project.

## Monorepo now contains Flang!

### In-tree build

1. Get the stuff.

To understand the compilers handling of intrinsics,
see the [discussion of intrinsics](docs/Intrinsics.md).

To understand how a flang program communicates with libraries at runtime,
see the discussion of [runtime descriptors](docs/RuntimeDescriptor.md).

If you're interested in contributing to the compiler,
read the [style guide](docs/C++style.md)
and
also review [how flang uses modern C++ features](docs/C++17.md).

If you are interested in writing new documentation, follow 
[markdown style guide from LLVM](https://github.com/llvm/llvm-project/blob/main/llvm/docs/MarkdownQuickstartTemplate.md).

2. Get "on" the right branches.

```
  (cd f18-llvm-project ; git checkout fir-dev)
```

3. (not needed!)
             
4. Create a build space for cmake and make (or ninja)

```
  mkdir build
  cd build
  cmake ../f18-llvm-project/llvm -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_ENABLE_PROJECTS="flang;mlir" -DCMAKE_CXX_STANDARD=17 -DLLVM_BUILD_TOOLS=On -DLLVM_INSTALL_UTILS=On <other-arguments>
```

5. Build everything

```
  make
  make check-flang
  make install
```

### Out-of-tree build

Assuming someone was nice enough to build MLIR and LLVM libraries and
install them in a convenient place for you, then you may want to do a
standalone build.

1. Get the stuff is the same as above. Get the code from the same repos.

2. Get on the right branches. Again, same as above.

3. Create a build space for cmake and make (or ninja)

```
  mkdir build
  cd build
  export CC=<my-favorite-C-compiler>
  export CXX=<my-favorite-C++-compiler>
  cmake -GNinja ../f18-llvm-project/llvm -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_ENABLE_PROJECTS=mlir -DCMAKE_CXX_STANDARD=17 -DLLVM_BUILD_TOOLS=On -DLLVM_INSTALL_UTILS=On -DCMAKE_INSTALL_PREFIX=<install-llvm-here> <other-arguments>
```

5. Build and install

```
  ninja
  ninja install
```

6. Add the new installation to your PATH

```
  PATH=<install-llvm-here>/bin:$PATH
```

7. Create a build space for another round of cmake and make (or ninja)

```
  mkdir build-flang
  cd build-flang
  cmake -GNinja ../f18 -DLLVM_DIR=<install-llvm-here> -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLLVM_TARGETS_TO_BUILD=X86 -DCMAKE_CXX_STANDARD=17 -DLLVM_BUILD_TOOLS=On -DCMAKE_INSTALL_PREFIX=<install-flang-here> <other-arguments>
```
Note: if you plan on running lit regression tests, you should either:
- Use `-DLLVM_DIR=<build-llvm-here>` instead of `-DLLVM_DIR=<install-llvm-here>`
- Or, keep `-DLLVM_DIR=<install-llvm-here>` but add `-DLLVM_EXTERNAL_LIT=<path to llvm-lit>`.
A valid `llvm-lit` path is `<build-llvm-here>/bin/llvm-lit`.
Note that LLVM must also have been built with `-DLLVM_INSTALL_UTILS=On` so that tools required by tests like `FileCheck` are available in `<install-llvm-here>`.

8. Build and install

```
  ninja
  ninja check-flang
  ninja install
```

### Running regression tests

Inside `build` for in-tree builds or inside `build-flang` for out-of-tree builds:

```
  ninja check-flang
```

### Build The New Flang Driver
The new Flang driver, `flang-new`, is currently under active development and
should be considered as an experimental feature. For this reason it is disabled
by default. This will change once the new driver replaces the _throwaway_
driver, `flang`.

In order to build the new driver, add `-DFLANG_BUILD_NEW_DRIVER=ON` to your
CMake invocation line. Additionally, when building out-of-tree, use `CLANG_DIR`
(similarly to `LLVM_DIR` and `MLIR_DIR`) to find the installed Clang
components.

**Note:** `CLANG_DIR` is only required when building the new Flang driver,
which currently depends on Clang.

# How to Run Tests

Flang supports 2 different categories of tests
1. Regression tests (https://www.llvm.org/docs/TestingGuide.html#regression-tests)
2. Unit tests (https://www.llvm.org/docs/TestingGuide.html#unit-tests)

## For out of tree builds
To run all tests:
```
cd ~/flang/build
cmake -DLLVM_DIR=$LLVM -DMLIR_DIR=$MLIR ~/flang/src
make test check-all
```

To run individual regression tests llvm-lit needs to know the lit
configuration for flang. The parameters in charge of this are:
flang_site_config and flang_config. And they can be set as shown below:
```
<path-to-llvm-lit>/llvm-lit \
 --param flang_site_config=<path-to-flang-build>/test-lit/lit.site.cfg.py \
 --param flang_config=<path-to-flang-build>/test-lit/lit.cfg.py \
  <path-to-fortran-test>

```

Unit tests:

If flang was built with `-DFLANG_INCLUDE_TESTS=On` (`ON` by default), it is possible to generate unittests.
Note: Unit-tests will be skipped for LLVM install for an out-of-tree build as it does not include googletest related headers and libraries.

There are various ways to run unit-tests.

```

1. make check-flang-unit
2. make check-all or make check-flang
3. <path-to-llvm-lit>/llvm-lit \
        test/Unit
4. Invoking tests from <out-of-tree flang build>/unittests/<respective unit test folder>

```


## For in tree builds
If flang was built with `-DFLANG_INCLUDE_TESTS=On` (`On` by default), it is possible to
generate unittests.

To run all of the flang unit tests use the `check-flang-unit` target:
```
make check-flang-unit
```
To run all of the flang regression tests use the `check-flang` target:
```
make check-flang
```

# How to Generate Documentation

## Generate FIR Documentation
It is possible to
generate FIR language documentation by running `make flang-doc`. This will
create `docs/Dialect/FIRLangRef.md` in flang build directory.

## Generate Doxygen-based Documentation
To generate doxygen-style documentation from source code
- Pass `-DLLVM_ENABLE_DOXYGEN=ON -DFLANG_INCLUDE_DOCS=ON` to the cmake command.

```
cd ~/llvm-project/build
cmake -DLLVM_ENABLE_DOXYGEN=ON -DFLANG_INCLUDE_DOCS=ON ../llvm
make doxygen-flang
```

It will generate html in

```
    <build-dir>/tools/flang/docs/doxygen/html # for flang docs
```
## Generate Sphinx-based Documentation
<!TODO: Add webpage once we have a website.
!>
Flang documentation should preferably be written in `markdown(.md)` syntax (they can be in `reStructuredText(.rst)` format as well but markdown is recommended in first place), it
is mostly meant to be processed by the Sphinx documentation generation
system to create HTML pages which would be hosted on the webpage of flang and
updated periodically.

If you would like to generate and view the HTML locally:
- Install [Sphinx](http://sphinx-doc.org/), including the [sphinx-markdown-tables](https://pypi.org/project/sphinx-markdown-tables/) extension.
- Pass `-DLLVM_ENABLE_SPHINX=ON -DSPHINX_WARNINGS_AS_ERRORS=OFF` to the cmake command.

```
cd ~/llvm-project/build
cmake -DLLVM_ENABLE_SPHINX=ON -DSPHINX_WARNINGS_AS_ERRORS=OFF ../llvm
make docs-flang-html
```

It will generate html in

```
   $BROWSER <build-dir>/tools/flang/docs/html/
```
