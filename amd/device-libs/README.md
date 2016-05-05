## OVERVIEW

ROCm OpenCL Library is in early development stages.

## BUILDING

This project requires reasonably recent LLVM/Clang build (April 2016 trunk). Testing also requires amdhsacod utility from ROCm Runtime.

Use out-of-source CMake build and create separate directory to run CMake.

The following build steps are performed:

    mkdir -p build
    cd build
    export LLVM_BUILD=... (path to LLVM build)
    CC=$LLVM_BUILD/bin/clang cmake -DLLVM_DIR=$LLVM_BUILD -DAMDHSACOD=$HSA_DIR/bin/x86_64/amdhsacod ..
    make
    make install
    make test

## TESTING

Currently all tests are offline:
 * OpenCL source is compiled to LLVM bitcode
 * Test bitcode is linked to library bitcode with llvm-link
 * Clang OpenCL compiler is run on resulting bitcode, producing code object.
 * Resulting code object is passed to llvm-objdump and amdhsacod -test.

The output of tests (which includes AMDGPU disassembly) can be displayed by running ctest -VV in build directory.
