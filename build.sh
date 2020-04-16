#!/bin/bash

if [ ! -d build ]; then
    mkdir build
fi

pushd build
cmake -G "Unix Makefiles" \
    -DCMAKE_INSTALL_PREFIX=/opt/aqnote/install/llvm/6.x \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=On \
    -DCLANG_DEFAULT_OPENMP_RUNTIME=libgomp \
    -DLLVM_ENABLE_PROJECTS=clang \
    ../llvm
popd

# make VERBOSE=1 -j6 -C build

exit 0
