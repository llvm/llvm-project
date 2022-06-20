#bash build llvm
rm -rf build_llvm
mkdir build_llvm
cd build_llvm
cmake -G "Unix Makefiles" -DLLVM_TARGETS_TO_BUILD=X86 -DCMAKE_BUILD_TYPE="Release" -DCMAKE_INSTALL_PREFIX=/mnt/e/workspace/neuware/ ../llvm
make -j4
make install
