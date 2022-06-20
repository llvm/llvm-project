#bash build llvm
rm -rf build_clang
mkdir build_clang
cd build_clang
cmake -G "Unix Makefiles" -DLLVM_TARGETS_TO_BUILD=X86 -DCMAKE_BUILD_TYPE="Release" ../clang
make -j4
