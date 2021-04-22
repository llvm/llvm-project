
cmake:
	rm -rf build
	mkdir -p build
	cmake -B build --target lsif-clang -S . \
		  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
		  -DPATH_TO_LLVM=${PATH_TO_LLVM} \
		  -DSG_LLVM_VERSION=11
