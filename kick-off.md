# configuration
> cmake -S llvm -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug

LLVM的工程很大，源码的源头一般认为是 llvm 文件夹，可以看到这里也是从其开始寻找 cmake 文件的。
当前要求配置时必须制定 build 类型。

# build
> cmake --build build
