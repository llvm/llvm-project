# Why do we make this project?
We want to use clang/llvm as comfortable as msvc.

# How to compile

```batch
X86：clang+lld+debug
    
mkdir build-debug-64
pushd build-debug-64
cmake .. -G "Visual Studio 17 2022" -A X64 -DLLVM_ENABLE_PROJECTS="clang;lld" -DCMAKE_INSTALL_PREFIX=E:\llvm\install-debug-64 -DLLVM_TARGETS_TO_BUILD=X86 -DCMAKE_BUILD_TYPE=Debug -DHAVE_STD_IS_TRIVIALLY_COPYABLE=0 -DLLVM_TOOL_LLVM_SHLIB_BUILD=off ../llvm
msbuild /m -p:Configuration=Debug INSTALL.vcxproj

X86：clang+lld+RelWithDebInfo

mkdir build-RelWithDebInfo-64
pushd build-RelWithDebInfo-64
cmake .. -G "Visual Studio 17 2022" -A X64 -DLLVM_ENABLE_PROJECTS="clang;lld" -DCMAKE_INSTALL_PREFIX=E:\llvm\install-RelWithDebInfo-64 -DLLVM_ENABLE_LIBXML2=ON -DLLVM_TARGETS_TO_BUILD=X86 -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLLVM_USE_CRT_RELEASE=MT -DHAVE_STD_IS_TRIVIALLY_COPYABLE=0 -DLLVM_TOOL_LLVM_SHLIB_BUILD=off ../llvm
msbuild /m -p:Configuration=RelWithDebInfo INSTALL.vcxproj 

X86：clang+lld+release

mkdir build-release-64
pushd build-release-64
cmake .. -G "Visual Studio 17 2022" -A X64 -DLLVM_ENABLE_PROJECTS="clang;lld" -DCMAKE_INSTALL_PREFIX=E:\llvm\install-release-64 -DLLVM_ENABLE_LIBXML2=ON -DLLVM_TARGETS_TO_BUILD=X86 -DCMAKE_BUILD_TYPE=release -DLLVM_USE_CRT_RELEASE=MT -DHAVE_STD_IS_TRIVIALLY_COPYABLE=0 -DLLVM_TOOL_LLVM_SHLIB_BUILD=off ../llvm
msbuild /m -p:Configuration=release INSTALL.vcxproj 
```
