This directory contains a VSPackage project to generate a Visual Studio extension
for clang-format.

Build prerequisites are:
- Visual Studio 2015
- Extensions SDK (you'll be prompted to install it if you open ClangFormat.sln)

The extension is built using CMake to generate the usual LLVM.sln by setting
the following CMake vars:

- BUILD_CLANG_FORMAT_VS_PLUGIN=ON

- NUGET_EXE_PATH=path/to/nuget_dir (unless nuget.exe is already available in PATH)

example:
  cd /d C:\code\llvm
  mkdir build & cd build
  cmake -DBUILD_CLANG_FORMAT_VS_PLUGIN=ON -DNUGET_EXE_PATH=C:\nuget ..

Once LLVM.sln is generated, build the clang_format_vsix target, which will build
ClangFormat.sln, the C# extension application.

The CMake build will copy clang-format.exe and LICENSE.TXT into the ClangFormat/
directory so they can be bundled with the plug-in, as well as creating
ClangFormat/source.extension.vsixmanifest. Once the plug-in has been built with
CMake once, it can be built manually from the ClangFormat.sln solution in Visual
Studio.
