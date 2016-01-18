This directory contains a VSPackage project to generate a Visual Studio extension
for clang-format.

Build prerequisites are:
- Visual Studio 2012 Professional
- Visual Studio 2010 Professional
- Visual Studio 2010 SDK.

The extension is built using CMake by setting BUILD_CLANG_FORMAT_VS_PLUGIN=ON
when configuring a Clang build, and building the clang_format_vsix target.

The CMake build will copy clang-format.exe and LICENSE.TXT into the ClangFormat/
directory so they can be bundled with the plug-in, as well as creating
ClangFormat/source.extension.vsixmanifest. Once the plug-in has been built with
CMake once, it can be built manually from the ClangFormat.sln solution in Visual
Studio.
