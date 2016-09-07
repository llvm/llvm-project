This directory contains a VSPackage project to generate a Visual Studio extension
for clang-tidy.

Build prerequisites are:
- Visual Studio 2013 Professional
- Visual Studio 2013 SDK
- Visual Studio 2010 Professional (?)
- Visual Studio 2010 SDK (?)

The extension is built using CMake by setting BUILD_CLANG_TIDY_VS_PLUGIN=ON
when configuring a Clang build, and building the clang_tidy_vsix target.

The CMake build will copy clang-tidy.exe and LICENSE.TXT into the ClangTidy/
directory so they can be bundled with the plug-in, as well as creating
ClangTidy/source.extension.vsixmanifest. Once the plug-in has been built with
CMake once, it can be built manually from the ClangTidy.sln solution in Visual
Studio.
