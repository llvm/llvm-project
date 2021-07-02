# Building and Testing LLVM libc on Windows

## Setting Up Environment

To build LLVM libc on Windows, first build Clang using the following steps.

1. Open Command Prompt in Windows
2. Set TEMP and TMP to a directory. Creating this path is necessary for a
   successful clang build.
    1. Create tmp under your preferred directory or under `C:\src`:

        ```
        cd C:\src
        mkdir tmp
        ```

    2. In the start menu, search for "environment variables for your account".
       Set TEMP and TMP to `C:\src\tmp` or the corresponding path elsewhere.
3. Download [Visual Studio Community](https://visualstudio.microsoft.com/downloads/).
4. Install [CMake](https://cmake.org/download/) and
   [Ninja](https://github.com/ninja-build/ninja/releases). (Optional, included
   in Visual Studio).
5. Load the Visual Studio environment variables using this command. This is
   crucial as it allows you to use build tools like CMake and Ninja:

    ```
    "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64
    ```

    Note: **Rerun this command every time you open a new Command Prompt
    window.**

6. If you have not used Git before, install
   [Git](https://git-scm.com/download/win) for Windows. Check out the LLVM
   source tree from Github using:

    ```
    git clone https://github.com/llvm/llvm-project.git
    ```

7. Ensure you have access to Clang, either by downloading from
   [LLVM Download](https://releases.llvm.org/download.html) or
   [building it yourself](https://clang.llvm.org/get_started.html).

## Building LLVM libc

In this section, Clang will be used to compile LLVM
libc, and finally, build and test the libc.

8. Create a empty build directory in `C:\src` or your preferred directory and
    cd to it using:

    ```
    mkdir libc-build
    cd libc-build
    ```

9. Run the following CMake command to generate build files. LLVM libc must be built
   by Clang, so ensure Clang is specified as the C and C++ compiler.

    ```
    cmake -G Ninja ../llvm-project/llvm -DCMAKE_C_COMPILER=C:/src/clang-build/bin/clang-cl.exe -DCMAKE_CXX_COMPILER=C:/src/clang-build/bin/clang-cl.exe  -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_FORCE_BUILD_RUNTIME=libc -DLLVM_ENABLE_PROJECTS=libc -DLLVM_NATIVE_ARCH=x86_64 -DLLVM_HOST_TRIPLE=x86_64-window-x86-gnu
    ```

10. Build LLVM libc using:

    ```
    ninja llvmlibc

    ```

11. Run tests using:

    ```
    ninja checklibc
    ```
