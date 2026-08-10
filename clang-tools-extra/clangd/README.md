## clangd

clangd is a language server, and provides C++ IDE features to editors.
This is not its documentation.

- the **website** is https://clangd.llvm.org/.
- the **bug tracker** is https://github.com/clangd/clangd/issues
- the **source code** is hosted at https://github.com/llvm/llvm-project/tree/main/clang-tools-extra/clangd.
- the **website source code** is at https://github.com/llvm/clangd-www/

### Communication channels

If you have any questions or feedback, you can reach community and developers
through one of these channels:

- chat: #clangd room hosted on [LLVM's Discord
  channel](https://discord.gg/xS7Z362).
- user questions and feature requests can be asked in the clangd topic on [LLVM
  Discussion Forums](https://llvm.discourse.group/c/llvm-project/clangd/34)

### Building and testing clangd

For a minimal setup on building clangd:
- Clone the LLVM repo to `$LLVM_ROOT`.
- Create a build directory, for example at `$LLVM_ROOT/build`.
- Inside the build directory run: `cmake $LLVM_ROOT/llvm/
  -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra"`.

  - We suggest building in `Release` mode as building DEBUG binaries requires
    considerably more resources. You can check
    [Building LLVM with CMake documentation](https://llvm.org/docs/CMake.html)
    for more details about cmake flags.
  - In addition to that using `Ninja` as a generator rather than default `make`
    is preferred. To do that consider passing `-G Ninja` to cmake invocation.
  - Finally, you can turn on assertions via `-DLLVM_ENABLE_ASSERTS=On`.

- Afterwards you can build clangd with `cmake --build $LLVM_ROOT/build --target
  clangd`, similarly run tests by changing target to `check-clangd`.
