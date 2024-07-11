clang的入口文件路径为： `/home/chenshuang/llvm-project/build/tools/clang/tools/driver/clang-driver.cpp`

与传统方法是不一样的是，这里的入口文件是在`cmake`生成阶段根据模板文件创建的。这是因为`LLVM`中有很多子项目，是否将这些子项目纳入编译范围是由用户使用宏定义来决定的。

因此`LLVM`开发团队并没有直接就将`main`函数写出来，因为可能用不上。且子项目较多，`main`方法具有高度重复性，因此使用模板文件生成更有利于去除代码冗余。

clang main方法的内容为：
```C++
//===-- driver-template.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/LLVMDriver.h"

int clang_main(int argc, char **, const llvm::ToolContext &);

// clang-format off
// Cratels: clang的真正入口。这个文件是在cmake生成阶段自动生成的，因此不在源码目录中而在build目录中。
// 其他项目也采用了类似的做法，只是为你避免重复工作而已。
// clang-format on
int main(int argc, char **argv) {
  llvm::InitLLVM X(argc, argv);
  return clang_main(argc, argv, {argv[0], nullptr, false});
}
```
