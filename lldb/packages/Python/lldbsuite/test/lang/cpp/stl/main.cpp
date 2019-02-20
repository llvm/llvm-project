//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <cstdio>
#include <iostream>
#include <string>
#include <map>
int main (int argc, char const *argv[])
{
    std::string hello_world ("Hello World!");
    std::cout << hello_world << std::endl;
    std::cout << hello_world.length() << std::endl;
    std::cout << hello_world[11] << std::endl;

    std::map<std::string, int> associative_array;
    std::cout << "size of upon construction associative_array: " << associative_array.size() << std::endl;
    associative_array[hello_world] = 1;
    associative_array["hello"] = 2;
    associative_array["world"] = 3;

    std::cout << "size of associative_array: " << associative_array.size() << std::endl;
    printf("associative_array[\"hello\"]=%d\n", associative_array["hello"]);

    printf("before returning....\n"); // Set break point at this line.
}
