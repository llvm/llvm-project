//===-- main.cpp -------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <functional>
#include <stdlib.h>

template<typename ElemType>
ElemType* alloc(size_t count, std::function<ElemType(size_t)> get)
{
  ElemType *elems = new ElemType[count];
  for(size_t i = 0; i < count; i++)
    elems[i] = get(i);
  return elems;
}

int main (int argc, const char * argv[])
{
  int* data = alloc<int>(5, [] (size_t idx) -> int {
    return 2 * idx + 1;
  });
  return 0; // break here
}

