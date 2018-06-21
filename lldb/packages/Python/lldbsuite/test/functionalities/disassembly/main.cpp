//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

int
sum (int a, int b)
{
    int result = a + b;
    return result;
}

int
main(int argc, char const *argv[])
{

    int array[3];

    array[0] = sum (1238, 78392); // Set a breakpoint here
    array[1] = sum (379265, 23674);
    array[2] = sum (872934, 234);

    return 0;
}
