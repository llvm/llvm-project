//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

struct Pair {
	int x;
	int y;
	
	Pair(int _x, int _y) : x(_x), y(_y) {}	
};

int main() {
	Pair p1(3,-3);
	return p1.x + p1.y; // Set break point at this line.
}
