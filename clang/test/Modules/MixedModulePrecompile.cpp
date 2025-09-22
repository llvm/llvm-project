// Tests mixed usage of precompiled headers and modules.
//
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -x c++-header -emit-pch %t/a.hpp \
// RUN: -o %t/a.pch

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/Part1.cppm \
// RUN: -include-pch %t/a.pch -o %t/Part1.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/Part2.cppm \
// RUN: -include-pch %t/a.pch -o %t/Part2.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/Part3.cppm \
// RUN: -include-pch %t/a.pch -o %t/Part3.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/Part4.cppm \
// RUN: -include-pch %t/a.pch -o %t/Part4.pcm

// RUN: %clang_cc1 -std=c++20 -emit-module-interface \
// RUN: -fmodule-file=mod:part1=%t/Part1.pcm \
// RUN: -fmodule-file=mod:part2=%t/Part2.pcm \
// RUN: -fmodule-file=mod:part3=%t/Part3.pcm \
// RUN: -fmodule-file=mod:part4=%t/Part4.pcm \
// RUN: %t/Mod.cppm \
// RUN: -include-pch %t/a.pch -o %t/Mod.pcm

// RUN: %clang_cc1 -std=c++20 -emit-obj \
// RUN: -main-file-name Mod.cppm \
// RUN: -fmodule-file=mod:part1=%t/Part1.pcm \
// RUN: -fmodule-file=mod:part2=%t/Part2.pcm \
// RUN: -fmodule-file=mod:part3=%t/Part3.pcm \
// RUN: -fmodule-file=mod:part4=%t/Part4.pcm \
// RUN: -x pcm %t/Mod.pcm \
// RUN: -include-pch %t/a.pch -o %t/Mod.o


//--- a.hpp
#pragma once

class a {
  virtual ~a();
  a() {}
};

//--- Part1.cppm
export module mod:part1;

//--- Part2.cppm
export module mod:part2;

//--- Part3.cppm
export module mod:part3;

//--- Part4.cppm
export module mod:part4;

//--- Mod.cppm
export module mod;
export import :part1;
export import :part2;
export import :part3;
export import :part4;

