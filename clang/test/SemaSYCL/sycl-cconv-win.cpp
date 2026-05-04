// RUN: %clang_cc1 -isystem %S/Inputs/ -fsycl-is-device -triple amdgcn-amd-hsa -aux-triple x86_64-pc-windows-msvc -fsyntax-only -verify %s

// expected-no-diagnostics

#include <vectorcall.hpp>
