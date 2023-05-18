@import recursive_visibility_b;
template<template<typename T> class Y> void g() {
  f(typename Y<A1_Inner::X>::type{});
  f(typename Y<A2_More_Inner::X>::type{});
}
