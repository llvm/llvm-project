// RUN: %clang_cc1 -ast-print -std=c++17 %s | FileCheck %s

struct S {
template<typename ... T>
void test1(int i, T... t) {
{
  auto lambda = [i]{};
  //CHECK: [i] {
}
{
  auto lambda = [=]{};
  //CHECK: [=] {
}
{
  auto lambda = [&]{};
  //CHECK: [&] {
}
{
  auto lambda = [k{i}] {};
  //CHECK: [k{i}] {
}
{
  auto lambda = [k(i)] {};
  //CHECK: [k(i)] {
}
{
  auto lambda = [k = i] {};
  //CHECK: [k = i] {
}
{
  auto lambda = [t..., i]{};
  //CHECK: [t..., i] {
}
{
  auto lambda = [&t...]{};
  //CHECK: [&t...] {
}
{
  auto lambda = [this, &t...]{};
  //CHECK: [this, &t...] {
}
{
  auto lambda = [t..., this]{};
  //CHECK: [t..., this] {
}
{
  auto lambda = [k(t...)] {};
  //CHECK: [k(t...)] {
}
{
  auto lambda = [k{t...}] {};
  //CHECK: [k{t...}] {
}
}

};