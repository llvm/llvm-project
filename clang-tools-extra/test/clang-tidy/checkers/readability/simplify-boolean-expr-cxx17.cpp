// RUN: clang-tidy %s -checks='-*,readability-simplify-boolean-expr' -- -std=c++17 | count 0
struct RAII {};
bool foo(bool Cond) {
  bool Result;

  if (RAII Object; Cond)
    Result = true;
  else
    Result = false;

  if (bool X = Cond; X)
    Result = true;
  else
    Result = false;

  if (bool X = Cond; X)
    return true;
  return false;
}
