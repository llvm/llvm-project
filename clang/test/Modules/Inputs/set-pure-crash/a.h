#pragma once

struct Tag {};

template <typename T>
class Base {
public:
  virtual void func() = 0;
};

Base<Tag> bar();
