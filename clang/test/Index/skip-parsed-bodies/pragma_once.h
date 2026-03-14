#pragma once

extern int some_val;

template <typename T>
struct TS {
  int tsmeth() {
    ++some_val; return undef_tsval;
  }
};
