// RUN: not clang-tidy %s -checks='-*,misc-header-include-cycle'

#include "header-include-cycle.self.cpp"
