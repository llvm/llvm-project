// RUN: not clang-tidy %s -checks='-*,misc-header-include-cycle'

#include "self-include.cpp"
