// RUN: clang-tidy %s -checks='-*,readability-braces-around-statements' -- -std=c++2b | count 0

constexpr void handle(bool) {}

constexpr void shouldPass() {
  if consteval {
    handle(true);
  } else {
    handle(false);
  }
}

constexpr void shouldPassNegated() {
  if !consteval {
    handle(false);
  } else {
    handle(true);
  }
}

constexpr void shouldPassSimple() {
  if consteval {
    handle(true);
  }
}

void run() {
    shouldPass();
    shouldPassNegated();
    shouldPassSimple();
}
